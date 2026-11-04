"""Entrenamiento, optimización con Optuna y registro de experimentos (M07/M08/F6-F9)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.db.base import utcnow
from app.ml.features import DatasetBundle, build_dataset, load_panel
from app.ml.metrics import evaluate_all
from app.ml.models.base import ModelData
from app.ml.models.registry import SEARCH_SPACES, create_model, get_algorithm
from app.ml.reproducibility import configure_threads, set_reproducible_seed

logger = logging.getLogger("dengai.ml.trainer")

_RUNNING: Dict[int, threading.Thread] = {}
_LOCK = threading.Lock()


def _as_model_data(bundle: DatasetBundle, split: str) -> ModelData:
    names = bundle.feature_names
    if split == "train":
        X, seq, grid, cell, y = bundle.X_train, bundle.seq_train, bundle.grid_train, bundle.cell_train, bundle.y_train
    elif split == "val":
        X, seq, grid, cell, y = bundle.X_val, bundle.seq_val, bundle.grid_val, bundle.cell_val, bundle.y_val
    else:
        X, seq, grid, cell, y = bundle.X_test, bundle.seq_test, bundle.grid_test, bundle.cell_test, bundle.y_test
    return ModelData(X=X, seq=seq, grid=grid, cell=cell, y=y, feature_names=names)


def _suggest(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec.get("type")
        if kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
    return params


def _artifact_dir() -> Path:
    path = settings.models_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bundle(algorithm: str, version: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    import joblib

    path = _artifact_dir() / f"{algorithm}-{version}.joblib"
    joblib.dump(payload, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), checksum


def load_bundle(path: str) -> Dict[str, Any]:
    import joblib

    return joblib.load(path)


def save_predictions(run_id: str, horizon: int, y_true, y_pred, meta) -> str:
    path = _artifact_dir() / f"preds-{run_id}-h{horizon}.npz"
    np.savez_compressed(
        path,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        district=np.asarray(meta["district_code"].astype(str).tolist()),
        target_week=np.asarray(meta["target_week"].astype(str).tolist()),
    )
    return str(path)


class ExperimentRunner:
    """Ejecuta un experimento completo en un hilo aparte con progreso persistido."""

    def __init__(self, experiment_id: int, options: Dict[str, Any]) -> None:
        self.experiment_id = experiment_id
        self.options = options

    # ------------------------------------------------------------------ helpers
    def _update(self, db, **fields) -> None:
        from app.models.ml import Experiment

        experiment = db.get(Experiment, self.experiment_id)
        if experiment is None:
            return
        for key, value in fields.items():
            setattr(experiment, key, value)
        db.commit()

    def run(self) -> None:
        from app.db.session import session_scope
        from app.models.ml import Experiment, MetricResult, ModelVersion, StatisticalTest
        from app.services import audit as audit_service

        configure_threads()
        set_reproducible_seed(int(self.options.get("seed", 42)))
        started = time.perf_counter()

        with session_scope() as db:
            experiment = db.get(Experiment, self.experiment_id)
            if experiment is None:
                return
            algorithm = experiment.algorithm
            horizons = self.options.get("horizons") or [1]
            n_trials = int(self.options.get("n_trials", settings.ML_DEFAULT_TRIALS))
            user_email = experiment.created_by

        try:
            from app.db.session import session_scope as _scope

            with _scope() as db:
                exp = db.get(Experiment, self.experiment_id)
                exp.status = "RUNNING"
                exp.stage = "cargando_datos"
                exp.progress = 0.02
                db.commit()

                panel = load_panel(db)
                meta = get_algorithm(algorithm)
                space = SEARCH_SPACES.get(algorithm, {})
                max_epochs = int(self.options.get("max_epochs", 40))

            horizon_metrics: Dict[str, Any] = {}
            best_params_per_horizon: Dict[str, Any] = {}
            total_steps = max(1, len(horizons))
            for idx, horizon in enumerate(horizons):
                base_progress = idx / total_steps
                with _scope() as db:
                    exp = db.get(Experiment, self.experiment_id)
                    exp.stage = f"horizonte_{horizon}_datos"
                    exp.progress = round(base_progress + 0.05 / total_steps, 4)
                    db.commit()
                    bundle = build_dataset(
                        panel,
                        horizon=int(horizon),
                        validation_weeks=settings.ML_VALIDATION_WEEKS,
                        test_weeks=settings.ML_TEST_WEEKS,
                        strategy="geographic" if self.options.get("strategy") == "geographic" else "walk_forward",
                    )

                train_data = _as_model_data(bundle, "train")
                val_data = _as_model_data(bundle, "val")
                test_data = _as_model_data(bundle, "test")

                best_params = self._optimize(
                    algorithm, space, train_data, val_data, n_trials, base_progress, total_steps, max_epochs, idx
                )

                # Reentreno con entrenamiento + validación (el conjunto de prueba nunca participa).
                combined = _concat_data(train_data, val_data)
                with _scope() as db:
                    exp = db.get(Experiment, self.experiment_id)
                    exp.stage = f"horizonte_{horizon}_entrenamiento_final"
                    exp.progress = round(base_progress + 0.8 / total_steps, 4)
                    db.commit()

                model = create_model(algorithm)
                model.fit(combined, val_data, params=best_params, epochs=max_epochs)

                y_test_pred = model.predict(test_data)
                y_val_pred = model.predict(val_data)
                interval = _residual_quantiles(bundle.y_val, y_val_pred, level=0.9)
                lower_test = np.clip(y_test_pred + interval["low"], 0, None)
                upper_test = np.clip(y_test_pred + interval["high"], 0, None)
                metrics = evaluate_all(
                    bundle.y_test,
                    y_test_pred,
                    lower=lower_test,
                    upper=upper_test,
                    y_insample=bundle.y_train[-120:],
                    outbreak_threshold=float(np.quantile(bundle.y_train, 0.9)),
                    interval_level=0.9,
                )
                horizon_metrics[str(horizon)] = {
                    "metrics": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)) for k, v in metrics.items()},
                    "best_params": best_params,
                    "n_train": int(len(bundle.y_train)),
                    "n_val": int(len(bundle.y_val)),
                    "n_test": int(len(bundle.y_test)),
                    "notes": bundle.notes,
                }
                best_params_per_horizon[str(horizon)] = best_params

                with _scope() as db:
                    exp = db.get(Experiment, self.experiment_id)
                    if exp is not None:
                        exp.trials_completed = n_trials
                    for metric_name, value in metrics.items():
                        if value is None or not np.isfinite(value):
                            continue
                        name = metric_name.replace("_ci_lower", "").replace("_ci_upper", "")
                        if metric_name.endswith("_ci_lower") or metric_name.endswith("_ci_upper"):
                            continue
                        db.add(
                            MetricResult(
                                experiment_id=self.experiment_id, algorithm=algorithm, split="test",
                                horizon_weeks=int(horizon), metric=metric_name, value=float(value),
                                ci_lower=float(metrics["MAE_ci_lower"]) if metric_name == "MAE" and metrics.get("MAE_ci_lower") is not None else None,
                                ci_upper=float(metrics["MAE_ci_upper"]) if metric_name == "MAE" and metrics.get("MAE_ci_upper") is not None else None,
                                n_observations=int(len(bundle.y_test)),
                            )
                        )
                    db.commit()
                    run_id_row = db.get(Experiment, self.experiment_id)
                    run_id = run_id_row.run_id

                pred_path = save_predictions(run_id, int(horizon), bundle.y_test, y_test_pred, bundle.meta_test)
                artifact_payload = {
                    "algorithm": algorithm,
                    "horizon": int(horizon),
                    "model": model,
                    "params": best_params,
                    "feature_names": bundle.feature_names,
                    "scaler_mean": bundle.scaler_mean,
                    "scaler_std": bundle.scaler_std,
                    "grid_shape": bundle.grid_shape,
                    "district_grid": bundle.notes.get("district_grid"),
                    "sequence_window": bundle.notes.get("sequence_window", 12),
                    "interval": interval,
                    "trained_at": utcnow().isoformat(),
                    "is_synthetic_data": bundle.is_synthetic,
                    "predictions_path": pred_path,
                }
                version = f"h{int(horizon)}-{run_id[:8]}"
                path, checksum = save_bundle(algorithm, version, artifact_payload)

                with _scope() as db:
                    db.add(
                        ModelVersion(
                            experiment_id=self.experiment_id, algorithm=algorithm, version=version,
                            artifact_path=path, artifact_checksum=checksum, status="CANDIDATE",
                            training_weeks=int(len(bundle.meta_train["target_week"].unique()) if len(bundle.meta_train) else 0),
                            feature_names={"columns": bundle.feature_names, "horizon": int(horizon)},
                            metrics={k: (float(v) if v is not None and np.isfinite(v) else None) for k, v in metrics.items()},
                            hyperparameters=best_params,
                            notes=f"Entrenado sobre {'datos sintéticos' if bundle.is_synthetic else 'datos cargados'}; horizonte {horizon} semanas.",
                        )
                    )
                    db.commit()

            duration = time.perf_counter() - started
            with _scope() as db:
                exp = db.get(Experiment, self.experiment_id)
                exp.status = "COMPLETED"
                exp.stage = "finalizado"
                exp.progress = 1.0
                exp.metrics = horizon_metrics
                exp.best_trial = {"params_by_horizon": best_params_per_horizon}
                exp.finished_at = utcnow()
                exp.duration_seconds = round(duration, 2)
                exp.hyperparameters = best_params_per_horizon
                db.commit()

                audit_service.record(
                    db,
                    event_type=audit_service.AuditEventType.MODEL_EXECUTION,
                    module="M07",
                    action="train_experiment",
                    result="SUCCESS",
                    user_email=user_email,
                    user_role="",
                    resource_type="experiment",
                    resource_id=self.experiment_id,
                    description=f"Experimento {algorithm} completado con {n_trials} trials y horizontes {horizons}.",
                    duration_ms=int(duration * 1000),
                    context={"algorithm": algorithm, "horizons": horizons, "n_trials": n_trials},
                )

            self._track_mlflow(algorithm, horizon_metrics, best_params_per_horizon, n_trials)
        except Exception as exc:  # pragma: no cover - se reporta al usuario
            logger.exception("Fallo el experimento %s", self.experiment_id)
            with _scope() as db:
                exp = db.get(Experiment, self.experiment_id)
                if exp is not None:
                    exp.status = "FAILED"
                    exp.stage = "error"
                    exp.error_message = f"{type(exc).__name__}: {exc}"[:2000]
                    exp.finished_at = utcnow()
                    exp.duration_seconds = round(time.perf_counter() - started, 2)
                    db.commit()
                    audit_service.record(
                        db,
                        event_type=audit_service.AuditEventType.MODEL_EXECUTION,
                        module="M07",
                        action="train_experiment",
                        result="ERROR",
                        user_email=exp.created_by,
                        resource_type="experiment",
                        resource_id=self.experiment_id,
                        description=f"Falló el experimento: {exc}"[:500],
                    )

    def _optimize(
        self,
        algorithm: str,
        space: Dict[str, Any],
        train_data: ModelData,
        val_data: ModelData,
        n_trials: int,
        base_progress: float,
        total_steps: int,
        max_epochs: int,
        horizon_idx: int,
    ) -> Dict[str, Any]:
        import optuna

        from app.db.session import session_scope

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            params = _suggest(trial, space)
            model = create_model(algorithm)
            model.fit(train_data, val_data, params=params, trial=trial, epochs=max_epochs)
            from sklearn.metrics import mean_absolute_error

            pred = model.predict(val_data)
            return float(mean_absolute_error(val_data.y, pred))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42 + horizon_idx),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(3, n_trials // 4), n_warmup_steps=5),
        )
        callback_state = {"done": 0}

        def _callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
            from app.models.ml import Experiment

            callback_state["done"] = len(study.trials)
            with session_scope() as db:
                exp = db.get(Experiment, self.experiment_id)
                if exp is not None:
                    exp.trials_completed = callback_state["done"]
                    exp.stage = f"optuna_{callback_state['done']}/{n_trials}"
                    exp.progress = round(
                        base_progress + (0.6 * callback_state["done"] / max(1, n_trials) + 0.1) / total_steps, 4
                    )
                    db.commit()

        study.optimize(objective, n_trials=n_trials, callbacks=[_callback], show_progress_bar=False)
        return dict(study.best_params)

    def _track_mlflow(self, algorithm: str, metrics: Dict[str, Any], params: Dict[str, Any], n_trials: int) -> None:
        """Registra el experimento en MLflow si está instalado; si no, deja constancia."""
        from app.db.session import session_scope
        from app.models.ml import Experiment

        try:
            import mlflow  # type: ignore

            tracking_uri = settings.MLFLOW_TRACKING_URI or (settings.data_dir / "mlruns").as_uri()
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("SP01-Dengue-Intelligence")
            with mlflow.start_run(run_name=f"{algorithm}-{self.experiment_id}") as run:
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("n_trials", n_trials)
                for key, value in (params.get("1") or {}).items():
                    mlflow.log_param(f"p_{key}", value)
                for horizon, payload in metrics.items():
                    for name, value in payload.get("metrics", {}).items():
                        if value is not None:
                            mlflow.log_metric(f"h{horizon}_{name}", float(value))
                mlflow.set_tag("data", "synthetic" if True else "real")
                mlflow.set_tag("platform", settings.APP_SHORT_NAME)
                run_id = run.info.run_id
            with session_scope() as db:
                exp = db.get(Experiment, self.experiment_id)
                if exp is not None:
                    exp.mlflow_run_id = run_id
                    db.commit()
        except ImportError:
            logger.info("MLflow no está instalado: se conserva el registro interno de experimentos.")
        except Exception:
            logger.exception("No se pudo registrar en MLflow")


class _NoopSession:
    """Marcador sin uso: conservado para compatibilidad de importaciones antiguas."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        return None


def _concat_data(a: ModelData, b: ModelData) -> ModelData:
    def _cat(x, y):
        if x is None or y is None:
            return None
        return np.concatenate([x, y], axis=0)

    return ModelData(
        X=_cat(a.X, b.X),
        seq=_cat(a.seq, b.seq),
        grid=_cat(a.grid, b.grid),
        cell=_cat(a.cell, b.cell),
        y=_cat(a.y, b.y),
        feature_names=a.feature_names,
    )


def _residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray, *, level: float = 0.9) -> Dict[str, float]:
    """Intervalos empíricos a partir de los residuales de validación (nunca del conjunto de prueba)."""
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    alpha = (1 - level) / 2
    return {
        "low": float(np.quantile(residual, alpha)),
        "high": float(np.quantile(residual, 1 - alpha)),
        "level": level,
    }


def submit_experiment(experiment_id: int, options: Dict[str, Any]) -> bool:
    with _LOCK:
        if experiment_id in _RUNNING and _RUNNING[experiment_id].is_alive():
            return False
        runner = ExperimentRunner(experiment_id, options)
        thread = threading.Thread(target=runner.run, name=f"experiment-{experiment_id}", daemon=True)
        _RUNNING[experiment_id] = thread
        thread.start()
        return True


def is_running(experiment_id: int) -> bool:
    thread = _RUNNING.get(experiment_id)
    return bool(thread and thread.is_alive())
