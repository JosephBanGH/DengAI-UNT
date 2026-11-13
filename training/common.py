"""Utilidades compartidas para los scripts CLI de entrenamiento (Fase 2).

Incluye:
- Resolución de rutas de módulos (``libs/`` y ``backend/``).
- Parsing consistente de argumentos CLI (``--trials``, ``--epochs``, ``--horizon``,
  ``--seed``, ``--gpu``, ``--output-dir``).
- Carga homogénea del panel desde Fase 1 (``real`` / ``auto`` / ``synthetic``).
- Validación estricta de dimensiones de matrices/tensores y aserciones antifuga.
- Protección explícita contra sobreescritura de ``data/models/*.joblib``.
- Escritor de telemetría JSONL por ``run_id`` en ``data/reports/telemetry/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_DIR / "backend"
LIBS_DIR = PROJECT_DIR / "libs"
DEFAULT_OUTPUT_DIR = "data/models/phase2"
DEFAULT_TELEMETRY_DIR = "data/reports/telemetry"


def _is_site_packages(entry: str) -> bool:
    return entry.lower().replace("\\", "/").rstrip("/").endswith("site-packages")


def prepare_sys_path() -> str:
    """Deja ``backend/`` y ``training/`` importables en entorno local o contenedor."""
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    if LIBS_DIR.is_dir():
        if str(LIBS_DIR) not in sys.path:
            sys.path.insert(0, str(LIBS_DIR))
        sys.path[:] = [entry for entry in sys.path if not _is_site_packages(entry)]
        return "libs-local"
    return "site-packages"


prepare_sys_path()

from app.core.config import settings  # noqa: E402
from app.db.base import utcnow  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.ml.features import DatasetBundle, PanelResult, build_dataset, load_panel  # noqa: E402
from app.ml.metrics import evaluate_all  # noqa: E402
from app.ml.models.base import ModelData  # noqa: E402
from app.ml.models.registry import SEARCH_SPACES, create_model, get_algorithm  # noqa: E402
from app.ml.reproducibility import configure_threads, set_reproducible_seed  # noqa: E402
from app.ml.trainer import _as_model_data, _concat_data, _residual_quantiles, _suggest  # noqa: E402

logger = logging.getLogger("dengai.training.cli")


def configure_cli_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def resolve_safe_output_dir(output_dir: str | Path) -> Path:
    """Resuelve el directorio de salida impidiendo sobreescribir ``data/models/`` raíz."""
    raw = Path(output_dir)
    target = raw if raw.is_absolute() else (PROJECT_DIR / raw)
    resolved = target.resolve()
    protected_root = (PROJECT_DIR / "data" / "models").resolve()
    settings_root = settings.models_dir.resolve()
    if resolved in {protected_root, settings_root}:
        raise ValueError(
            "Por seguridad científica, --output-dir no puede apuntar directamente a 'data/models/' "
            "(protege los artefactos consolidados del manuscrito). Use 'data/models/phase2/' u otro subdirectorio."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_telemetry_dir(telemetry_dir: str | Path) -> Path:
    raw = Path(telemetry_dir)
    target = raw if raw.is_absolute() else (PROJECT_DIR / raw)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _get_psutil():
    """Importa psutil desde libs/ o site-packages sin alterar la prioridad de libs/."""
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        import site

        saved = list(sys.path)
        try:
            for sp in site.getsitepackages():
                if sp not in sys.path:
                    sys.path.append(sp)
            import psutil  # type: ignore

            return psutil
        except Exception:
            return None
        finally:
            sys.path[:] = saved


def collect_hardware_snapshot() -> Dict[str, Any]:
    """Captura uso de CPU, RAM y estado seguro de GPU CUDA."""
    cpu_pct: Optional[float] = None
    ram_pct: Optional[float] = None
    ram_used_mb: Optional[float] = None
    psutil = _get_psutil()
    if psutil is not None:
        try:
            cpu_pct = round(float(psutil.cpu_percent(interval=None)), 2)
            mem = psutil.virtual_memory()
            ram_pct = round(float(mem.percent), 2)
            ram_used_mb = round(float(mem.used) / (1024 ** 2), 2)
        except Exception:
            pass

    gpu_available = False
    gpu_vram_mb: Optional[float] = None
    gpu_name: Optional[str] = None
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_vram_mb = round(float(torch.cuda.memory_allocated(0)) / (1024 ** 2), 2)
            gpu_name = str(torch.cuda.get_device_name(0))
    except Exception:
        gpu_available = False
        gpu_vram_mb = None

    return {
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "ram_used_mb": ram_used_mb,
        "gpu_available": gpu_available,
        "gpu_vram_mb": gpu_vram_mb,
        "gpu_name": gpu_name,
    }


def resolve_device_request(request_gpu: bool) -> Dict[str, Any]:
    """Verifica disponibilidad de GPU sin fallar en entornos CPU-only."""
    cuda_ok = False
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if request_gpu and not cuda_ok:
        logger.warning(
            "Se solicitó --gpu, pero torch.cuda.is_available() es False. "
            "Se ejecutará de forma segura en CPU."
        )
    device = "cuda" if cuda_ok else "cpu"
    return {
        "gpu_requested": bool(request_gpu),
        "gpu_available": cuda_ok,
        "device": device,
    }


class TelemetryWriter:
    """Escritor JSONL por ``run_id`` en ``data/reports/telemetry/{run_id}.jsonl``."""

    def __init__(
        self,
        run_id: str,
        algorithm: str,
        telemetry_dir: str | Path = DEFAULT_TELEMETRY_DIR,
    ) -> None:
        self.run_id = run_id
        self.algorithm = algorithm
        self.directory = resolve_telemetry_dir(telemetry_dir)
        self.path = self.directory / f"{run_id}.jsonl"
        self._started = time.perf_counter()
        psutil = _get_psutil()
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    def emit(
        self,
        event: str,
        *,
        stage: str = "",
        trial: Optional[int] = None,
        n_trials: Optional[int] = None,
        epoch: Optional[int] = None,
        max_epochs: Optional[int] = None,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_mae: Optional[float] = None,
        best_score: Optional[float] = None,
        progress: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        elapsed_s = round(time.perf_counter() - self._started, 3)
        eta_s: Optional[float] = None
        if progress is not None and 0.0 < progress < 1.0:
            eta_s = round(elapsed_s * (1.0 - progress) / progress, 2)
        elif progress is not None and progress >= 1.0:
            eta_s = 0.0

        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "source": "cli",
            "algorithm": self.algorithm,
            "event": event,
            "stage": stage,
            "trial": trial,
            "n_trials": n_trials,
            "epoch": epoch,
            "max_epochs": max_epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "best_score": best_score,
            "progress": round(float(progress), 4) if progress is not None else None,
            "elapsed_s": elapsed_s,
            "eta_s": eta_s,
            **collect_hardware_snapshot(),
        }
        if extra:
            record["extra"] = extra

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        return record


def build_base_parser(
    description: str,
    *,
    allow_n_estimators: bool = False,
    allow_models_selection: bool = False,
) -> argparse.ArgumentParser:
    """Construye el parser CLI estándar de los scripts de Fase 2."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--trials", type=int, default=20, help="Número de trials de Optuna (defecto: 20).")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Épocas máximas para modelos neuronales (o n_estimators en XGBoost si se indica).",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte de pronóstico en semanas (defecto: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria reproducible (defecto: 42).")
    parser.add_argument("--gpu", action="store_true", help="Solicita aceleración GPU CUDA si está disponible.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida para artefactos Fase 2 (defecto: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--data-mode",
        choices=["auto", "real", "synthetic"],
        default="auto",
        help="Modo de datos para load_panel (defecto: auto).",
    )
    parser.add_argument(
        "--strategy",
        choices=["walk_forward", "geographic"],
        default="walk_forward",
        help="Estrategia de partición (defecto: walk_forward).",
    )
    parser.add_argument(
        "--validation-weeks",
        type=int,
        default=None,
        help="Semanas de validación (defecto: settings.ML_VALIDATION_WEEKS).",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        default=None,
        help="Semanas de prueba intocables (defecto: settings.ML_TEST_WEEKS).",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=None,
        help="Máximo de rezagos semanales (defecto: settings.ML_MAX_LAG).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Identificador explícito de corrida.")
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default=DEFAULT_TELEMETRY_DIR,
        help=f"Directorio para telemetría JSONL (defecto: {DEFAULT_TELEMETRY_DIR}).",
    )
    if allow_n_estimators:
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=None,
            help="Número fijo de árboles para XGBoost (tiene prioridad sobre --epochs).",
        )
    if allow_models_selection:
        parser.add_argument(
            "--models",
            nargs="+",
            default=["all"],
            help="Modelos a optimizar (xgboost, lstm, convlstm, cnn_lstm, convlstm_attention o all).",
        )
        parser.add_argument(
            "--storage",
            type=str,
            default=None,
            help="URI opcional de almacenamiento Optuna (p. ej. sqlite:///data/reports/phase2/optuna.db).",
        )
        parser.add_argument(
            "--study-prefix",
            type=str,
            default="phase2",
            help="Prefijo para nombres de estudio Optuna (defecto: phase2).",
        )
    return parser


def load_training_panel(data_mode: str = "auto") -> PanelResult:
    """Carga el panel homogéneo (``real`` / ``auto`` / ``synthetic``) desde la BD."""
    with SessionLocal() as db:
        panel = load_panel(db, mode=data_mode)
    if panel.panel.empty:
        raise ValueError("El panel cargado está vacío.")
    return panel


def validate_bundle_dimensions(
    bundle: DatasetBundle,
    algorithm: str,
    *,
    strategy: str = "walk_forward",
) -> Dict[str, Any]:
    """Valida dimensiones de tensores/matrices y reglas estrictas antifuga."""
    n_features = len(bundle.feature_names)
    if n_features == 0:
        raise ValueError("No hay columnas predictoras en el DatasetBundle.")

    splits = {
        "train": (bundle.X_train, bundle.y_train, bundle.seq_train, bundle.grid_train, bundle.cell_train, bundle.meta_train),
        "val": (bundle.X_val, bundle.y_val, bundle.seq_val, bundle.grid_val, bundle.cell_val, bundle.meta_val),
        "test": (bundle.X_test, bundle.y_test, bundle.seq_test, bundle.grid_test, bundle.cell_test, bundle.meta_test),
    }

    shapes: Dict[str, Any] = {"n_features": n_features, "grid_shape": list(bundle.grid_shape)}
    for split_name, (X, y, seq, grid, cell, meta) in splits.items():
        n = len(y)
        if n == 0:
            raise ValueError(f"La partición '{split_name}' quedó vacía.")
        if X.ndim != 2 or X.shape != (n, n_features):
            raise ValueError(f"Dimensión inválida en X_{split_name}: {X.shape}, esperado {(n, n_features)}.")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"X_{split_name} contiene NaN o Inf tras la estandarización.")
        if not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError(f"y_{split_name} contiene valores inválidos o negativos.")

        if seq is None or seq.ndim != 3 or seq.shape[0] != n or seq.shape[2] != n_features:
            raise ValueError(
                f"Dimensión inválida en seq_{split_name}: {None if seq is None else seq.shape}, "
                f"esperado ({n}, T, {n_features})."
            )
        if not np.all(np.isfinite(seq)):
            raise ValueError(f"seq_{split_name} contiene NaN o Inf.")

        rows, cols = bundle.grid_shape
        if grid is None or grid.ndim != 5 or grid.shape[0] != n or grid.shape[2:4] != (rows, cols):
            raise ValueError(
                f"Dimensión inválida en grid_{split_name}: {None if grid is None else grid.shape}, "
                f"esperado ({n}, T, {rows}, {cols}, C)."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"grid_{split_name} contiene NaN o Inf.")

        if cell is None or cell.ndim != 1 or cell.shape[0] != n:
            raise ValueError(
                f"Dimensión inválida en cell_{split_name}: {None if cell is None else cell.shape}, esperado ({n},)."
            )
        if np.any(cell < 0) or np.any(cell >= rows * cols):
            raise ValueError(f"Índices de celda fuera de rango en cell_{split_name}.")

        if len(meta) != n:
            raise ValueError(f"meta_{split_name} tiene {len(meta)} filas y se esperaban {n}.")

        shapes[split_name] = {
            "n": int(n),
            "X": list(X.shape),
            "seq": list(seq.shape),
            "grid": list(grid.shape),
            "cell": list(cell.shape),
        }

    # Aserciones antifuga
    if strategy == "geographic":
        tr_d = set(bundle.meta_train["district_code"])
        va_d = set(bundle.meta_val["district_code"])
        te_d = set(bundle.meta_test["district_code"])
        if not tr_d.isdisjoint(va_d) or not tr_d.isdisjoint(te_d) or not va_d.isdisjoint(te_d):
            raise AssertionError("Fuga espacial detectada: distritos compartidos entre particiones.")
    else:
        tr_origin_max = bundle.meta_train["origin_week"].max()
        tr_target_max = bundle.meta_train["target_week"].max()
        va_origin_min = bundle.meta_val["origin_week"].min()
        va_origin_max = bundle.meta_val["origin_week"].max()
        va_target_max = bundle.meta_val["target_week"].max()
        te_origin_min = bundle.meta_test["origin_week"].min()
        if not (tr_origin_max < va_origin_min <= va_origin_max < te_origin_min):
            raise AssertionError("Fuga temporal detectada en origin_week entre train/val/test.")
        if not (tr_target_max < va_origin_min and va_target_max < te_origin_min):
            raise AssertionError("Fuga temporal detectada: target_week cruza la frontera de validación/prueba.")
        if not (bundle.meta_test["target_week"] > bundle.meta_test["origin_week"]).all():
            raise AssertionError("Fuga temporal detectada: target_week <= origin_week.")

    algo_meta = get_algorithm(algorithm)
    logger.info(
        "Dimensiones validadas [%s | %s]: train=%s, val=%s, test=%s, features=%d",
        algo_meta["name"],
        strategy,
        shapes["train"]["X"],
        shapes["val"]["X"],
        shapes["test"]["X"],
        n_features,
    )
    return shapes


def resolve_xgb_n_estimators(epochs: Optional[int], n_estimators: Optional[int]) -> Optional[int]:
    """Resuelve ``n_estimators`` explícito para XGBoost (prioriza ``--n-estimators`` sobre ``--epochs``)."""
    if n_estimators is not None:
        return max(1, int(n_estimators))
    if epochs is not None:
        return max(1, int(epochs))
    return None


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    cleaned: Dict[str, Optional[float]] = {}
    for k, v in metrics.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            cleaned[k] = None
        else:
            cleaned[k] = round(float(v), 4)
    return cleaned


def run_training_pipeline(
    algorithm: str,
    args: argparse.Namespace,
    *,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """Ejecuta optimización Optuna, reentreno final, evaluación y guardado aislado."""
    import joblib
    import optuna
    from sklearn.metrics import mean_absolute_error

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    horizon = int(args.horizon)
    if horizon < 1:
        raise ValueError(f"--horizon debe ser >= 1; recibido {horizon}.")
    trials = max(1, int(args.trials))
    seed = int(args.seed)
    effective_epochs = int(args.epochs) if args.epochs is not None else 40
    if effective_epochs < 1:
        raise ValueError(f"--epochs debe ser >= 1; recibido {effective_epochs}.")

    xgb_fixed_estimators: Optional[int] = None
    if algorithm == "xgboost":
        xgb_fixed_estimators = resolve_xgb_n_estimators(
            args.epochs,
            getattr(args, "n_estimators", None),
        )

    output_dir = resolve_safe_output_dir(args.output_dir)
    run_id = args.run_id or f"p2-{algorithm}-h{horizon}-{uuid.uuid4().hex[:8]}"
    telemetry = TelemetryWriter(run_id, algorithm, telemetry_dir=args.telemetry_dir)

    configure_threads()
    set_reproducible_seed(seed)
    hw_device = resolve_device_request(bool(args.gpu))

    telemetry.emit(
        "start",
        stage="inicializando",
        n_trials=trials,
        max_epochs=xgb_fixed_estimators if (algorithm == "xgboost" and xgb_fixed_estimators) else effective_epochs,
        progress=0.02,
        extra={
            "horizon": horizon,
            "seed": seed,
            "data_mode": args.data_mode,
            "strategy": args.strategy,
            "output_dir": str(output_dir),
            **hw_device,
        },
    )

    try:
        panel = load_training_panel(args.data_mode)
        validation_weeks = int(args.validation_weeks or settings.ML_VALIDATION_WEEKS)
        test_weeks = int(args.test_weeks or settings.ML_TEST_WEEKS)
        max_lag = int(args.max_lag or settings.ML_MAX_LAG)

        bundle = build_dataset(
            panel,
            horizon=horizon,
            validation_weeks=validation_weeks,
            test_weeks=test_weeks,
            strategy=args.strategy,
            max_lag=max_lag,
        )
        shapes = validate_bundle_dimensions(bundle, algorithm, strategy=args.strategy)

        train_data = _as_model_data(bundle, "train")
        val_data = _as_model_data(bundle, "val")
        test_data = _as_model_data(bundle, "test")

        space = SEARCH_SPACES.get(algorithm, {})
        best_tracker = {"score": float("inf")}

        def objective(trial: "optuna.Trial") -> float:
            trial_num = trial.number + 1
            params = _suggest(trial, space)
            if algorithm == "xgboost":
                params["random_state"] = seed
                if xgb_fixed_estimators is not None:
                    params["n_estimators"] = xgb_fixed_estimators
            else:
                params["seed"] = seed

            telemetry.emit(
                "trial_start",
                stage=f"optuna_trial_{trial_num}/{trials}",
                trial=trial_num,
                n_trials=trials,
                max_epochs=params.get("n_estimators") if algorithm == "xgboost" else effective_epochs,
                best_score=None if not np.isfinite(best_tracker["score"]) else round(best_tracker["score"], 5),
                progress=0.05 + 0.65 * ((trial_num - 1) / trials),
                extra={"params": params},
            )

            model = create_model(algorithm)
            model.fit(train_data, val_data, params=params, trial=trial, epochs=effective_epochs)

            for rec in model.training_history():
                telemetry.emit(
                    "epoch",
                    stage=f"optuna_trial_{trial_num}/{trials}",
                    trial=trial_num,
                    n_trials=trials,
                    epoch=int(rec["epoch"]),
                    max_epochs=effective_epochs,
                    train_loss=rec.get("train_loss"),
                    val_loss=rec.get("val_mae"),
                    val_mae=rec.get("val_mae"),
                    best_score=None if not np.isfinite(best_tracker["score"]) else round(best_tracker["score"], 5),
                    progress=0.05 + 0.65 * ((trial_num - 0.5) / trials),
                )

            pred = model.predict(val_data)
            val_mae = float(mean_absolute_error(val_data.y, pred))
            if val_mae < best_tracker["score"]:
                best_tracker["score"] = val_mae

            telemetry.emit(
                "trial_end",
                stage=f"optuna_trial_{trial_num}/{trials}",
                trial=trial_num,
                n_trials=trials,
                val_loss=round(val_mae, 5),
                val_mae=round(val_mae, 5),
                best_score=round(best_tracker["score"], 5),
                progress=0.05 + 0.65 * (trial_num / trials),
            )
            return val_mae

        study_kwargs: Dict[str, Any] = {
            "direction": "minimize",
            "sampler": optuna.samplers.TPESampler(seed=seed + horizon),
            "pruner": optuna.pruners.MedianPruner(n_startup_trials=max(3, trials // 4), n_warmup_steps=5),
        }
        if study_name:
            study_kwargs["study_name"] = study_name
        if storage:
            study_kwargs["storage"] = storage
            study_kwargs["load_if_exists"] = True

        study = optuna.create_study(**study_kwargs)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_params = dict(study.best_params)
        if algorithm == "xgboost":
            best_params["random_state"] = seed
            if xgb_fixed_estimators is not None:
                best_params["n_estimators"] = xgb_fixed_estimators
        else:
            best_params["seed"] = seed

        telemetry.emit(
            "final_train_start",
            stage="entrenamiento_final",
            trial=trials,
            n_trials=trials,
            max_epochs=best_params.get("n_estimators") if algorithm == "xgboost" else effective_epochs,
            best_score=round(float(study.best_value), 5),
            progress=0.75,
            extra={"best_params": best_params},
        )

        combined = _concat_data(train_data, val_data)
        final_model = create_model(algorithm)
        final_model.fit(combined, val_data, params=best_params, epochs=effective_epochs)

        for rec in final_model.training_history():
            telemetry.emit(
                "epoch",
                stage="entrenamiento_final",
                trial=trials,
                n_trials=trials,
                epoch=int(rec["epoch"]),
                max_epochs=effective_epochs,
                train_loss=rec.get("train_loss"),
                val_loss=rec.get("val_mae"),
                val_mae=rec.get("val_mae"),
                best_score=round(float(study.best_value), 5),
                progress=0.88,
            )

        y_val_pred = final_model.predict(val_data)
        y_test_pred = final_model.predict(test_data)
        interval = _residual_quantiles(bundle.y_val, y_val_pred, level=0.9)
        lower_test = np.clip(y_test_pred + interval["low"], 0, None)
        upper_test = np.clip(y_test_pred + interval["high"], 0, None)
        outbreak_threshold = float(np.quantile(bundle.y_train, 0.9))
        raw_metrics = evaluate_all(
            bundle.y_test,
            y_test_pred,
            lower=lower_test,
            upper=upper_test,
            y_insample=bundle.y_train[-120:],
            outbreak_threshold=outbreak_threshold,
            interval_level=0.9,
        )
        metrics = _clean_metrics(raw_metrics)

        pred_path = output_dir / f"preds-{run_id}-h{horizon}.npz"
        np.savez_compressed(
            pred_path,
            y_true=np.asarray(bundle.y_test),
            y_pred=np.asarray(y_test_pred),
            district=np.asarray(bundle.meta_test["district_code"].astype(str).tolist()),
            target_week=np.asarray(bundle.meta_test["target_week"].astype(str).tolist()),
        )

        artifact_payload = {
            "algorithm": algorithm,
            "horizon": horizon,
            "model": final_model,
            "params": best_params,
            "feature_names": bundle.feature_names,
            "scaler_mean": bundle.scaler_mean,
            "scaler_std": bundle.scaler_std,
            "grid_shape": bundle.grid_shape,
            "district_grid": bundle.notes.get("district_grid"),
            "sequence_window": bundle.notes.get("sequence_window", 12),
            "interval": interval,
            "outbreak_threshold": outbreak_threshold,
            "data_mode": panel.data_mode,
            "source_summary": panel.source_summary,
            "trained_at": utcnow().isoformat(),
            "is_synthetic_data": bundle.is_synthetic,
            "predictions_path": str(pred_path),
            "run_id": run_id,
            "seed": seed,
            "metrics": metrics,
        }
        artifact_path = output_dir / f"{algorithm}-h{horizon}-{run_id}.joblib"
        joblib.dump(artifact_payload, artifact_path)
        checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

        summary: Dict[str, Any] = {
            "run_id": run_id,
            "algorithm": algorithm,
            "horizon": horizon,
            "trials": trials,
            "epochs": effective_epochs,
            "xgb_n_estimators": xgb_fixed_estimators,
            "seed": seed,
            "data_mode": panel.data_mode,
            "is_synthetic": bundle.is_synthetic,
            "source_summary": panel.source_summary,
            "shapes": shapes,
            "best_val_mae": round(float(study.best_value), 5),
            "best_params": best_params,
            "metrics": metrics,
            "outbreak_threshold": round(outbreak_threshold, 4),
            "artifact_path": str(artifact_path),
            "artifact_checksum": checksum,
            "predictions_path": str(pred_path),
            "telemetry_path": str(telemetry.path),
            "device": hw_device,
            "notes": bundle.notes,
        }
        summary_path = output_dir / f"{algorithm}-h{horizon}-{run_id}-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["summary_path"] = str(summary_path)

        telemetry.emit(
            "completed",
            stage="finalizado",
            trial=trials,
            n_trials=trials,
            val_mae=round(float(study.best_value), 5),
            best_score=round(float(study.best_value), 5),
            progress=1.0,
            extra={
                "test_MAE": metrics.get("MAE"),
                "test_RMSE": metrics.get("RMSE"),
                "artifact_path": str(artifact_path),
                "summary_path": str(summary_path),
            },
        )
        return summary
    except Exception as exc:
        telemetry.emit(
            "failed",
            stage="error",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def cli_entrypoint(
    algorithm: str,
    description: str,
    *,
    allow_n_estimators: bool = False,
    argv: Optional[Sequence[str]] = None,
) -> int:
    configure_cli_logging()
    parser = build_base_parser(description, allow_n_estimators=allow_n_estimators)
    args = parser.parse_args(argv)
    summary = run_training_pipeline(algorithm, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0
