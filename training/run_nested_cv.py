"""Validación Cruzada Anidada temporal tipo Walk-Forward (Fase 2 - Sesión 2).

Ejecuta un esquema cronológico estricto de ``outer_folds`` (por defecto 5) x
``inner_folds`` (por defecto 3) sobre la historia previa al bloque intocable
de 26 semanas de prueba (test 2024):

- Nivel interno (inner loop): optimiza hiperparámetros con Optuna sobre los 3
  inner folds con embargo temporal estricto
  ``max(inner_train.target_week) < min(inner_val.origin_week)`` y
  ``max(inner_val.target_week) < min(outer_val.origin_week)``.
- Nivel externo (outer loop): reentrena con los mejores hiperparámetros sobre
  ``outer_train`` (sin exponer ``outer_val`` a early stopping ni escalado) y
  evalúa una sola vez sobre ``outer_val``, con embargo
  ``max(outer_train.target_week) < min(outer_val.origin_week)`` y
  ``max(outer_val.target_week) < test_start``.
- Registra las 13 métricas del Anexo A por fold externo, calcula agregados
  (media, DE, mín, máx e IC 95%) y guarda incrementalmente tras cada fold en
  ``data/reports/phase2/nested_cv_results.json`` con soporte ``--resume``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.common import (  # noqa: E402
    DEFAULT_TELEMETRY_DIR,
    PROJECT_DIR,
    TelemetryWriter,
    _clean_metrics,
    configure_cli_logging,
    load_training_panel,
    resolve_device_request,
    resolve_xgb_n_estimators,
)
from app.core.config import settings  # noqa: E402
from app.ml.features import (  # noqa: E402
    PanelResult,
    _window_context,
    _windows_for,
    build_features,
    feature_columns,
    fit_scaler,
    to_matrix,
)
from app.ml.metrics import evaluate_all  # noqa: E402
from app.ml.models.base import ModelData  # noqa: E402
from app.ml.models.registry import SEARCH_SPACES, create_model  # noqa: E402
from app.ml.reproducibility import configure_threads, set_reproducible_seed  # noqa: E402
from app.ml.trainer import _residual_quantiles, _suggest  # noqa: E402

logger = logging.getLogger("dengai.training.nested_cv")

ALL_ALGORITHMS = ("xgboost", "lstm", "convlstm", "cnn_lstm", "convlstm_attention")
DEFAULT_NESTED_CV_OUTPUT = "data/reports/phase2/nested_cv_results.json"
ANEXO_A_METRICS: Tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MASE",
    "WIS",
    "coverage",
    "AUC_PR",
    "AUC_ROC",
    "F1",
    "MCC",
    "sensitivity",
    "specificity",
    "Brier",
    "tiempo_s",
)


def _resolve_algorithms(models_arg: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for item in models_arg:
        for part in str(item).split(","):
            cleaned = part.strip().lower()
            if cleaned:
                tokens.append(cleaned)
    if not tokens or "all" in tokens:
        return list(ALL_ALGORITHMS)
    unknown = [m for m in tokens if m not in ALL_ALGORITHMS]
    if unknown:
        raise ValueError(f"Algoritmo(s) desconocido(s): {unknown}. Opciones válidas: {ALL_ALGORITHMS}")
    return list(dict.fromkeys(tokens))


def _resolve_output_path(output: str | Path) -> Path:
    raw = Path(output)
    target = raw if raw.is_absolute() else (PROJECT_DIR / raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _date_str(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def assert_no_nested_leakage(nested_plan: Dict[str, Any]) -> None:
    """Comprueba matemáticamente ausencia de fuga temporal y de objetivos en ambos niveles."""
    test_start = nested_plan["test_start"]
    test_origins = set(nested_plan["test_origins"])
    outer_list = nested_plan["outer_folds"]
    if not outer_list:
        raise AssertionError("No existen outer folds para auditar.")

    prev_outer_val_max = None
    for outer in outer_list:
        o_train: pd.DataFrame = outer["outer_train"]
        o_val: pd.DataFrame = outer["outer_val"]
        o_num = outer["outer_fold"]

        if o_train.empty or o_val.empty:
            raise AssertionError(f"Outer fold {o_num} tiene partición vacía.")
        if not (o_train["target_week"] > o_train["origin_week"]).all():
            raise AssertionError(f"Outer fold {o_num}: target_week <= origin_week en outer_train.")
        if not (o_val["target_week"] > o_val["origin_week"]).all():
            raise AssertionError(f"Outer fold {o_num}: target_week <= origin_week en outer_val.")

        o_train_origin_max = o_train["origin_week"].max()
        o_train_target_max = o_train["target_week"].max()
        o_val_origin_min = o_val["origin_week"].min()
        o_val_origin_max = o_val["origin_week"].max()
        o_val_target_max = o_val["target_week"].max()

        if not (o_train_origin_max < o_val_origin_min):
            raise AssertionError(
                f"Outer fold {o_num}: max(outer_train.origin_week)={o_train_origin_max} >= min(outer_val.origin_week)={o_val_origin_min}."
            )
        if not (o_train_target_max < o_val_origin_min):
            raise AssertionError(
                f"Fuga temporal en outer fold {o_num}: max(outer_train.target_week)={o_train_target_max} >= min(outer_val.origin_week)={o_val_origin_min}."
            )
        if not (o_val_target_max < test_start):
            raise AssertionError(
                f"Fuga hacia test en outer fold {o_num}: max(outer_val.target_week)={o_val_target_max} >= test_start={test_start}."
            )

        o_train_origins = set(o_train["origin_week"])
        o_val_origins = set(o_val["origin_week"])
        o_train_targets = set(o_train["target_week"])
        o_val_targets = set(o_val["target_week"])

        if not o_train_origins.isdisjoint(o_val_origins):
            raise AssertionError(f"Outer fold {o_num}: origin_week compartidas entre outer_train y outer_val.")
        if not o_train_targets.isdisjoint(o_val_targets):
            raise AssertionError(f"Outer fold {o_num}: target_week compartidas entre outer_train y outer_val.")
        if not o_val_origins.isdisjoint(test_origins) or not o_val_targets.isdisjoint(test_origins):
            raise AssertionError(f"Outer fold {o_num}: solapamiento con el bloque intocable de test.")

        if prev_outer_val_max is not None and not (prev_outer_val_max < o_val_origin_min):
            raise AssertionError(f"Outer fold {o_num}: ventanas outer_val no son estrictamente crecientes/disjuntas.")
        prev_outer_val_max = o_val_origin_max

        prev_inner_val_max = None
        for inner in outer["inner_folds"]:
            i_train: pd.DataFrame = inner["inner_train"]
            i_val: pd.DataFrame = inner["inner_val"]
            i_num = inner["inner_fold"]

            if i_train.empty or i_val.empty:
                raise AssertionError(f"Outer {o_num} / Inner {i_num} tiene partición vacía.")
            if not (i_train["target_week"] > i_train["origin_week"]).all():
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: target_week <= origin_week en inner_train.")
            if not (i_val["target_week"] > i_val["origin_week"]).all():
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: target_week <= origin_week en inner_val.")

            i_train_origin_max = i_train["origin_week"].max()
            i_train_target_max = i_train["target_week"].max()
            i_val_origin_min = i_val["origin_week"].min()
            i_val_origin_max = i_val["origin_week"].max()
            i_val_target_max = i_val["target_week"].max()

            if not (i_train_origin_max < i_val_origin_min):
                raise AssertionError(
                    f"Outer {o_num} / Inner {i_num}: max(inner_train.origin_week) >= min(inner_val.origin_week)."
                )
            if not (i_train_target_max < i_val_origin_min):
                raise AssertionError(
                    f"Fuga temporal en Outer {o_num} / Inner {i_num}: max(inner_train.target_week)={i_train_target_max} >= min(inner_val.origin_week)={i_val_origin_min}."
                )
            if not (i_val_target_max < o_val_origin_min):
                raise AssertionError(
                    f"Fuga temporal en Outer {o_num} / Inner {i_num}: max(inner_val.target_week)={i_val_target_max} >= min(outer_val.origin_week)={o_val_origin_min}."
                )

            i_train_origins = set(i_train["origin_week"])
            i_val_origins = set(i_val["origin_week"])
            i_train_targets = set(i_train["target_week"])
            i_val_targets = set(i_val["target_week"])

            if not i_train_origins.isdisjoint(i_val_origins):
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: origin_week compartidas entre inner_train e inner_val.")
            if not i_train_targets.isdisjoint(i_val_targets):
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: target_week compartidas entre inner_train e inner_val.")
            if not i_val_origins.isdisjoint(o_val_origins) or not i_val_targets.isdisjoint(o_val_targets):
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: solapamiento entre inner_val y outer_val.")
            if not i_val_origins.isdisjoint(test_origins) or not i_val_targets.isdisjoint(test_origins):
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: solapamiento con el bloque de test.")

            if prev_inner_val_max is not None and not (prev_inner_val_max < i_val_origin_min):
                raise AssertionError(f"Outer {o_num} / Inner {i_num}: ventanas inner_val no son crecientes/disjuntas.")
            prev_inner_val_max = i_val_origin_max


def build_nested_walk_forward_folds(
    frame: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    outer_val_weeks: int = 26,
    inner_val_weeks: int = 13,
    test_weeks: int = 26,
    horizon: int = 1,
    min_inner_train_weeks: int = 20,
) -> Dict[str, Any]:
    """Construye particiones cronológicas expansivas anidadas (outer x inner) sin fuga temporal."""
    if outer_folds < 1 or inner_folds < 1:
        raise ValueError("outer_folds e inner_folds deben ser >= 1.")
    if outer_val_weeks < 1 or inner_val_weeks < 1 or test_weeks < 1:
        raise ValueError("outer_val_weeks, inner_val_weeks y test_weeks deben ser >= 1.")
    if horizon < 1:
        raise ValueError(f"horizon debe ser >= 1; recibido {horizon}.")

    valid = frame.dropna(subset=["target", "target_week"]).copy()
    origin_weeks = sorted(valid["origin_week"].unique())
    outer_span = outer_folds * outer_val_weeks
    inner_span = inner_folds * inner_val_weeks
    required = test_weeks + outer_span + inner_span + min_inner_train_weeks + horizon
    if len(origin_weeks) < required:
        raise ValueError(
            f"Historia insuficiente para Nested CV ({outer_folds}x{inner_folds}): "
            f"se requieren al menos {required} semanas de origen, disponibles {len(origin_weeks)}."
        )

    test_start = origin_weeks[-test_weeks]
    test_origins = [w for w in origin_weeks if w >= test_start]
    history = [w for w in origin_weeks if w < test_start]
    first_outer_val_idx = len(history) - outer_span

    outer_entries: List[Dict[str, Any]] = []
    for o_idx in range(outer_folds):
        o_val_start_idx = first_outer_val_idx + o_idx * outer_val_weeks
        o_val_end_idx = o_val_start_idx + outer_val_weeks
        o_val_start = history[o_val_start_idx]
        o_train_origins = history[:o_val_start_idx]
        o_val_origins = history[o_val_start_idx:o_val_end_idx]

        # Embargo estricto en ambos lados: target_week < frontera siguiente.
        outer_train = valid[
            valid["origin_week"].isin(o_train_origins) & (valid["target_week"] < o_val_start)
        ].copy()
        outer_val = valid[
            valid["origin_week"].isin(o_val_origins) & (valid["target_week"] < test_start)
        ].copy()
        if outer_train.empty or outer_val.empty:
            raise ValueError(
                f"Outer fold {o_idx + 1} quedó vacío tras aplicar embargo temporal (horizon={horizon})."
            )

        if len(o_train_origins) < inner_span + min_inner_train_weeks + horizon:
            raise ValueError(
                f"Outer fold {o_idx + 1}: semanas insuficientes ({len(o_train_origins)}) para {inner_folds} inner folds."
            )

        first_inner_val_idx = len(o_train_origins) - inner_span
        inner_entries: List[Dict[str, Any]] = []
        for i_idx in range(inner_folds):
            i_val_start_idx = first_inner_val_idx + i_idx * inner_val_weeks
            i_val_end_idx = i_val_start_idx + inner_val_weeks
            i_val_start = o_train_origins[i_val_start_idx]
            i_train_origins = o_train_origins[:i_val_start_idx]
            i_val_origins = o_train_origins[i_val_start_idx:i_val_end_idx]

            inner_train = valid[
                valid["origin_week"].isin(i_train_origins) & (valid["target_week"] < i_val_start)
            ].copy()
            inner_val = valid[
                valid["origin_week"].isin(i_val_origins) & (valid["target_week"] < o_val_start)
            ].copy()
            if inner_train.empty or inner_val.empty:
                raise ValueError(
                    f"Outer {o_idx + 1} / Inner {i_idx + 1} quedó vacío tras aplicar embargo temporal (horizon={horizon})."
                )
            inner_entries.append(
                {
                    "inner_fold": i_idx + 1,
                    "inner_train": inner_train,
                    "inner_val": inner_val,
                    "metadata": {
                        "inner_fold": i_idx + 1,
                        "n_inner_train": int(len(inner_train)),
                        "n_inner_val": int(len(inner_val)),
                        "inner_train_origin_range": [
                            _date_str(inner_train["origin_week"].min()),
                            _date_str(inner_train["origin_week"].max()),
                        ],
                        "inner_train_target_max": _date_str(inner_train["target_week"].max()),
                        "inner_val_origin_range": [
                            _date_str(inner_val["origin_week"].min()),
                            _date_str(inner_val["origin_week"].max()),
                        ],
                        "inner_val_target_max": _date_str(inner_val["target_week"].max()),
                    },
                }
            )

        outer_entries.append(
            {
                "outer_fold": o_idx + 1,
                "outer_train": outer_train,
                "outer_val": outer_val,
                "inner_folds": inner_entries,
                "metadata": {
                    "outer_fold": o_idx + 1,
                    "n_outer_train": int(len(outer_train)),
                    "n_outer_val": int(len(outer_val)),
                    "outer_train_origin_range": [
                        _date_str(outer_train["origin_week"].min()),
                        _date_str(outer_train["origin_week"].max()),
                    ],
                    "outer_train_target_max": _date_str(outer_train["target_week"].max()),
                    "outer_val_origin_range": [
                        _date_str(outer_val["origin_week"].min()),
                        _date_str(outer_val["origin_week"].max()),
                    ],
                    "outer_val_target_range": [
                        _date_str(outer_val["target_week"].min()),
                        _date_str(outer_val["target_week"].max()),
                    ],
                },
            }
        )

    plan = {
        "test_start": test_start,
        "test_origins": test_origins,
        "history_weeks": len(history),
        "test_weeks_reserved": len(test_origins),
        "outer_folds": outer_entries,
    }
    assert_no_nested_leakage(plan)
    return plan


def prepare_fold_model_data(
    full_frame: pd.DataFrame,
    panel_result: PanelResult,
    columns: Sequence[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    needs_tensors: bool,
    window: int = 12,
) -> Tuple[ModelData, ModelData]:
    """Ajusta el escalador estrictamente sobre ``train_df`` y construye ModelData."""
    mean, std = fit_scaler(train_df, columns)
    X_train = to_matrix(train_df, columns, mean, std)
    X_val = to_matrix(val_df, columns, mean, std)
    y_train = train_df["target"].to_numpy(dtype=float)
    y_val = val_df["target"].to_numpy(dtype=float)

    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_val)):
        raise ValueError("La matriz estandarizada del fold contiene NaN o Inf.")

    seq_tr = grid_tr = cell_tr = None
    seq_va = grid_va = cell_va = None
    if needs_tensors:
        context = _window_context(full_frame, panel_result, columns, mean, std, window=window)
        seq_tr, grid_tr, cell_tr = _windows_for(train_df, context)
        seq_va, grid_va, cell_va = _windows_for(val_df, context)

    train_data = ModelData(
        X=X_train,
        seq=seq_tr,
        grid=grid_tr,
        cell=cell_tr,
        y=y_train,
        feature_names=list(columns),
    )
    val_data = ModelData(
        X=X_val,
        seq=seq_va,
        grid=grid_va,
        cell=cell_va,
        y=y_val,
        feature_names=list(columns),
    )
    return train_data, val_data


def summarize_fold_metrics(fold_records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Calcula media, DE, mín, máx e IC 95% (t de Student) sobre los outer folds."""
    summary: Dict[str, Dict[str, Any]] = {}
    for metric_name in ANEXO_A_METRICS:
        values = [
            float(rec["metrics"][metric_name])
            for rec in fold_records
            if rec.get("metrics", {}).get(metric_name) is not None
            and np.isfinite(float(rec["metrics"][metric_name]))
        ]
        n = len(values)
        if n == 0:
            summary[metric_name] = {
                "n": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "ci_95_lower": None,
                "ci_95_upper": None,
            }
            continue
        arr = np.asarray(values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if n > 1:
            se = std_val / np.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            margin = t_crit * se
            ci_low: Optional[float] = round(mean_val - margin, 4)
            ci_high: Optional[float] = round(mean_val + margin, 4)
        else:
            ci_low = None
            ci_high = None

        summary[metric_name] = {
            "n": n,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
        }
    return summary


def _evaluate_outer_fold(
    algorithm: str,
    outer_entry: Dict[str, Any],
    full_frame: pd.DataFrame,
    panel_result: PanelResult,
    columns: Sequence[str],
    *,
    horizon: int,
    trials: int,
    effective_epochs: int,
    xgb_fixed_estimators: Optional[int],
    seed: int,
) -> Dict[str, Any]:
    """Ejecuta Optuna en los inner folds, reentrena en outer_train y evalúa en outer_val."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outer_num = int(outer_entry["outer_fold"])
    needs_tensors = algorithm != "xgboost"
    space = SEARCH_SPACES.get(algorithm, {})

    inner_pairs: List[Dict[str, Any]] = []
    for inner in outer_entry["inner_folds"]:
        tr_data, va_data = prepare_fold_model_data(
            full_frame,
            panel_result,
            columns,
            inner["inner_train"],
            inner["inner_val"],
            needs_tensors=needs_tensors,
        )
        inner_pairs.append({"inner_fold": inner["inner_fold"], "train": tr_data, "val": va_data})

    oof_by_trial: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest(trial, space)
        if algorithm == "xgboost":
            params["random_state"] = seed
            if xgb_fixed_estimators is not None:
                params["n_estimators"] = xgb_fixed_estimators
        else:
            params["seed"] = seed

        inner_maes: List[float] = []
        oof_true_parts: List[np.ndarray] = []
        oof_pred_parts: List[np.ndarray] = []
        for idx, pair in enumerate(inner_pairs):
            model = create_model(algorithm)
            model.fit(pair["train"], pair["val"], params=params, trial=None, epochs=effective_epochs)
            pred = model.predict(pair["val"])
            fold_mae = float(mean_absolute_error(pair["val"].y, pred))
            inner_maes.append(fold_mae)
            oof_true_parts.append(pair["val"].y)
            oof_pred_parts.append(pred)
            trial.report(float(np.mean(inner_maes)), step=idx + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

        oof_by_trial[trial.number] = (np.concatenate(oof_true_parts), np.concatenate(oof_pred_parts))
        return float(np.mean(inner_maes))

    t0 = time.perf_counter()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed + horizon * 100 + outer_num),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=max(3, trials // 4), n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    if algorithm == "xgboost":
        best_params["random_state"] = seed
        if xgb_fixed_estimators is not None:
            best_params["n_estimators"] = xgb_fixed_estimators
    else:
        best_params["seed"] = seed

    # Reentreno exclusivo en outer_train (val=None: jamás se expone outer_val al entrenamiento).
    outer_train_data, outer_val_data = prepare_fold_model_data(
        full_frame,
        panel_result,
        columns,
        outer_entry["outer_train"],
        outer_entry["outer_val"],
        needs_tensors=needs_tensors,
    )
    final_model = create_model(algorithm)
    final_model.fit(outer_train_data, val=None, params=best_params, epochs=effective_epochs)

    # Evaluación única sobre outer_val.
    y_outer_pred = final_model.predict(outer_val_data)
    elapsed_s = round(time.perf_counter() - t0, 4)

    oof_true, oof_pred = oof_by_trial[study.best_trial.number]
    interval = _residual_quantiles(oof_true, oof_pred, level=0.9)
    lower_outer = np.clip(y_outer_pred + interval["low"], 0, None)
    upper_outer = np.clip(y_outer_pred + interval["high"], 0, None)

    outbreak_threshold = float(np.quantile(outer_train_data.y, 0.9))
    # Usa toda la serie de outer_train (todos los distritos) para que la escala ingenua de MASE
    # nunca sea nula cuando un único distrito final tiene ceros consecutivos.
    y_insample = outer_train_data.y
    raw_metrics = evaluate_all(
        outer_val_data.y,
        y_outer_pred,
        lower=lower_outer,
        upper=upper_outer,
        y_insample=y_insample,
        outbreak_threshold=outbreak_threshold,
        interval_level=0.9,
        bootstrap=True,
    )
    cleaned = _clean_metrics(raw_metrics)
    fold_metrics: Dict[str, Optional[float]] = {key: cleaned.get(key) for key in ANEXO_A_METRICS if key != "tiempo_s"}
    fold_metrics["tiempo_s"] = elapsed_s
    for extra_key in ("MAE_ci_lower", "MAE_ci_upper", "RMSE_ci_lower", "RMSE_ci_upper", "MAPE"):
        if extra_key in cleaned:
            fold_metrics[extra_key] = cleaned[extra_key]

    return {
        **outer_entry["metadata"],
        "inner_folds": [inner["metadata"] for inner in outer_entry["inner_folds"]],
        "best_inner_val_mae": round(float(study.best_value), 5),
        "best_params": best_params,
        "outbreak_threshold": round(outbreak_threshold, 4),
        "interval": {
            "low": round(float(interval["low"]), 4),
            "high": round(float(interval["high"]), 4),
            "level": 0.9,
        },
        "metrics": fold_metrics,
    }


def _validate_resume_compatibility(existing: Dict[str, Any], current_config: Dict[str, Any]) -> None:
    prev_cfg = existing.get("config", {})
    keys = ("horizon", "outer_folds", "inner_folds", "outer_val_weeks", "inner_val_weeks", "test_weeks", "data_mode")
    diffs = [
        f"{k}: previo={prev_cfg.get(k)!r} vs actual={current_config.get(k)!r}"
        for k in keys
        if prev_cfg.get(k) != current_config.get(k)
    ]
    if diffs:
        raise ValueError(
            "No se puede usar --resume con una configuración estructural distinta: " + "; ".join(diffs)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validación Cruzada Anidada temporal Walk-Forward (5 outer x 3 inner) DengAI-UNT."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Modelos a evaluar (xgboost, lstm, convlstm, cnn_lstm, convlstm_attention o all).",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte de pronóstico en semanas (defecto: 1).")
    parser.add_argument("--trials", type=int, default=20, help="Trials de Optuna por fold externo (defecto: 20).")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Épocas máximas para modelos neuronales (o n_estimators en XGBoost si no se indica --n-estimators).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Número fijo de árboles para XGBoost (prioridad sobre --epochs).",
    )
    parser.add_argument("--outer-folds", type=int, default=5, help="Número de outer folds (defecto: 5).")
    parser.add_argument("--inner-folds", type=int, default=3, help="Número de inner folds (defecto: 3).")
    parser.add_argument(
        "--inner-val-weeks",
        type=int,
        default=13,
        help="Semanas de validación por inner fold (defecto: 13).",
    )
    parser.add_argument(
        "--outer-val-weeks",
        "--validation-weeks",
        dest="outer_val_weeks",
        type=int,
        default=None,
        help="Semanas de validación por outer fold (defecto: settings.ML_VALIDATION_WEEKS = 26).",
    )
    parser.add_argument(
        "--test-weeks",
        type=int,
        default=None,
        help="Semanas finales intocables reservadas para test (defecto: settings.ML_TEST_WEEKS = 26).",
    )
    parser.add_argument("--max-lag", type=int, default=None, help="Máximo de rezagos (defecto: settings.ML_MAX_LAG = 12).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible (defecto: 42).")
    parser.add_argument("--gpu", action="store_true", help="Solicita GPU CUDA si está disponible.")
    parser.add_argument("--data-mode", choices=["auto", "real", "synthetic"], default="auto")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_NESTED_CV_OUTPUT,
        help=f"Archivo JSON de salida incremental (defecto: {DEFAULT_NESTED_CV_OUTPUT}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reanuda saltando pares (modelo, outer_fold) ya guardados en --output.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Identificador explícito de ejecución.")
    parser.add_argument("--telemetry-dir", type=str, default=DEFAULT_TELEMETRY_DIR)
    return parser


def run_nested_cv(args: argparse.Namespace) -> Dict[str, Any]:
    algorithms = _resolve_algorithms(args.models)
    horizon = int(args.horizon)
    trials = max(1, int(args.trials))
    effective_epochs = int(args.epochs) if args.epochs is not None else 40
    if effective_epochs < 1:
        raise ValueError(f"--epochs debe ser >= 1; recibido {effective_epochs}.")
    seed = int(args.seed)
    outer_folds = int(args.outer_folds)
    inner_folds = int(args.inner_folds)
    inner_val_weeks = int(args.inner_val_weeks)
    outer_val_weeks = int(args.outer_val_weeks or settings.ML_VALIDATION_WEEKS)
    test_weeks = int(args.test_weeks or settings.ML_TEST_WEEKS)
    max_lag = int(args.max_lag or settings.ML_MAX_LAG)
    xgb_fixed_estimators = resolve_xgb_n_estimators(args.epochs, getattr(args, "n_estimators", None))

    output_path = _resolve_output_path(args.output)
    run_id = args.run_id or f"nested-cv-h{horizon}-{uuid.uuid4().hex[:8]}"
    telemetry = TelemetryWriter(run_id, ",".join(algorithms), telemetry_dir=args.telemetry_dir)

    configure_threads()
    set_reproducible_seed(seed)
    hw_device = resolve_device_request(bool(args.gpu))

    panel_result = load_training_panel(args.data_mode)
    full_frame = build_features(panel_result, horizon=horizon, max_lag=max_lag)
    cols = feature_columns(full_frame)
    nested_plan = build_nested_walk_forward_folds(
        full_frame,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        outer_val_weeks=outer_val_weeks,
        inner_val_weeks=inner_val_weeks,
        test_weeks=test_weeks,
        horizon=horizon,
    )

    config_snapshot = {
        "models": algorithms,
        "horizon": horizon,
        "trials": trials,
        "epochs": effective_epochs,
        "xgb_n_estimators": xgb_fixed_estimators,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "outer_val_weeks": outer_val_weeks,
        "inner_val_weeks": inner_val_weeks,
        "test_weeks": test_weeks,
        "max_lag": max_lag,
        "seed": seed,
        "data_mode": panel_result.data_mode,
        "is_synthetic": panel_result.is_synthetic,
        "device": hw_device,
    }

    now_iso = datetime.now(timezone.utc).isoformat()
    if args.resume and output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        _validate_resume_compatibility(report, config_snapshot)
        report["updated_at"] = now_iso
        report["run_id"] = run_id
        existing_models = report.setdefault("models", {})
        # Conserva modelos previos + añade los solicitados.
        merged_models = list(dict.fromkeys(list(report.get("config", {}).get("models", [])) + algorithms))
        report["config"]["models"] = merged_models
    else:
        report = {
            "schema_version": "1.0",
            "protocol": "nested_walk_forward_cv",
            "generated_at": now_iso,
            "updated_at": now_iso,
            "run_id": run_id,
            "config": config_snapshot,
            "source_summary": panel_result.source_summary,
            "leakage_audit": {
                "strict_no_leakage": True,
                "test_block_untouched": True,
                "test_start": _date_str(nested_plan["test_start"]),
                "test_weeks_reserved": nested_plan["test_weeks_reserved"],
                "history_weeks": nested_plan["history_weeks"],
                "outer_folds_verified": len(nested_plan["outer_folds"]),
                "inner_folds_verified": sum(len(o["inner_folds"]) for o in nested_plan["outer_folds"]),
                "assertions_passed": [
                    "max(outer_train.target_week) < min(outer_val.origin_week)",
                    "max(outer_val.target_week) < test_start",
                    "max(inner_train.target_week) < min(inner_val.origin_week)",
                    "max(inner_val.target_week) < min(outer_val.origin_week)",
                    "disjoint_origin_and_target_weeks",
                ],
            },
            "models": {},
        }
        existing_models = report["models"]

    total_units = len(algorithms) * len(nested_plan["outer_folds"])
    completed_units = 0
    telemetry.emit(
        "start",
        stage="nested_cv_init",
        n_trials=trials,
        max_epochs=effective_epochs,
        progress=0.01,
        extra={"algorithms": algorithms, "outer_folds": outer_folds, "inner_folds": inner_folds},
    )

    try:
        for algo in algorithms:
            model_entry = existing_models.setdefault(
                algo,
                {"status": "IN_PROGRESS", "n_outer_folds_completed": 0, "outer_folds": [], "summary": {}},
            )
            by_fold: Dict[int, Dict[str, Any]] = {
                int(rec["outer_fold"]): rec for rec in model_entry.get("outer_folds", [])
            }

            for outer_entry in nested_plan["outer_folds"]:
                o_num = int(outer_entry["outer_fold"])
                if args.resume and o_num in by_fold:
                    completed_units += 1
                    logger.info("[%s] Outer fold %d/%d ya completado (--resume); omitiendo.", algo, o_num, outer_folds)
                    continue

                telemetry.emit(
                    "outer_fold_start",
                    stage=f"{algo}_outer_{o_num}/{outer_folds}",
                    trial=0,
                    n_trials=trials,
                    progress=completed_units / max(1, total_units),
                )
                logger.info(
                    "[%s] Ejecutando outer fold %d/%d (inner=%dx%dw, trials=%d, h=%d)...",
                    algo,
                    o_num,
                    outer_folds,
                    inner_folds,
                    inner_val_weeks,
                    trials,
                    horizon,
                )
                fold_record = _evaluate_outer_fold(
                    algo,
                    outer_entry,
                    full_frame,
                    panel_result,
                    cols,
                    horizon=horizon,
                    trials=trials,
                    effective_epochs=effective_epochs,
                    xgb_fixed_estimators=xgb_fixed_estimators if algo == "xgboost" else None,
                    seed=seed,
                )
                by_fold[o_num] = fold_record
                sorted_folds = [by_fold[k] for k in sorted(by_fold)]
                model_entry["outer_folds"] = sorted_folds
                model_entry["n_outer_folds_completed"] = len(sorted_folds)
                model_entry["status"] = "COMPLETED" if len(sorted_folds) == outer_folds else "IN_PROGRESS"
                model_entry["summary"] = summarize_fold_metrics(sorted_folds)
                report["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_write_json(output_path, report)

                completed_units += 1
                telemetry.emit(
                    "outer_fold_end",
                    stage=f"{algo}_outer_{o_num}/{outer_folds}",
                    trial=trials,
                    n_trials=trials,
                    val_mae=fold_record["metrics"].get("MAE"),
                    best_score=fold_record["best_inner_val_mae"],
                    progress=completed_units / max(1, total_units),
                )
                logger.info(
                    "[%s] Outer fold %d/%d guardado | MAE=%.4f RMSE=%.4f WIS=%.4f MCC=%s tiempo=%.2fs",
                    algo,
                    o_num,
                    outer_folds,
                    fold_record["metrics"]["MAE"],
                    fold_record["metrics"]["RMSE"],
                    fold_record["metrics"]["WIS"],
                    fold_record["metrics"]["MCC"],
                    fold_record["metrics"]["tiempo_s"],
                )

        telemetry.emit(
            "completed",
            stage="finalizado",
            progress=1.0,
            extra={"output_path": str(output_path)},
        )
        return report
    except Exception as exc:
        telemetry.emit(
            "failed",
            stage="error",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_nested_cv(args)
    output_path = _resolve_output_path(args.output)
    compact = {
        "run_id": report["run_id"],
        "output_path": str(output_path),
        "leakage_audit": report["leakage_audit"],
        "models": {
            algo: {
                "status": entry["status"],
                "n_outer_folds_completed": entry["n_outer_folds_completed"],
                "summary": entry["summary"],
            }
            for algo, entry in report["models"].items()
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
