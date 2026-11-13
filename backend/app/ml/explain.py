"""Explicabilidad (M09): SHAP, LIME, importancia temporal/rezagos, atención y narrativa verificable."""
from __future__ import annotations

import logging
import re
import site
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.features import (
    GRID_SHAPE,
    PanelResult,
    build_dataset,
    build_features,
    load_panel,
    make_sample_tensors,
    to_matrix,
)
from app.ml.inference import pick_version, risk_thresholds
from app.ml.models.base import ModelData
from app.ml.trainer import load_bundle
from app.models.geo import GeographicUnit

logger = logging.getLogger("dengai.explain")

# Códigos UBIGEO oficiales de La Libertad (INEI / MINSA) mapeados al catálogo interno.
UBIGEO_TO_DISTRICT: Dict[str, str] = {
    "130101": "PE-LL-TRU",  # Trujillo
    "130102": "PE-LL-EPO",  # El Porvenir
    "130103": "PE-LL-FDM",  # Florencia de Mora
    "130104": "PE-LL-HUA",  # Huanchaco
    "130105": "PE-LL-LES",  # La Esperanza
    "130106": "PE-LL-LAR",  # Laredo
    "130107": "PE-LL-MOC",  # Moche
    "130109": "PE-LL-SAL",  # Salaverry
    "130111": "PE-LL-VLH",  # Víctor Larco Herrera
    "130201": "PE-LL-ASC",  # Ascope
    "130205": "PE-LL-PAI",  # Paiján
    "130401": "PE-LL-CHE",  # Chepén
    "130704": "PE-LL-PAC",  # Pacasmayo
    "131001": "PE-LL-SDC",  # Santiago de Chuco
}
DISTRICT_TO_UBIGEO: Dict[str, str] = {code: ubigeo for ubigeo, code in UBIGEO_TO_DISTRICT.items()}

_STATIC_FEATURES = frozenset(
    {
        "altitude_m",
        "altitude_km",
        "population",
        "area_km2",
        "latitude",
        "longitude",
        "population_density",
        "log_population",
        "log_density",
    }
)

_LABELS: Dict[str, str] = {
    "cases_lag_1": "Casos confirmados (1 semana antes)",
    "cases_mean_4": "Promedio de casos (4 semanas)",
    "cases_mean_8": "Promedio de casos (8 semanas)",
    "cases_mean_12": "Promedio de casos (12 semanas)",
    "cases_std_4": "Variabilidad de casos (4 semanas)",
    "cases_diff_1": "Cambio semanal de casos",
    "cases_growth_4": "Razón de crecimiento de casos",
    "neighbour_cases_lag_1": "Casos en distritos vecinos (semana previa)",
    "temperature_mean": "Temperatura media",
    "temperature_mean_lag_0": "Temperatura media",
    "temperature_max": "Temperatura máxima",
    "temperature_min": "Temperatura mínima",
    "humidity_mean": "Humedad relativa",
    "humidity_mean_lag_0": "Humedad relativa",
    "precipitation_mm": "Precipitación",
    "precipitation_mm_lag_0": "Precipitación",
    "precipitation_mean_4": "Precipitación acumulada (4 semanas)",
    "precipitation_mean_12": "Precipitación acumulada (12 semanas)",
    "temperature_mean_4": "Temperatura media (4 semanas)",
    "temperature_mean_12": "Temperatura media (12 semanas)",
    "altitude_m": "Altitud (m)",
    "altitude_km": "Altitud (km)",
    "area_km2": "Superficie (km²)",
    "latitude": "Latitud",
    "longitude": "Longitud",
    "population": "Población",
    "log_population": "Población (log)",
    "log_density": "Densidad poblacional (log)",
    "population_density": "Densidad poblacional",
    "sin_week": "Estacionalidad (seno)",
    "cos_week": "Estacionalidad (coseno)",
    "sin_week_2": "Estacionalidad armónica 2 (seno)",
    "cos_week_2": "Estacionalidad armónica 2 (coseno)",
    "time_index": "Tendencia temporal",
    "solar_radiation": "Radiación solar",
    "wind_speed": "Velocidad del viento",
}

_LAG_PATTERN = re.compile(r"^(cases_lag|temperature_mean_lag|humidity_mean_lag|precipitation_mm_lag)_(\d+)$")


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", text)).strip().upper()


def resolve_district_code(district_or_ubigeo: str, db: Optional[Session] = None) -> str:
    """Resuelve códigos ``PE-LL-*``, UBIGEO de 6 dígitos (p. ej. ``130101``) o nombres de distrito."""
    raw = str(district_or_ubigeo or "").strip()
    if not raw:
        raise ValidationError("Debe indicar un código de distrito o UBIGEO válido.")

    digits = re.sub(r"\D", "", raw)
    if len(digits) == 6 and digits in UBIGEO_TO_DISTRICT:
        return UBIGEO_TO_DISTRICT[digits]

    upper = raw.upper()
    if upper in DISTRICT_TO_UBIGEO:
        return upper

    if db is not None:
        direct = db.scalar(select(GeographicUnit).where(GeographicUnit.code == upper))
        if direct is not None:
            return direct.code
        norm_query = _normalize_text(raw)
        for unit in db.scalars(select(GeographicUnit)):
            if _normalize_text(unit.name) == norm_query or unit.code.upper() == upper:
                return unit.code

    return upper


def feature_label(name: str) -> str:
    if name in _LABELS:
        return _LABELS[name]
    match = _LAG_PATTERN.match(name)
    if match:
        base, lag = match.group(1), int(match.group(2))
        base_label = {
            "cases_lag": "Casos confirmados",
            "temperature_mean_lag": "Temperatura media",
            "humidity_mean_lag": "Humedad relativa",
            "precipitation_mm_lag": "Precipitación",
        }[base]
        return f"{base_label} (rezago {lag} sem.)"
    return name.replace("_", " ").capitalize()


def _import_lime_tabular():
    """Importa LimeTabularExplainer respetando la precedencia de ``libs/``."""
    try:
        from lime.lime_tabular import LimeTabularExplainer

        return LimeTabularExplainer
    except ImportError:
        saved = list(sys.path)
        try:
            for sp in site.getsitepackages():
                if sp not in sys.path:
                    sys.path.append(sp)
            from lime.lime_tabular import LimeTabularExplainer

            return LimeTabularExplainer
        finally:
            sys.path[:] = saved


def _model_requires(model: Any) -> str:
    requires = getattr(model, "requires", None)
    if requires in ("tabular", "sequence", "grid"):
        return str(requires)
    key = str(getattr(model, "key", "")).lower()
    if key == "lstm":
        return "sequence"
    if key in {"convlstm", "cnn_lstm", "convlstm_attention"}:
        return "grid"
    return "tabular"


class TabularModelAdapter:
    """Adaptador unificado que traduce perturbaciones tabulares (M, F) a X/seq/grid."""

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        base_X: np.ndarray,
        *,
        base_seq: Optional[np.ndarray] = None,
        base_grid: Optional[np.ndarray] = None,
        base_cell: Optional[np.ndarray] = None,
        grid_shape: Tuple[int, int] = GRID_SHAPE,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.requires = _model_requires(model)
        self.base_X = np.asarray(base_X, dtype=np.float32).reshape(1, -1)
        self.base_seq = None if base_seq is None else np.asarray(base_seq, dtype=np.float32)[:1]
        self.base_grid = None if base_grid is None else np.asarray(base_grid, dtype=np.float32)[:1]
        self.base_cell = None if base_cell is None else np.asarray(base_cell, dtype=np.int64)[:1]
        self.grid_shape = tuple(int(v) for v in grid_shape)
        self._col_idx = {name: idx for idx, name in enumerate(self.feature_names)}
        self._static_indices = [idx for idx, name in enumerate(self.feature_names) if name in _STATIC_FEATURES]

        if self.requires == "sequence" and self.base_seq is None:
            raise ValueError("El modelo secuencial requiere 'base_seq' para explicar perturbaciones.")
        if self.requires == "grid" and (self.base_grid is None or self.base_cell is None):
            raise ValueError("El modelo espacio-temporal requiere 'base_grid' y 'base_cell' para explicar.")

        self._grid_ops = self._build_grid_ops() if self.requires == "grid" else []
        if self.requires == "grid":
            self.active_indices: List[int] = sorted({feat_idx for feat_idx, *_ in self._grid_ops})
        else:
            self.active_indices = list(range(len(self.feature_names)))

    def _build_grid_ops(self) -> List[Tuple[int, slice, int, float, bool]]:
        """Mapea cada predictor tabular relevante a (col_idx, step_slice, channel, weight, is_neighbour)."""
        assert self.base_grid is not None
        steps = int(self.base_grid.shape[1])
        ops: List[Tuple[int, slice, int, float, bool]] = []
        channel_bases = (
            ("temperature_mean", 1),
            ("humidity_mean", 2),
            ("precipitation_mm", 3),
        )
        for idx, name in enumerate(self.feature_names):
            if name.startswith("cases_lag_"):
                lag = int(name.rsplit("_", 1)[1])
                step = steps - 1 - lag
                if 0 <= step < steps:
                    ops.append((idx, slice(step, step + 1), 0, 1.0, False))
            elif name == "cases_mean_4":
                start = max(0, steps - 5)
                if start < steps - 1:
                    ops.append((idx, slice(start, steps - 1), 0, 1.0, False))
            elif name == "cases_mean_8":
                start = max(0, steps - 9)
                if start < steps - 1:
                    ops.append((idx, slice(start, steps - 1), 0, 1.0, False))
            elif name == "cases_mean_12":
                if steps > 1:
                    ops.append((idx, slice(0, steps - 1), 0, 1.0, False))
            elif name == "cases_diff_1" and steps >= 3:
                ops.append((idx, slice(steps - 2, steps - 1), 0, 0.5, False))
                ops.append((idx, slice(steps - 3, steps - 2), 0, -0.5, False))
            elif name == "cases_growth_4" and steps >= 2:
                ops.append((idx, slice(steps - 2, steps - 1), 0, 1.0, False))
            elif name == "neighbour_cases_lag_1" and steps >= 2:
                ops.append((idx, slice(steps - 2, steps - 1), 0, 1.0, True))
            else:
                for base_col, ch_idx in channel_bases:
                    if name == base_col or name == f"{base_col}_lag_0":
                        ops.append((idx, slice(steps - 1, steps), ch_idx, 1.0, False))
                    elif name.startswith(f"{base_col}_lag_"):
                        lag = int(name.rsplit("_", 1)[1])
                        step = steps - 1 - lag
                        if 0 <= step < steps:
                            ops.append((idx, slice(step, step + 1), ch_idx, 1.0, False))
                    elif name == f"{base_col}_4":
                        start = max(0, steps - 4)
                        ops.append((idx, slice(start, steps), ch_idx, 1.0, False))
                    elif name == f"{base_col}_12":
                        ops.append((idx, slice(0, steps), ch_idx, 1.0, False))
        return ops

    def build_model_data(self, matrix: np.ndarray) -> ModelData:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        batch = arr.shape[0]
        zeros_y = np.zeros(batch, dtype=np.float32)

        if self.requires == "tabular":
            return ModelData(
                X=arr,
                seq=None,
                grid=None,
                cell=None,
                y=zeros_y,
                feature_names=self.feature_names,
            )

        if self.requires == "sequence":
            assert self.base_seq is not None
            seq_batch = np.repeat(self.base_seq, batch, axis=0)
            seq_batch[:, -1, :] = arr
            if self._static_indices:
                seq_batch[:, :, self._static_indices] = arr[:, None, self._static_indices]
            return ModelData(
                X=arr,
                seq=seq_batch,
                grid=None,
                cell=None,
                y=zeros_y,
                feature_names=self.feature_names,
            )

        assert self.base_grid is not None and self.base_cell is not None
        grid_batch = np.repeat(self.base_grid, batch, axis=0)
        cell_batch = np.repeat(self.base_cell, batch, axis=0)
        delta = arr - self.base_X

        cols = int(self.base_grid.shape[3])
        row, col = divmod(int(self.base_cell[0]), cols)
        active_mask = self.base_grid[0, -1, :, :, -1] > 0.5
        neighbour_mask = active_mask.copy()
        neighbour_mask[row, col] = False
        nr, nc = np.where(neighbour_mask)

        for feat_idx, step_slice, ch_idx, weight, is_neighbour in self._grid_ops:
            d = delta[:, feat_idx]
            if not np.any(d):
                continue
            shift = (d * weight)[:, None]
            if is_neighbour:
                if len(nr) > 0:
                    grid_batch[:, step_slice, nr, nc, ch_idx] += shift[:, :, None]
            else:
                grid_batch[:, step_slice, row, col, ch_idx] += shift

        return ModelData(
            X=arr,
            seq=None,
            grid=grid_batch,
            cell=cell_batch,
            y=zeros_y,
            feature_names=self.feature_names,
        )

    def predict_from_matrix(self, matrix: np.ndarray) -> np.ndarray:
        data = self.build_model_data(matrix)
        return np.asarray(self.model.predict(data), dtype=float).reshape(-1)

    def predict_active(self, active_matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(active_matrix, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if len(self.active_indices) == len(self.feature_names):
            return self.predict_from_matrix(arr)
        full = np.repeat(self.base_X, arr.shape[0], axis=0)
        full[:, self.active_indices] = arr
        return self.predict_from_matrix(full)


def _shap_values(
    model,
    X: np.ndarray,
    background: np.ndarray,
    *,
    adapter: Optional[TabularModelAdapter] = None,
    nsamples: int = 120,
    seed: int = 42,
    raise_on_error: bool = False,
) -> Optional[np.ndarray]:
    """SHAP para XGBoost (TreeExplainer) y modelos PyTorch (KernelExplainer vía adaptador)."""
    try:
        import shap

        logging.getLogger("shap").setLevel(logging.WARNING)
        algorithm = str(getattr(model, "key", "")).lower()
        if algorithm == "xgboost" and getattr(model, "model", None) is not None:
            explainer = shap.TreeExplainer(model.model)
            values = explainer.shap_values(X)
            scale = float(getattr(model, "_base_score_scale", 1.0))
            return np.asarray(values, dtype=float).reshape(X.shape) * scale

        rng = np.random.default_rng(seed)
        bg_size = min(40, len(background))
        idx = rng.choice(len(background), bg_size, replace=False) if len(background) > bg_size else np.arange(len(background))
        background_sample = np.asarray(background[idx], dtype=np.float32)

        if adapter is not None and len(adapter.active_indices) < X.shape[1]:
            act = adapter.active_indices
            explainer = shap.KernelExplainer(adapter.predict_active, background_sample[:, act])
            sub_values = explainer.shap_values(np.asarray(X[:, act], dtype=np.float32), nsamples=nsamples, silent=True)
            full_values = np.zeros(X.shape, dtype=float)
            full_values[:, act] = np.asarray(sub_values, dtype=float).reshape(X.shape[0], len(act))
            return full_values

        predict_fn = adapter.predict_from_matrix if adapter is not None else (
            lambda matrix: model.predict(ModelData(X=matrix, seq=None, grid=None, cell=None, y=np.zeros(len(matrix))))
        )
        explainer = shap.KernelExplainer(predict_fn, background_sample)
        values = explainer.shap_values(np.asarray(X, dtype=np.float32), nsamples=nsamples, silent=True)
        return np.asarray(values, dtype=float).reshape(X.shape)
    except Exception:
        logger.exception("No fue posible calcular SHAP")
        if raise_on_error:
            raise
        return None


def _permutation_contributions(
    model,
    X: np.ndarray,
    mean_vector: Optional[np.ndarray],
    feature_names: Sequence[str],
    *,
    adapter: Optional[TabularModelAdapter] = None,
    seq: Optional[np.ndarray] = None,
    grid: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Contribución por perturbación: reemplaza cada variable estandarizada por su media de train (0.0)."""
    active_adapter = adapter
    if active_adapter is None:
        active_adapter = TabularModelAdapter(
            model,
            feature_names,
            X,
            base_seq=seq,
            base_grid=grid,
            base_cell=cell,
        )

    # En espacio estandarizado, la media de entrenamiento es 0.0 (solo se usa baseline explícito si se pasa en escala estandarizada).
    baseline = np.zeros(len(feature_names), dtype=np.float32)
    if mean_vector is not None and len(mean_vector) == len(feature_names) and np.all(np.abs(mean_vector) < 1e-6):
        baseline = np.asarray(mean_vector, dtype=np.float32)

    base_pred = float(active_adapter.predict_from_matrix(X)[0])
    batch = np.repeat(np.asarray(X[:1], dtype=np.float32), len(feature_names), axis=0)
    for idx in range(len(feature_names)):
        batch[idx, idx] = baseline[idx]
    perturbed_preds = active_adapter.predict_from_matrix(batch)
    return (base_pred - perturbed_preds).astype(float)


def temporal_importance(
    model,
    seq: Optional[np.ndarray] = None,
    cell: Optional[np.ndarray] = None,
    window: int = 12,
    *,
    grid: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Importancia temporal: impacto de neutralizar cada semana de la ventana (0.0 = media train)."""
    requires = _model_requires(model)
    if requires == "grid":
        if grid is None or cell is None:
            raise ValueError("temporal_importance requiere 'grid' y 'cell' para modelos espacio-temporales.")
        steps = int(grid.shape[1])
        base = float(model.predict(ModelData(X=None, seq=None, grid=grid[:1], cell=cell[:1], y=np.zeros(1)))[0])
        results: List[Dict[str, Any]] = []
        for step in range(steps):
            perturbed = np.array(grid[:1], copy=True)
            # Se preservan los indicadores de máscara del canal final (:-1 = clima y casos estandarizados).
            perturbed[:, step, :, :, :-1] = 0.0
            value = float(model.predict(ModelData(X=None, seq=None, grid=perturbed, cell=cell[:1], y=np.zeros(1)))[0])
            delta = float(base - value)
            results.append(
                {
                    "step": step - steps + 1,
                    "lag": steps - 1 - step,
                    "delta": round(delta, 4),
                    "abs_delta": round(abs(delta), 4),
                }
            )
        return results

    if requires == "sequence":
        if seq is None:
            raise ValueError("temporal_importance requiere 'seq' para modelos LSTM.")
        steps = int(seq.shape[1])
        base = float(model.predict(ModelData(X=None, seq=seq[:1], grid=None, cell=cell, y=np.zeros(1)))[0])
        results = []
        for step in range(steps):
            perturbed = np.array(seq[:1], copy=True)
            perturbed[:, step, :] = 0.0
            value = float(model.predict(ModelData(X=None, seq=perturbed, grid=None, cell=cell, y=np.zeros(1)))[0])
            delta = float(base - value)
            results.append(
                {
                    "step": step - steps + 1,
                    "lag": steps - 1 - step,
                    "delta": round(delta, 4),
                    "abs_delta": round(abs(delta), 4),
                }
            )
        return results

    return []


def lime_explanation(
    adapter: TabularModelAdapter,
    X: np.ndarray,
    background: np.ndarray,
    feature_names: Sequence[str],
    *,
    raw_row: Optional[pd.Series] = None,
    num_features: Optional[int] = None,
    num_samples: int = 500,
    top_k: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Calcula explicación local con LimeTabularExplainer sobre fondo de entrenamiento."""
    LimeTabularExplainer = _import_lime_tabular()
    bg = np.asarray(background, dtype=float)
    instance = np.asarray(X[0], dtype=float)
    names = list(feature_names)
    active_idx = list(getattr(adapter, "active_indices", range(len(names))))
    use_active = len(active_idx) < len(names)
    sub_names = [names[i] for i in active_idx] if use_active else names
    sub_bg = bg[:, active_idx] if use_active else bg
    sub_inst = instance[active_idx] if use_active else instance
    predict_fn = adapter.predict_active if use_active else adapter.predict_from_matrix

    explainer = LimeTabularExplainer(
        training_data=sub_bg,
        feature_names=sub_names,
        mode="regression",
        discretize_continuous=True,
        random_state=random_state,
    )
    exp = explainer.explain_instance(
        sub_inst,
        predict_fn,
        num_features=min(int(num_features or len(sub_names)), len(sub_names)),
        num_samples=int(num_samples),
    )

    label_key = 1 if 1 in exp.local_exp else next(iter(exp.local_exp.keys()))
    weights = np.zeros(len(names), dtype=float)
    rules: Dict[int, str] = {}
    for local_pos, weight in exp.local_exp[label_key]:
        full_idx = active_idx[int(local_pos)] if use_active else int(local_pos)
        weights[full_idx] = float(weight)

    try:
        for (local_pos, _), (rule_text, _) in zip(exp.local_exp[label_key], exp.as_list(label=label_key)):
            full_idx = active_idx[int(local_pos)] if use_active else int(local_pos)
            rules[full_idx] = str(rule_text)
    except Exception:
        pass

    intercept_raw = exp.intercept[label_key] if isinstance(exp.intercept, dict) else exp.intercept
    local_pred_raw = exp.local_pred[0] if isinstance(exp.local_pred, (list, tuple, np.ndarray)) else exp.local_pred
    order = np.argsort(-np.abs(weights))[:top_k]
    factors: List[Dict[str, Any]] = []
    for idx in order:
        w = float(weights[idx])
        raw_val = None if raw_row is None else raw_row[names[idx]]
        factors.append(
            {
                "feature": names[idx],
                "label": feature_label(names[idx]),
                "contribution": round(w, 4),
                "direction": "aumenta" if w > 0 else "reduce",
                "value": None if raw_val is None or pd.isna(raw_val) else round(float(raw_val), 4),
                "rule": rules.get(int(idx), ""),
            }
        )

    return {
        "method": "LIME (LimeTabularExplainer)",
        "intercept": round(float(intercept_raw), 4),
        "local_prediction": round(float(local_pred_raw), 4),
        "r2_score": round(float(exp.score), 4) if exp.score is not None and np.isfinite(exp.score) else None,
        "contributions": weights,
        "factors": factors,
    }


def aggregate_lag_importance(
    feature_names: Sequence[str],
    contributions: np.ndarray,
    *,
    max_lag: int = 12,
) -> Dict[str, Any]:
    """Agrega importancia absoluta y acumulada de rezagos t-1..t-12 para casos y clima."""
    contrib_map = {name: float(contributions[idx]) for idx, name in enumerate(feature_names)}
    climate_cols = ("temperature_mean_lag", "humidity_mean_lag", "precipitation_mm_lag")

    rows: List[Dict[str, Any]] = []
    cum_cases = 0.0
    cum_climate = 0.0
    cum_total = 0.0

    for lag in range(1, max_lag + 1):
        case_val = contrib_map.get(f"cases_lag_{lag}", 0.0)
        case_abs = abs(case_val)

        clim_parts = {col: abs(contrib_map.get(f"{col}_{lag}", 0.0)) for col in climate_cols}
        clim_abs = sum(clim_parts.values())
        clim_signed = sum(contrib_map.get(f"{col}_{lag}", 0.0) for col in climate_cols)
        total_abs = case_abs + clim_abs

        cum_cases += case_abs
        cum_climate += clim_abs
        cum_total += total_abs

        rows.append(
            {
                "lag": lag,
                "label": f"t-{lag}",
                "cases_abs": round(case_abs, 4),
                "cases_signed": round(case_val, 4),
                "climate_abs": round(clim_abs, 4),
                "climate_signed": round(clim_signed, 4),
                "climate_breakdown": {
                    "temperature_abs": round(clim_parts["temperature_mean_lag"], 4),
                    "humidity_abs": round(clim_parts["humidity_mean_lag"], 4),
                    "precipitation_abs": round(clim_parts["precipitation_mm_lag"], 4),
                },
                "total_abs": round(total_abs, 4),
                "cumulative_cases_abs": round(cum_cases, 4),
                "cumulative_climate_abs": round(cum_climate, 4),
                "cumulative_total_abs": round(cum_total, 4),
            }
        )

    denom = cum_total if cum_total > 1e-12 else 1.0
    for row in rows:
        row["cumulative_share_pct"] = round(100.0 * row["cumulative_total_abs"] / denom, 2)

    contemp_climate = sum(
        abs(contrib_map.get(name, 0.0))
        for name in (
            "temperature_mean",
            "temperature_mean_lag_0",
            "humidity_mean",
            "humidity_mean_lag_0",
            "precipitation_mm",
            "precipitation_mm_lag_0",
        )
    )
    rolling_climate = sum(
        abs(contrib_map.get(name, 0.0))
        for name in ("temperature_mean_4", "temperature_mean_12", "precipitation_mean_4", "precipitation_mean_12")
    )
    rolling_cases = sum(
        abs(contrib_map.get(name, 0.0))
        for name in ("cases_mean_4", "cases_mean_8", "cases_mean_12", "cases_std_4", "cases_diff_1", "cases_growth_4")
    )

    return {
        "lags": rows,
        "summary": {
            "total_cases_lags_1_12": round(cum_cases, 4),
            "total_climate_lags_1_12": round(cum_climate, 4),
            "total_lags_1_12": round(cum_total, 4),
            "cases_share_pct": round(100.0 * cum_cases / denom, 2),
            "climate_share_pct": round(100.0 * cum_climate / denom, 2),
            "contemporaneous_climate_t0": round(contemp_climate, 4),
            "rolling_climate_windows": round(rolling_climate, 4),
            "rolling_cases_windows": round(rolling_cases, 4),
        },
    }


def extract_convlstm_attention(
    model: Any,
    data: ModelData,
    *,
    district_grid: Optional[Dict[str, Sequence[int]]] = None,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
    district_names: Optional[Dict[str, str]] = None,
    target_district: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extrae pesos reales de atención temporal multi-cabezal y espacial 4x5 de ConvLSTMAttention."""
    if not hasattr(model, "predict_with_attention"):
        return None

    preds, raw_attn = model.predict_with_attention(data)
    temporal_w = np.asarray(raw_attn["temporal_weights"][0], dtype=float)
    temporal_heads = np.asarray(raw_attn["temporal_head_weights"][0], dtype=float)
    spatial_w = np.asarray(raw_attn["spatial_weights"][0], dtype=float)

    rows, cols = int(spatial_w.shape[0]), int(spatial_w.shape[1])
    cell_to_code: Dict[Tuple[int, int], str] = {}
    if district_grid:
        for code, rc in district_grid.items():
            cell_to_code[(int(rc[0]), int(rc[1]))] = str(code)

    spatial_districts: List[Dict[str, Any]] = []
    for r in range(rows):
        for c in range(cols):
            code = cell_to_code.get((r, c))
            if code is None:
                continue
            spatial_districts.append(
                {
                    "district_code": code,
                    "ubigeo": DISTRICT_TO_UBIGEO.get(code),
                    "district_name": (district_names or {}).get(code, code),
                    "row": r,
                    "col": c,
                    "weight": round(float(spatial_w[r, c]), 6),
                    "is_target_district": code == target_district,
                }
            )
    spatial_districts.sort(key=lambda item: -item["weight"])

    steps = len(temporal_w)
    temporal_list = []
    for step in range(steps):
        temporal_list.append(
            {
                "step": step - steps + 1,
                "lag": steps - 1 - step,
                "label": f"t-{steps - 1 - step}" if step < steps - 1 else "t",
                "weight": round(float(temporal_w[step]), 6),
                "head_weights": [round(float(w), 6) for w in temporal_heads[step]],
            }
        )

    return {
        "prediction": round(float(preds[0]), 4),
        "grid_shape": [rows, cols],
        "n_occupied_cells": len(spatial_districts),
        "n_heads": int(temporal_heads.shape[1]) if temporal_heads.ndim == 2 else 1,
        "spatial_matrix": [[round(float(v), 6) for v in row_vals] for row_vals in spatial_w],
        "spatial_districts": spatial_districts,
        "temporal_weights": temporal_list,
    }


def explain_sample(
    *,
    model: Any,
    panel: PanelResult,
    frame: pd.DataFrame,
    subset: pd.DataFrame,
    background_frame: pd.DataFrame,
    columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    window: int = 12,
    district_grid: Optional[Dict[str, Sequence[int]]] = None,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
    top_k: int = 8,
    shap_samples: int = 120,
    lime_samples: int = 500,
    include_lime: bool = True,
    raise_on_error: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Ejecuta explicación completa (SHAP, LIME, rezagos, perturbación temporal y atención) para 1 fila."""
    if len(subset) != 1:
        raise ValidationError(f"Se esperaba exactamente 1 fila para explicar; se recibieron {len(subset)}.")

    cols = list(columns)
    district_code = str(subset["district_code"].iloc[0])
    seq, grid, cell, context = make_sample_tensors(panel, frame, subset, cols, mean, std, window=window)
    X = to_matrix(subset, cols, mean, std)
    effective_grid = district_grid or context.get("mapping")

    adapter = TabularModelAdapter(
        model,
        cols,
        X,
        base_seq=seq,
        base_grid=grid,
        base_cell=cell,
        grid_shape=grid_shape,
    )
    sample_data = ModelData(
        X=X,
        seq=seq,
        grid=grid,
        cell=cell,
        y=subset["target"].to_numpy(dtype=float),
        feature_names=cols,
    )
    prediction = float(model.predict(sample_data)[0])

    bg_source = background_frame if len(background_frame) > 0 else frame
    bg_sample = bg_source.sample(min(400, len(bg_source)), random_state=seed)
    background = to_matrix(bg_sample, cols, mean, std)

    values = _shap_values(
        model,
        X,
        background,
        adapter=adapter,
        nsamples=shap_samples,
        seed=seed,
        raise_on_error=raise_on_error,
    )
    method = "SHAP"
    if values is None:
        values = _permutation_contributions(model, X, None, cols, adapter=adapter).reshape(1, -1)
        method = "permutación (respuesta de contingencia)"

    contributions = np.asarray(values, dtype=float).reshape(-1)
    base_value = prediction - float(np.sum(contributions))
    raw_row = subset.iloc[0]
    order = np.argsort(-np.abs(contributions))[:top_k]
    factors = []
    for idx in order:
        contrib = float(contributions[idx])
        raw_val = raw_row[cols[idx]]
        factors.append(
            {
                "feature": cols[idx],
                "label": feature_label(cols[idx]),
                "contribution": round(contrib, 4),
                "direction": "aumenta" if contrib > 0 else "reduce",
                "value": None if pd.isna(raw_val) else round(float(raw_val), 4),
            }
        )

    temporal: List[Dict[str, Any]] = []
    if _model_requires(model) in ("sequence", "grid"):
        try:
            temporal = temporal_importance(model, seq=seq, grid=grid, cell=cell, window=window)
        except Exception:
            logger.exception("Error calculando importancia temporal")
            if raise_on_error:
                raise

    lag_summary = aggregate_lag_importance(cols, contributions, max_lag=12)
    lag_compat = [
        {"lag": item["lag"], "abs_contribution": item["cases_abs"], "climate_abs": item["climate_abs"], "total_abs": item["total_abs"]}
        for item in lag_summary["lags"]
    ]

    lime_result: Optional[Dict[str, Any]] = None
    if include_lime:
        try:
            lime_result = lime_explanation(
                adapter,
                X,
                background,
                cols,
                raw_row=raw_row,
                num_features=len(cols),
                num_samples=lime_samples,
                top_k=top_k,
                random_state=seed,
            )
        except Exception:
            logger.exception("No fue posible calcular LIME")
            if raise_on_error:
                raise

    attention = extract_convlstm_attention(
        model,
        sample_data,
        district_grid=effective_grid,
        grid_shape=grid_shape,
        district_names=panel.district_names,
        target_district=district_code,
    )

    return {
        "district_code": district_code,
        "ubigeo": DISTRICT_TO_UBIGEO.get(district_code),
        "district_name": panel.district_names.get(district_code, district_code),
        "method": method,
        "base_value": round(base_value, 4),
        "prediction": round(prediction, 4),
        "actual_target": None if pd.isna(raw_row.get("target")) else round(float(raw_row["target"]), 4),
        "factors": factors,
        "shap_contributions": contributions,
        "temporal_importance": temporal or lag_compat,
        "temporal_perturbation": temporal,
        "lag_importance": lag_summary,
        "lime": lime_result,
        "attention": attention,
        "feature_names": cols,
    }


def explain_district(
    db: Session,
    district_code: str,
    *,
    horizon: int = 1,
    model_version_id: Optional[int] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    resolved_code = resolve_district_code(district_code, db=db)
    unit = db.scalar(select(GeographicUnit).where(GeographicUnit.code == resolved_code))
    if unit is None:
        raise NotFoundError("El distrito indicado no existe.")
    version = pick_version(db, horizon, model_version_id)
    bundle = load_bundle(version.artifact_path)
    model = bundle["model"]
    columns: List[str] = list(bundle["feature_names"])
    mean = np.asarray(bundle["scaler_mean"], dtype=float)
    std = np.asarray(bundle["scaler_std"], dtype=float)

    panel = load_panel(db)
    frame = build_features(panel, horizon=horizon)
    origin = max(panel.weeks)
    subset = frame[(frame["origin_week"] == origin) & (frame["district_code"] == resolved_code)]
    if subset.empty:
        raise ValidationError("No hay datos suficientes para explicar este distrito en la última semana.")

    # Fondo estrictamente de entrenamiento cuando el rango permite partición temporal.
    try:
        ds = build_dataset(panel, horizon=horizon)
        bg_frame = frame.merge(ds.meta_train[["district_code", "origin_week"]], on=["district_code", "origin_week"])
    except Exception:
        bg_frame = frame[frame["origin_week"] < origin]

    result = explain_sample(
        model=model,
        panel=panel,
        frame=frame,
        subset=subset,
        background_frame=bg_frame,
        columns=columns,
        mean=mean,
        std=std,
        window=int(bundle.get("sequence_window", 12)),
        district_grid=bundle.get("district_grid"),
        grid_shape=tuple(bundle.get("grid_shape", GRID_SHAPE)),
        top_k=top_k,
        include_lime=False,
        raise_on_error=False,
    )

    thresholds = risk_thresholds(db)
    population = float(unit.population or 0) or None
    prediction = float(result["prediction"])
    incidence = (prediction / population * 100_000) if population else None
    narrative = _build_narrative(
        unit.name,
        prediction,
        incidence,
        result["factors"],
        thresholds,
        horizon,
        result["method"],
        panel.is_synthetic,
    )

    return {
        "district_code": unit.code,
        "ubigeo": DISTRICT_TO_UBIGEO.get(unit.code),
        "district_name": unit.name,
        "model_version": version.version,
        "algorithm": version.algorithm,
        "method": result["method"],
        "base_value": result["base_value"],
        "prediction": result["prediction"],
        "factors": result["factors"],
        "temporal_importance": result["temporal_importance"],
        "lag_importance": result["lag_importance"],
        "attention": result["attention"],
        "narrative": narrative,
        "caveat": (
            "Las contribuciones describen el comportamiento del modelo sobre los datos disponibles; "
            "no establecen relaciones causales entre clima y casos."
        ),
        "thresholds": thresholds,
        "is_synthetic_data": panel.is_synthetic,
    }


def _build_narrative(
    district: str,
    prediction: float,
    incidence: Optional[float],
    factors: List[Dict[str, Any]],
    thresholds: Dict[str, Any],
    horizon: int,
    method: str,
    is_synthetic: bool,
) -> str:
    from app.ml.inference import classify_risk

    level, _rationale = classify_risk(incidence, thresholds)
    level_es = {"LOW": "bajo", "MODERATE": "medio", "HIGH": "alto", "CRITICAL": "crítico", "UNKNOWN": "indeterminado"}[level]
    positives = [f for f in factors if f["contribution"] > 0][:3]
    negatives = [f for f in factors if f["contribution"] < 0][:3]
    parts = [
        f"Para {district}, el modelo estima {prediction:.1f} casos para las próximas {horizon} semana(s)",
    ]
    if incidence is not None:
        parts.append(f"equivalente a {incidence:.1f} casos por 100 000 habitantes, un nivel de riesgo {level_es}.")
    else:
        parts.append("sin denominador poblacional disponible para calcular incidencia.")
    if positives:
        detail = "; ".join(f"{f['label']} (+{f['contribution']:.2f})" for f in positives)
        parts.append(f"Los factores que más elevan la estimación son: {detail}.")
    if negatives:
        detail = "; ".join(f"{f['label']} ({f['contribution']:.2f})" for f in negatives)
        parts.append(f"Los factores que más la reducen son: {detail}.")
    parts.append(f"Atribución calculada con {method}.")
    if is_synthetic:
        parts.append("Los datos de entrada son sintéticos de demostración.")
    return " ".join(parts)
