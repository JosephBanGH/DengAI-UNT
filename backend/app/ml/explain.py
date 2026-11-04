"""Explicabilidad (M09): SHAP, importancia temporal y narrativa verificable."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.features import build_features, load_panel, make_sample_tensors, to_matrix
from app.ml.inference import pick_version, risk_thresholds
from app.ml.models.base import ModelData
from app.ml.trainer import load_bundle
from app.models.geo import GeographicUnit

logger = logging.getLogger("dengai.explain")

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
    "humidity_mean": "Humedad relativa",
    "precipitation_mm": "Precipitación",
    "precipitation_mean_4": "Precipitación acumulada (4 semanas)",
    "precipitation_mean_12": "Precipitación acumulada (12 semanas)",
    "temperature_mean_4": "Temperatura media (4 semanas)",
    "temperature_mean_12": "Temperatura media (12 semanas)",
    "altitude_km": "Altitud",
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


def _shap_values(model, X: np.ndarray, background: np.ndarray) -> Optional[np.ndarray]:
    """SHAP para el modelo indicado; devuelve contribuciones por variable (1 fila)."""
    try:
        import shap

        algorithm = getattr(model, "key", "")
        if algorithm == "xgboost" and getattr(model, "model", None) is not None:
            explainer = shap.TreeExplainer(model.model)
            values = explainer.shap_values(X)
            return np.asarray(values).reshape(X.shape)
        background_sample = background[np.random.default_rng(0).choice(len(background), min(40, len(background)), replace=False)]

        def _predict(matrix: np.ndarray) -> np.ndarray:
            data = ModelData(X=matrix, seq=None, grid=None, cell=None, y=np.zeros(len(matrix)))
            return model.predict(data)

        explainer = shap.KernelExplainer(_predict, background_sample)
        values = explainer.shap_values(X, nsamples=120)
        return np.asarray(values).reshape(X.shape)
    except Exception:
        logger.exception("No fue posible calcular SHAP; se usa perturbación por permutación")
        return None


def _permutation_contributions(model, X: np.ndarray, mean_vector: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Contribución por perturbación: reemplaza cada variable por su media de entrenamiento."""
    base_pred = float(model.predict(ModelData(X=X, seq=None, grid=None, cell=None, y=np.zeros(len(X))))[0])
    contributions = np.zeros(len(feature_names))
    for idx in range(len(feature_names)):
        perturbed = X.copy()
        perturbed[0, idx] = mean_vector[idx]
        value = float(model.predict(ModelData(X=perturbed, seq=None, grid=None, cell=None, y=np.zeros(len(X))))[0])
        contributions[idx] = base_pred - value
    return contributions


def temporal_importance(model, seq: np.ndarray, cell: np.ndarray, window: int) -> List[Dict[str, Any]]:
    """Importancia temporal: impacto de neutralizar cada semana de la ventana."""
    base = float(model.predict(ModelData(X=None, seq=seq, grid=None, cell=cell, y=np.zeros(1)))[0])
    results = []
    for step in range(window):
        perturbed = seq.copy()
        perturbed[:, step, :] = 0.0
        value = float(model.predict(ModelData(X=None, seq=perturbed, grid=None, cell=cell, y=np.zeros(1)))[0])
        results.append({"step": step - window + 1, "delta": round(base - value, 4)})
    return results


def explain_district(
    db: Session,
    district_code: str,
    *,
    horizon: int = 1,
    model_version_id: Optional[int] = None,
    top_k: int = 8,
) -> Dict[str, Any]:
    unit = db.scalar(select(GeographicUnit).where(GeographicUnit.code == district_code))
    if unit is None:
        raise NotFoundError("El distrito indicado no existe.")
    version = pick_version(db, horizon, model_version_id)
    bundle = load_bundle(version.artifact_path)
    model = bundle["model"]
    columns: List[str] = bundle["feature_names"]
    mean = np.asarray(bundle["scaler_mean"])
    std = np.asarray(bundle["scaler_std"])

    panel = load_panel(db)
    frame = build_features(panel, horizon=horizon)
    origin = max(panel.weeks)
    subset = frame[(frame["origin_week"] == origin) & (frame["district_code"] == district_code)]
    if subset.empty:
        raise ValidationError("No hay datos suficientes para explicar este distrito en la última semana.")

    seq, grid, cell, _ctx = make_sample_tensors(panel, frame, subset, columns, mean, std)
    X = to_matrix(subset, columns, mean, std)
    data = ModelData(X=X, seq=seq, grid=grid, cell=cell, y=subset["target"].to_numpy(dtype=float), feature_names=columns)
    prediction = float(model.predict(data)[0])

    background = to_matrix(frame.sample(min(400, len(frame)), random_state=42), columns, mean, std)
    values = _shap_values(model, X, background)
    method = "SHAP"
    if values is None:
        values = _permutation_contributions(model, X, mean, columns).reshape(1, -1)
        method = "permutación (respuesta de contingencia)"

    contributions = np.asarray(values).reshape(-1)
    base_value = prediction - float(np.sum(contributions))
    order = np.argsort(-np.abs(contributions))[:top_k]
    factors = []
    for idx in order:
        contrib = float(contributions[idx])
        raw_value = subset[columns[idx]].iloc[0]
        factors.append(
            {
                "feature": columns[idx],
                "label": feature_label(columns[idx]),
                "contribution": round(contrib, 4),
                "direction": "aumenta" if contrib > 0 else "reduce",
                "value": None if pd.isna(raw_value) else round(float(raw_value), 4),
            }
        )

    temporal = []
    if seq is not None:
        try:
            temporal = temporal_importance(model, seq, cell, int(bundle.get("sequence_window", 12)))
        except Exception:
            temporal = []

    # Importancia temporal agregada por rezago (modelos tabulares).
    lag_importance = []
    for lag in range(1, 13):
        name = f"cases_lag_{lag}"
        if name in columns:
            lag_importance.append(
                {"lag": lag, "abs_contribution": round(abs(float(contributions[columns.index(name)])), 4)}
            )

    thresholds = risk_thresholds(db)
    population = float(unit.population or 0) or None
    incidence = (prediction / population * 100_000) if population else None
    narrative = _build_narrative(unit.name, prediction, incidence, factors, thresholds, horizon, method, panel.is_synthetic)

    return {
        "district_code": unit.code,
        "district_name": unit.name,
        "model_version": version.version,
        "algorithm": version.algorithm,
        "method": method,
        "base_value": round(base_value, 4),
        "prediction": round(prediction, 4),
        "factors": factors,
        "temporal_importance": temporal or lag_importance,
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

    level, rationale = classify_risk(incidence, thresholds)
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
