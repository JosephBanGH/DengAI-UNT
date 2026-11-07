"""Comparación de modelos con pruebas estadísticas (M08)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.stats_tests import compare_models, diebold_mariano
from app.ml.trainer import load_bundle
from app.models.ml import Experiment, MetricResult, ModelVersion


def _load_predictions(version: ModelVersion) -> Optional[Dict[str, Any]]:
    bundle = load_bundle(version.artifact_path) if version.artifact_path and os.path.exists(version.artifact_path) else None
    if not bundle:
        return None
    path = bundle.get("predictions_path")
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "district": data["district"].astype(str),
        "target_week": data["target_week"].astype(str),
        "horizon": int((version.feature_names or {}).get("horizon") or 0),
        "algorithm": version.algorithm,
        "threshold": bundle.get("outbreak_threshold"),
        "data_mode": bundle.get("data_mode"),
        "is_synthetic": bundle.get("is_synthetic_data"),
    }


def compare(db: Session, *, horizon: int = 1) -> Dict[str, Any]:
    """Compara las versiones más recientes de cada algoritmo para un horizonte dado."""
    versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())))
    selected: Dict[str, ModelVersion] = {}
    for version in versions:
        if int((version.feature_names or {}).get("horizon") or -1) != horizon:
            continue
        selected.setdefault(version.algorithm, version)
    if len(selected) < 2:
        raise ValidationError(
            "Se requieren al menos dos modelos entrenados para el mismo horizonte para comparar.",
            details={"horizon": horizon, "algorithms": sorted(selected)},
        )

    predictions: Dict[str, np.ndarray] = {}
    actual = None
    blocks: List[str] = []
    aligned_meta = None
    threshold: Optional[float] = None
    for algorithm, version in selected.items():
        payload = _load_predictions(version)
        if payload is None:
            continue
        if actual is None:
            actual = payload["y_true"]
            # Bloques independientes: semana objetivo (agrega distritos de la misma
            # semana). Nunca una observación distrito-semana por bloque.
            blocks = [str(week) for week in payload["target_week"]]
            aligned_meta = payload
            threshold = payload.get("threshold")
        if len(payload["y_true"]) != len(actual):
            continue
        predictions[algorithm] = payload["y_pred"]

    if actual is None or len(predictions) < 2:
        raise ValidationError("No hay predicciones almacenadas comparables para ese horizonte.")

    binary_actual = None
    binary_predictions: Dict[str, np.ndarray] = {}
    if threshold is not None:
        binary_actual = (np.asarray(actual) >= float(threshold)).astype(int)
        binary_predictions = {
            algorithm: (np.asarray(pred) >= float(threshold)).astype(int)
            for algorithm, pred in predictions.items()
        }
    stats = compare_models(
        predictions,
        actual,
        blocks,
        binary_actual=binary_actual,
        binary_predictions=binary_predictions,
        # DeLong solo usa el orden de los puntajes; se documenta que no son
        # probabilidades calibradas sino el conteo pronosticado como score de riesgo.
        probabilities=predictions if binary_actual is not None else None,
        block_unit="target_week",
    )

    rows = []
    ranking = []
    for algorithm, pred in predictions.items():
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        rows.append({"algorithm": algorithm, "split": "test", "horizon_weeks": horizon, "metric": "MAE", "value": round(mae, 4), "n_observations": int(len(actual))})
        rows.append({"algorithm": algorithm, "split": "test", "horizon_weeks": horizon, "metric": "RMSE", "value": round(rmse, 4), "n_observations": int(len(actual))})
        ranking.append({"algorithm": algorithm, "MAE": round(mae, 4), "RMSE": round(rmse, 4)})
    ranking.sort(key=lambda item: item["MAE"])

    winner = ranking[0]["algorithm"] if ranking else None
    dm_tests = []
    if winner and aligned_meta is not None:
        for algorithm, pred in predictions.items():
            if algorithm == winner:
                continue
            dm_tests.append(
                diebold_mariano(actual, predictions[winner], pred, winner, algorithm, blocks=blocks)
            )

    basis = (
        f"El modelo {winner} obtuvo el MAE más bajo ({ranking[0]['MAE']}) sobre el conjunto de prueba "
        f"con {len(actual)} observaciones distrito-semana. "
        if ranking
        else ""
    )
    friedman = stats.get("friedman", {})
    if isinstance(friedman, dict) and friedman.get("applicable"):
        basis += (
            "La prueba de Friedman "
            + ("detectó diferencias significativas" if friedman.get("significant") else "no detectó diferencias significativas")
            + f" (p = {friedman.get('p_value'):.4f})."
        )
    return {
        "rows": rows,
        "ranking": ranking,
        "tests": [stats.get("friedman", {})]
        + stats.get("pairwise", [])
        + stats.get("mcnemar", [])
        + stats.get("delong", [])
        + dm_tests,
        "blocks": stats.get("blocks"),
        "block_unit": stats.get("block_unit"),
        "assumptions": stats.get("assumptions"),
        "winner": winner,
        "winner_basis": basis,
        "horizon": horizon,
        "n_observations": int(len(actual)),
        "caveat": (
            "La comparación utiliza las predicciones almacenadas del conjunto de prueba. "
            "Las pruebas de Friedman/Wilcoxon agregan pérdidas por semana objetivo como bloque "
            "independiente. Un MAE menor no implica superioridad causal; revise significancia, "
            "tamaño del efecto y estabilidad temporal."
        ),
    }


def metric_matrix(db: Session, *, horizon: Optional[int] = None) -> List[Dict[str, Any]]:
    stmt = select(MetricResult)
    if horizon:
        stmt = stmt.where(MetricResult.horizon_weeks == horizon)
    return [
        {
            "algorithm": row.algorithm,
            "split": row.split,
            "horizon_weeks": row.horizon_weeks,
            "metric": row.metric,
            "value": row.value,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "n_observations": row.n_observations,
        }
        for row in db.scalars(stmt.order_by(MetricResult.algorithm, MetricResult.metric))
    ]
