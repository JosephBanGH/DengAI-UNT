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
    for algorithm, version in selected.items():
        payload = _load_predictions(version)
        if payload is None:
            continue
        if actual is None:
            actual = payload["y_true"]
            blocks = [f"{district}|{week}" for district, week in zip(payload["district"], payload["target_week"])]
            aligned_meta = payload
        if len(payload["y_true"]) != len(actual):
            continue
        predictions[algorithm] = payload["y_pred"]

    if actual is None or len(predictions) < 2:
        raise ValidationError("No hay predicciones almacenadas comparables para ese horizonte.")

    stats = compare_models(predictions, actual, blocks)

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
                diebold_mariano(actual, predictions[winner], pred, winner, algorithm)
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
        "tests": [stats.get("friedman", {})] + stats.get("pairwise", []) + dm_tests,
        "winner": winner,
        "winner_basis": basis,
        "horizon": horizon,
        "n_observations": int(len(actual)),
        "caveat": (
            "La comparación utiliza las predicciones almacenadas del conjunto de prueba. "
            "Un MAE menor no implica superioridad causal; revise significancia y estabilidad temporal."
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
