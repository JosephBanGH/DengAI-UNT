"""Inferencia: pronósticos multihorizonte, intervalos y clasificación de riesgo (M07/M11)."""
from __future__ import annotations

import logging
import math
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.features import build_features, load_panel, make_sample_tensors, to_matrix
from app.ml.models.base import ModelData
from app.ml.trainer import load_bundle
from app.models.geo import GeographicUnit
from app.models.ml import ModelVersion, Prediction, RiskClassification
from app.models.security import SystemConfig

logger = logging.getLogger("dengai.inference")

RISK_ORDER = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 0}


def risk_thresholds(db: Session) -> Dict[str, Any]:
    config = db.scalar(select(SystemConfig).where(SystemConfig.key == "risk_thresholds"))
    if config is None:
        return {"low": 5.0, "moderate": 20.0, "high": 50.0, "method": "incidencia_semanal_por_100k", "justification": ""}
    value = dict(config.value or {})
    value.setdefault("low", 5.0)
    value.setdefault("moderate", 20.0)
    value.setdefault("high", 50.0)
    return value


def classify_risk(incidence_per_100k: Optional[float], thresholds: Dict[str, Any]) -> Tuple[str, str]:
    if incidence_per_100k is None or not math.isfinite(incidence_per_100k):
        return "UNKNOWN", "Sin denominador poblacional para calcular la incidencia."
    if incidence_per_100k >= thresholds["high"]:
        return "CRITICAL", (
            f"Incidencia pronosticada de {incidence_per_100k:.1f} casos por 100 000 hab., "
            f"por encima del umbral crítico de {thresholds['high']:.0f}."
        )
    if incidence_per_100k >= thresholds["moderate"]:
        return "HIGH", (
            f"Incidencia pronosticada de {incidence_per_100k:.1f} casos por 100 000 hab., "
            f"sobre el umbral alto de {thresholds['moderate']:.0f}."
        )
    if incidence_per_100k >= thresholds["low"]:
        return "MODERATE", (
            f"Incidencia pronosticada de {incidence_per_100k:.1f} casos por 100 000 hab., "
            f"sobre el umbral bajo de {thresholds['low']:.0f}."
        )
    return "LOW", (
        f"Incidencia pronosticada de {incidence_per_100k:.1f} casos por 100 000 hab., "
        f"por debajo del umbral bajo de {thresholds['low']:.0f}."
    )


def production_versions(db: Session) -> List[ModelVersion]:
    return list(
        db.scalars(
            select(ModelVersion).where(ModelVersion.is_production.is_(True)).order_by(ModelVersion.created_at.desc())
        )
    )


def _horizon_of(version: ModelVersion) -> Optional[int]:
    payload = version.feature_names or {}
    value = payload.get("horizon")
    return int(value) if value is not None else None


def pick_version(db: Session, horizon: int, model_version_id: Optional[int] = None) -> ModelVersion:
    if model_version_id:
        version = db.get(ModelVersion, model_version_id)
        if version is None:
            raise NotFoundError("La versión de modelo indicada no existe.")
        if _horizon_of(version) not in (None, horizon,):
            raise ValidationError("La versión de modelo indicada fue entrenada para otro horizonte.")
        return version
    for version in production_versions(db):
        if _horizon_of(version) == horizon:
            return version
    raise ValidationError(
        "No hay modelo en producción para el horizonte solicitado. "
        "Entrene un experimento en el Laboratorio de IA y promueva la versión resultante.",
        details={"horizon": horizon},
    )


def _last_origin_week(panel) -> date:
    return max(panel.weeks)


def generate_forecast(
    db: Session,
    *,
    horizons: Sequence[int],
    district_codes: Optional[Sequence[str]] = None,
    model_version_id: Optional[int] = None,
    user_email: str = "system",
    persist: bool = True,
) -> Dict[str, Any]:
    """Genera pronósticos para la última semana con datos usando los modelos en producción."""
    panel = load_panel(db)
    thresholds = risk_thresholds(db)
    origin_week = _last_origin_week(panel)
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    run_key = f"run-{uuid.uuid4().hex[:12]}"
    rows: List[Dict[str, Any]] = []
    used_versions: List[str] = []
    algorithm_used = ""
    is_synthetic = panel.is_synthetic

    for horizon in horizons:
        version = pick_version(db, int(horizon), model_version_id)
        bundle = load_bundle(version.artifact_path)
        model = bundle["model"]
        columns = bundle["feature_names"]
        mean = np.asarray(bundle["scaler_mean"])
        std = np.asarray(bundle["scaler_std"])
        interval = bundle.get("interval") or {"low": -2.0, "high": 2.0, "level": 0.9}

        frame = build_features(panel, horizon=int(horizon))
        subset = frame[frame["origin_week"] == origin_week].copy()
        if district_codes:
            subset = subset[subset["district_code"].isin(district_codes)]
        if subset.empty:
            continue

        seq, grid, cell, _context = make_sample_tensors(panel, frame, subset, columns, mean, std)
        data = ModelData(
            X=to_matrix(subset, columns, mean, std),
            seq=seq,
            grid=grid,
            cell=cell,
            y=subset["target"].to_numpy(dtype=float),
            feature_names=columns,
        )
        predicted = model.predict(data)
        used_versions.append(f"{version.algorithm}:{version.version}")
        algorithm_used = version.algorithm

        for idx, record in enumerate(subset.itertuples(index=False)):
            unit = units.get(record.district_code)
            population = float(unit.population) if unit and unit.population else None
            value = float(predicted[idx])
            lower = max(0.0, value + float(interval.get("low", -2.0)))
            upper = max(lower, value + float(interval.get("high", 2.0)))
            incidence = (value / population * 100_000) if population else None
            level, rationale = classify_risk(incidence, thresholds)
            target_start = origin_week + timedelta(weeks=int(horizon))
            rows.append(
                {
                    "district_code": record.district_code,
                    "district_name": unit.name if unit else record.district_code,
                    "target_week_start": target_start,
                    "target_year": target_start.isocalendar().year,
                    "target_week": target_start.isocalendar().week,
                    "horizon_weeks": int(horizon),
                    "predicted_cases": round(value, 3),
                    "lower_bound": round(lower, 3),
                    "upper_bound": round(upper, 3),
                    "risk_level": level,
                    "risk_score": round(incidence, 4) if incidence is not None else None,
                    "incidence_per_100k": round(incidence, 4) if incidence is not None else None,
                    "population": int(population) if population else None,
                    "rationale": rationale,
                    "model_version_id": version.id,
                    "algorithm": version.algorithm,
                    "model_version": version.version,
                }
            )

    if not rows:
        raise ValidationError("No fue posible generar pronósticos con los datos disponibles.")

    if persist:
        _persist(db, run_key, rows, origin_week, is_synthetic, user_email)

    return {
        "run_key": run_key,
        "model_version": ", ".join(sorted(set(used_versions))),
        "algorithm": algorithm_used,
        "generated_at": datetime.now(timezone.utc),
        "origin_week": origin_week.isocalendar().year * 100 + origin_week.isocalendar().week,
        "origin_week_start": origin_week,
        "is_synthetic_data": is_synthetic,
        "predictions": rows,
        "thresholds": thresholds,
        "disclaimer": (
            "Pronóstico de apoyo a la decisión epidemiológica. No constituye diagnóstico clínico "
            "ni sustituye la vigilancia epidemiológica oficial."
            + (" Los datos usados son SINTÉTICOS de demostración." if is_synthetic else "")
        ),
    }


def _persist(db: Session, run_key: str, rows: List[Dict[str, Any]], origin_week: date, is_synthetic: bool, user_email: str) -> None:
    for row in rows:
        db.add(
            Prediction(
                run_key=run_key,
                model_version_id=row["model_version_id"],
                district_code=row["district_code"],
                origin_year=origin_week.isocalendar().year,
                origin_week=origin_week.isocalendar().week,
                target_year=row["target_year"],
                target_week=row["target_week"],
                target_week_start=row["target_week_start"],
                horizon_weeks=row["horizon_weeks"],
                predicted_cases=row["predicted_cases"],
                lower_bound=row["lower_bound"],
                upper_bound=row["upper_bound"],
                interval_level=0.9,
                risk_level=row["risk_level"],
                risk_score=row["risk_score"],
                incidence_per_100k=row["incidence_per_100k"],
                is_synthetic_data=is_synthetic,
            )
        )
        existing = db.scalar(
            select(RiskClassification).where(
                RiskClassification.district_code == row["district_code"],
                RiskClassification.epi_year == row["target_year"],
                RiskClassification.epi_week == row["target_week"],
            )
        )
        values = {
            "level": row["risk_level"],
            "score": float(row["risk_score"] or 0.0),
            "method": "incidencia_semanal_por_100k",
            "rationale": row["rationale"],
            "predicted_cases": row["predicted_cases"],
            "incidence_per_100k": row["incidence_per_100k"],
            "factors": {"horizon_weeks": row["horizon_weeks"], "model_version": row["model_version"]},
            "model_version_id": row["model_version_id"],
        }
        if existing is None:
            db.add(
                RiskClassification(
                    district_code=row["district_code"],
                    epi_year=row["target_year"],
                    epi_week=row["target_week"],
                    **values,
                )
            )
        else:
            for key, value in values.items():
                setattr(existing, key, value)
    db.commit()


def latest_forecast(db: Session, *, horizon: Optional[int] = None) -> List[Dict[str, Any]]:
    stmt = select(Prediction).order_by(Prediction.generated_at.desc())
    if horizon:
        stmt = stmt.where(Prediction.horizon_weeks == horizon)
    predictions = list(db.scalars(stmt.limit(500)))
    if not predictions:
        return []
    latest_run = predictions[0].run_key
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    rows = []
    for item in predictions:
        if item.run_key != latest_run:
            continue
        unit = units.get(item.district_code)
        rows.append(
            {
                "district_code": item.district_code,
                "district_name": unit.name if unit else item.district_code,
                "target_week_start": item.target_week_start,
                "target_year": item.target_year,
                "target_week": item.target_week,
                "horizon_weeks": item.horizon_weeks,
                "predicted_cases": item.predicted_cases,
                "lower_bound": item.lower_bound,
                "upper_bound": item.upper_bound,
                "risk_level": item.risk_level,
                "risk_score": item.risk_score,
                "incidence_per_100k": item.incidence_per_100k,
            }
        )
    rows.sort(key=lambda r: (r["horizon_weeks"], -float(r["predicted_cases"] or 0)))
    return rows
