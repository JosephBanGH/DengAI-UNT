"""Servicio geoespacial (M10): GeoJSON, mapas de riesgo e indicadores territoriales."""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.ml.features import resolve_data_mode
from app.ml.inference import risk_thresholds
from app.models.epi import DengueCase
from app.models.geo import GeographicUnit
from app.models.ml import Prediction, RiskClassification


def list_units(db: Session, *, department: Optional[str] = None, level: str = "district") -> List[Dict[str, Any]]:
    stmt = select(GeographicUnit).where(GeographicUnit.level == level)
    if department:
        stmt = stmt.where(GeographicUnit.department == department)
    return [
        {
            "code": unit.code,
            "name": unit.name,
            "level": unit.level,
            "province": unit.province,
            "department": unit.department,
            "population": unit.population,
            "area_km2": unit.area_km2,
            "altitude_m": unit.altitude_m,
            "centroid": {"lat": unit.centroid_lat, "lon": unit.centroid_lon},
        }
        for unit in db.scalars(stmt.order_by(GeographicUnit.name))
    ]


def geojson(db: Session) -> Dict[str, Any]:
    features = []
    for unit in db.scalars(select(GeographicUnit)):
        geometry = None
        if unit.geometry_geojson:
            try:
                geometry = json.loads(unit.geometry_geojson)
            except json.JSONDecodeError:
                geometry = None
        features.append(
            {
                "type": "Feature",
                "id": unit.code,
                "geometry": geometry,
                "properties": {
                    "code": unit.code,
                    "name": unit.name,
                    "province": unit.province,
                    "department": unit.department,
                    "population": unit.population,
                    "area_km2": unit.area_km2,
                    "altitude_m": unit.altitude_m,
                    "lat": unit.centroid_lat,
                    "lon": unit.centroid_lon,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def risk_map(db: Session, *, horizon: Optional[int] = None) -> Dict[str, Any]:
    thresholds = risk_thresholds(db)
    latest_prediction = db.scalar(select(func.max(Prediction.generated_at)))
    predictions: Dict[str, Prediction] = {}
    if latest_prediction is not None:
        stmt = select(Prediction).where(Prediction.generated_at == latest_prediction)
        if horizon:
            stmt = stmt.where(Prediction.horizon_weeks == horizon)
        for item in db.scalars(stmt):
            predictions[item.district_code] = item

    risk_by_code: Dict[str, RiskClassification] = {}
    for item in db.scalars(select(RiskClassification).order_by(RiskClassification.created_at.desc()).limit(400)):
        risk_by_code.setdefault(item.district_code, item)

    features = []
    collection = geojson(db)
    for feature in collection["features"]:
        code = feature["properties"]["code"]
        prediction = predictions.get(code)
        risk = risk_by_code.get(code)
        properties = dict(feature["properties"])
        properties.update(
            {
                "risk_level": (prediction.risk_level if prediction else (risk.level if risk else "UNKNOWN")),
                "risk_score": (
                    prediction.incidence_per_100k if prediction and prediction.incidence_per_100k is not None
                    else (risk.incidence_per_100k if risk else None)
                ),
                "predicted_cases": prediction.predicted_cases if prediction else (risk.predicted_cases if risk else None),
                "lower_bound": prediction.lower_bound if prediction else None,
                "upper_bound": prediction.upper_bound if prediction else None,
                "horizon_weeks": prediction.horizon_weeks if prediction else None,
                "rationale": risk.rationale if risk else "",
            }
        )
        feature["properties"] = properties
        features.append(feature)
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "thresholds": thresholds,
            "generated_at": latest_prediction.isoformat() if latest_prediction else None,
            "legend": [
                {"level": "LOW", "label": "Bajo", "min": 0, "max": thresholds["low"]},
                {"level": "MODERATE", "label": "Medio", "min": thresholds["low"], "max": thresholds["moderate"]},
                {"level": "HIGH", "label": "Alto", "min": thresholds["moderate"], "max": thresholds["high"]},
                {"level": "CRITICAL", "label": "Crítico", "min": thresholds["high"], "max": None},
            ],
        },
    }


def incidence_map(db: Session, *, weeks: int = 4) -> Dict[str, Any]:
    use_synthetic = resolve_data_mode(db) == "synthetic"
    latest = db.scalar(
        select(func.max(DengueCase.week_start)).where(DengueCase.is_synthetic.is_(use_synthetic))
    )
    if latest is None:
        return {"type": "FeatureCollection", "features": [], "metadata": {"window_weeks": weeks}}
    start = latest - timedelta(weeks=weeks)
    totals = {
        code: int(total or 0)
        for code, total in db.execute(
            select(DengueCase.district_code, func.sum(DengueCase.confirmed_cases))
            .where(DengueCase.week_start > start, DengueCase.is_synthetic.is_(use_synthetic))
            .group_by(DengueCase.district_code)
        ).all()
    }
    collection = geojson(db)
    for feature in collection["features"]:
        code = feature["properties"]["code"]
        population = feature["properties"].get("population") or 0
        cases = totals.get(code, 0)
        feature["properties"]["cases"] = cases
        feature["properties"]["incidence_per_100k"] = (
            round(cases / population * 100_000, 2) if population else None
        )
    return {
        "type": "FeatureCollection",
        "features": collection["features"],
        "metadata": {"window_weeks": weeks, "end_week": latest.isoformat()},
    }


def territory_series(db: Session, district_code: str, *, weeks: int = 52) -> Dict[str, Any]:
    unit = db.scalar(select(GeographicUnit).where(GeographicUnit.code == district_code))
    if unit is None:
        return {"district_code": district_code, "series": [], "unit": None}
    use_synthetic = resolve_data_mode(db) == "synthetic"
    latest = db.scalar(
        select(func.max(DengueCase.week_start)).where(
            DengueCase.district_code == district_code,
            DengueCase.is_synthetic.is_(use_synthetic),
        )
    )
    if latest is None:
        return {"district_code": district_code, "series": [], "unit": {"code": unit.code, "name": unit.name}}
    start = latest - timedelta(weeks=weeks)
    rows = db.execute(
        select(DengueCase.week_start, DengueCase.confirmed_cases)
        .where(
            and_(
                DengueCase.district_code == district_code,
                DengueCase.week_start > start,
                DengueCase.is_synthetic.is_(use_synthetic),
            )
        )
        .order_by(DengueCase.week_start)
    ).all()
    return {
        "district_code": district_code,
        "unit": {"code": unit.code, "name": unit.name, "population": unit.population, "province": unit.province},
        "series": [{"week": str(week), "cases": int(cases)} for week, cases in rows],
    }
