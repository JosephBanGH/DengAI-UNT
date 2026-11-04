"""Routers de dashboard, mapas y pronósticos (M04, M07, M10)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, get_auth_context, get_db, require_permissions
from app.ml.inference import generate_forecast, latest_forecast, risk_thresholds
from app.schemas.ml import ForecastRequest, ForecastResponse, PredictionOut, RiskMatrixResponse, RiskRow, ShapFactor
from app.schemas.ops import DashboardOverview
from app.services import audit as audit_service
from app.services import dashboard_service, geo_service

router = APIRouter(tags=["M04 Dashboard · M10 Mapas · M07 Pronósticos"])


@router.get("/dashboard/overview", response_model=DashboardOverview, summary="Indicadores del dashboard")
def overview(
    department: Optional[str] = None,
    province: Optional[str] = None,
    district: Optional[str] = None,
    weeks: int = Query(52, ge=8, le=260),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("dashboard:read")),
):
    data = dashboard_service.overview(db, department=department, province=province, district=district, weeks=weeks)
    return DashboardOverview(generated_at=datetime.now(timezone.utc), **{k: v for k, v in data.items() if k != "generated_at"})


@router.get("/geo/units", summary="Unidades geográficas")
def units(
    department: Optional[str] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("geo:read")),
):
    return geo_service.list_units(db, department=department)


@router.get("/geo/geojson", summary="GeoJSON de distritos")
def geojson(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("geo:read"))):
    return geo_service.geojson(db)


@router.get("/geo/risk-map", summary="Mapa de riesgo pronosticado")
def risk_map(
    horizon: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("geo:read")),
):
    return geo_service.risk_map(db, horizon=horizon)


@router.get("/geo/incidence-map", summary="Mapa de incidencia histórica")
def incidence_map(
    weeks: int = Query(4, ge=1, le=52),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("geo:read")),
):
    return geo_service.incidence_map(db, weeks=weeks)


@router.get("/geo/territory/{district_code}", summary="Serie territorial de un distrito")
def territory(
    district_code: str,
    weeks: int = Query(52, ge=8, le=260),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("geo:read")),
):
    return geo_service.territory_series(db, district_code, weeks=weeks)


@router.post("/forecasts/generate", summary="Generar pronósticos multihorizonte")
def generate(
    payload: ForecastRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("predictions:run")),
):
    result = generate_forecast(
        db,
        horizons=payload.horizons or [1],
        district_codes=payload.district_codes,
        model_version_id=payload.model_version_id,
        user_email=auth.user.email,
    )
    audit_service.record(
        db, event_type=audit_service.AuditEventType.MODEL_EXECUTION, module="M07", action="generate_forecast",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        description=f"Inferencia generada para horizontes {payload.horizons} ({len(result['predictions'])} registros).",
    )
    db.commit()
    return result


@router.get("/forecasts/latest", summary="Últimos pronósticos persistidos")
def latest(
    horizon: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("predictions:read")),
):
    return {"items": latest_forecast(db, horizon=horizon)}


@router.get("/risk/current", response_model=RiskMatrixResponse, summary="Matriz de riesgo vigente")
def current_risk(
    horizon: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("predictions:read")),
):
    rows = latest_forecast(db, horizon=horizon)
    thresholds = risk_thresholds(db)
    summary: Dict[str, int] = {}
    for row in rows:
        summary[row["risk_level"]] = summary.get(row["risk_level"], 0) + 1
    return RiskMatrixResponse(
        generated_at=datetime.now(timezone.utc),
        method=thresholds.get("method", "incidencia_semanal_por_100k"),
        thresholds={"low": thresholds["low"], "moderate": thresholds["moderate"], "high": thresholds["high"]},
        rows=[
            RiskRow(
                district_code=row["district_code"],
                district_name=row["district_name"],
                level=row["risk_level"],
                score=float(row["risk_score"] or 0),
                predicted_cases=row["predicted_cases"],
                incidence_per_100k=row["incidence_per_100k"],
                population=None,
                rationale="",
                factors={"horizon_weeks": row["horizon_weeks"]},
            )
            for row in rows
        ],
        summary=summary,
    )
