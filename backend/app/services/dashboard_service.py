"""Dashboard ejecutivo (M04): indicadores epidemiológicos y operacionales."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.models.epi import ClimateObservation, DataSource, DengueCase, ImportJob
from app.models.geo import GeographicUnit
from app.models.ml import Experiment, MetricResult, ModelVersion, Prediction, RiskClassification
from app.models.ops import Alert, Report, WorkflowExecution
from app.models.security import UserSession


def _window_cases(db: Session, start: date, end: date, codes: Optional[List[str]] = None) -> int:
    stmt = select(func.coalesce(func.sum(DengueCase.confirmed_cases), 0)).where(
        and_(DengueCase.week_start >= start, DengueCase.week_start <= end)
    )
    if codes:
        stmt = stmt.where(DengueCase.district_code.in_(codes))
    return int(db.scalar(stmt) or 0)


def _resolve_districts(db: Session, department: Optional[str], province: Optional[str], district: Optional[str]) -> List[str]:
    stmt = select(GeographicUnit)
    if department:
        stmt = stmt.where(GeographicUnit.department == department)
    if province:
        stmt = stmt.where(GeographicUnit.province == province)
    if district:
        stmt = stmt.where(GeographicUnit.code == district)
    return [unit.code for unit in db.scalars(stmt)]


def overview(
    db: Session,
    *,
    department: Optional[str] = None,
    province: Optional[str] = None,
    district: Optional[str] = None,
    weeks: int = 52,
) -> Dict[str, Any]:
    codes = _resolve_districts(db, department, province, district)
    latest_week = db.scalar(select(func.max(DengueCase.week_start)))
    if latest_week is None:
        return {
            "generated_at": None,
            "data_freshness": None,
            "is_synthetic_data": True,
            "kpis": [],
            "weekly_series": [],
            "districts_top_incidence": [],
            "districts_top_risk": [],
            "risk_distribution": [],
            "active_alerts": 0,
            "alerts_by_level": [],
            "climate_summary": [],
            "model_performance": [],
            "integration_status": [],
            "data_quality": {},
            "empty": True,
        }

    current_start = latest_week - timedelta(weeks=12)
    previous_start = current_start - timedelta(weeks=12)
    current_cases = _window_cases(db, current_start, latest_week, codes)
    previous_cases = _window_cases(db, previous_start, current_start - timedelta(days=1), codes)
    change_pct = ((current_cases - previous_cases) / previous_cases * 100) if previous_cases else None

    week_cases = _window_cases(db, latest_week - timedelta(days=6), latest_week, codes)

    series_start = latest_week - timedelta(weeks=weeks)
    series_stmt = (
        select(DengueCase.week_start, func.sum(DengueCase.confirmed_cases))
        .where(DengueCase.week_start >= series_start)
        .group_by(DengueCase.week_start)
        .order_by(DengueCase.week_start)
    )
    if codes:
        series_stmt = series_stmt.where(DengueCase.district_code.in_(codes))
    weekly_series = [
        {"week": str(week), "cases": int(total or 0), "label": f"S{week.isocalendar().week:02d}"}
        for week, total in db.execute(series_stmt).all()
    ]

    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    incidence_stmt = (
        select(
            DengueCase.district_code,
            func.sum(DengueCase.confirmed_cases).label("total"),
        )
        .where(DengueCase.week_start >= current_start, DengueCase.week_start <= latest_week)
        .group_by(DengueCase.district_code)
        .order_by(func.sum(DengueCase.confirmed_cases).desc())
    )
    if codes:
        incidence_stmt = incidence_stmt.where(DengueCase.district_code.in_(codes))
    top_incidence = []
    for code, total in db.execute(incidence_stmt).all()[:10]:
        unit = units.get(code)
        population = float(unit.population or 0) if unit else 0.0
        rate = (float(total) / population * 100_000) if population else None
        top_incidence.append(
            {
                "district_code": code,
                "district_name": unit.name if unit else code,
                "province": unit.province if unit else "",
                "cases_12w": int(total or 0),
                "incidence_per_100k": round(rate, 1) if rate is not None else None,
                "population": int(population) if population else None,
            }
        )

    risk_stmt = (
        select(RiskClassification)
        .where(RiskClassification.epi_year == latest_week.isocalendar().year)
        .order_by(RiskClassification.score.desc())
    )
    latest_risks = list(db.scalars(risk_stmt))[:12]
    top_risk = [
        {
            "district_code": item.district_code,
            "district_name": units[item.district_code].name if item.district_code in units else item.district_code,
            "level": item.level,
            "score": round(float(item.score or 0), 2),
            "predicted_cases": item.predicted_cases,
            "incidence_per_100k": item.incidence_per_100k,
        }
        for item in latest_risks
    ]
    distribution: Dict[str, int] = {}
    for item in db.scalars(select(RiskClassification)):
        distribution[item.level] = distribution.get(item.level, 0) + 1

    alert_rows = list(db.scalars(select(Alert).order_by(Alert.generated_at.desc()).limit(200)))
    active_alerts = sum(1 for alert in alert_rows if alert.status in ("OPEN", "ACKNOWLEDGED"))
    alerts_by_level: Dict[str, int] = {}
    for alert in alert_rows:
        if alert.status in ("OPEN", "ACKNOWLEDGED"):
            alerts_by_level[alert.risk_level] = alerts_by_level.get(alert.risk_level, 0) + 1

    climate_stmt = (
        select(
            func.avg(ClimateObservation.temperature_mean),
            func.avg(ClimateObservation.humidity_mean),
            func.avg(ClimateObservation.precipitation_mm),
        ).where(ClimateObservation.week_start >= latest_week - timedelta(weeks=8))
    )
    temp_avg, humidity_avg, precip_avg = db.execute(climate_stmt).one()
    baseline_stmt = select(
        func.avg(ClimateObservation.temperature_mean),
        func.avg(ClimateObservation.humidity_mean),
        func.avg(ClimateObservation.precipitation_mm),
    ).where(ClimateObservation.week_start < latest_week - timedelta(weeks=8))
    temp_base, humidity_base, precip_base = db.execute(baseline_stmt).one()

    def _delta(current, baseline):
        if current is None or baseline in (None, 0):
            return None
        return round((float(current) - float(baseline)) / float(baseline) * 100, 1)

    climate_summary = [
        {
            "variable": "Temperatura media",
            "value": round(float(temp_avg), 2) if temp_avg is not None else None,
            "unit": "°C",
            "change_pct": _delta(temp_avg, temp_base),
        },
        {
            "variable": "Humedad relativa",
            "value": round(float(humidity_avg), 2) if humidity_avg is not None else None,
            "unit": "%",
            "change_pct": _delta(humidity_avg, humidity_base),
        },
        {
            "variable": "Precipitación semanal",
            "value": round(float(precip_avg), 2) if precip_avg is not None else None,
            "unit": "mm",
            "change_pct": _delta(precip_avg, precip_base),
        },
    ]

    production = list(db.scalars(select(ModelVersion).where(ModelVersion.is_production.is_(True))))
    model_performance = []
    for version in production:
        metrics = version.metrics or {}
        model_performance.append(
            {
                "algorithm": version.algorithm,
                "version": version.version,
                "horizon": (version.feature_names or {}).get("horizon"),
                "MAE": metrics.get("MAE"),
                "RMSE": metrics.get("RMSE"),
                "AUC_PR": metrics.get("AUC_PR"),
                "coverage": metrics.get("coverage"),
                "promoted_at": version.promoted_at.isoformat() if version.promoted_at else None,
            }
        )

    source_rows = list(db.scalars(select(DataSource)))
    integration_status = [
        {
            "code": source.code,
            "name": source.name,
            "kind": source.kind,
            "status": source.integration_status,
            "is_synthetic": source.is_synthetic,
            "last_success_at": source.last_success_at.isoformat() if source.last_success_at else None,
        }
        for source in source_rows
    ]

    synthetic_rows = db.scalar(select(func.count()).select_from(DengueCase).where(DengueCase.is_synthetic.is_(True))) or 0
    total_rows = db.scalar(select(func.count()).select_from(DengueCase)) or 0
    last_import = db.scalar(select(ImportJob).order_by(ImportJob.started_at.desc()).limit(1))

    kpis = [
        {
            "key": "cases_12w",
            "label": "Casos confirmados (12 semanas)",
            "value": current_cases,
            "unit": "casos",
            "change_pct": round(change_pct, 1) if change_pct is not None else None,
            "trend": "up" if (change_pct or 0) > 3 else ("down" if (change_pct or 0) < -3 else "flat"),
            "hint": f"Comparado con las 12 semanas previas ({previous_cases} casos)",
        },
        {
            "key": "cases_last_week",
            "label": "Casos en la última semana",
            "value": week_cases,
            "unit": "casos",
            "hint": f"Semana epidemiológica {latest_week.isocalendar().week} de {latest_week.isocalendar().year}",
        },
        {
            "key": "active_alerts",
            "label": "Alertas activas",
            "value": active_alerts,
            "unit": "alertas",
            "hint": "Alertas en estado abierto o reconocido",
        },
        {
            "key": "districts_at_risk",
            "label": "Distritos en riesgo alto o crítico",
            "value": sum(1 for item in latest_risks if item.level in ("HIGH", "CRITICAL")),
            "unit": "distritos",
            "hint": "Clasificación vigente por incidencia pronosticada",
        },
        {
            "key": "production_models",
            "label": "Modelos en producción",
            "value": len(production),
            "unit": "modelos",
            "hint": "Versiones promovidas por horizonte",
        },
        {
            "key": "synthetic_share",
            "label": "Proporción de datos sintéticos",
            "value": round(synthetic_rows / total_rows * 100, 1) if total_rows else 0.0,
            "unit": "%",
            "hint": "0 % indica datos cargados desde fuentes externas",
        },
    ]

    return {
        "generated_at": None,
        "data_freshness": latest_week,
        "is_synthetic_data": bool(total_rows and synthetic_rows == total_rows),
        "kpis": kpis,
        "weekly_series": weekly_series,
        "districts_top_incidence": top_incidence,
        "districts_top_risk": top_risk,
        "risk_distribution": [{"level": level, "count": count} for level, count in sorted(distribution.items())],
        "active_alerts": active_alerts,
        "alerts_by_level": [{"level": level, "count": count} for level, count in sorted(alerts_by_level.items())],
        "climate_summary": climate_summary,
        "model_performance": model_performance,
        "integration_status": integration_status,
        "data_quality": {
            "total_cases": int(total_rows),
            "synthetic_rows": int(synthetic_rows),
            "districts": len(units),
            "weeks": len({week for (week,) in db.execute(select(DengueCase.week_start).distinct()).all()}),
            "last_import": {
                "job_key": last_import.job_key,
                "status": last_import.status,
                "started_at": last_import.started_at.isoformat() if last_import.started_at else None,
                "rows_inserted": last_import.rows_inserted,
            }
            if last_import
            else None,
        },
        "operations": {
            "experiments": int(db.scalar(select(func.count()).select_from(Experiment)) or 0),
            "reports": int(db.scalar(select(func.count()).select_from(Report)) or 0),
            "workflow_runs": int(db.scalar(select(func.count()).select_from(WorkflowExecution)) or 0),
            "active_sessions": int(
                db.scalar(select(func.count()).select_from(UserSession).where(UserSession.revoked_at.is_(None))) or 0
            ),
        },
    }
