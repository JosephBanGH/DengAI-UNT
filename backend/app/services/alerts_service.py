"""Alertas tempranas (M11): generación, seguimiento y cierre."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.models.geo import GeographicUnit
from app.models.ml import Prediction
from app.models.ops import Alert
from app.services import audit as audit_service

LEVEL_WEIGHT = {"LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4, "UNKNOWN": 0}


def _alert_key(district: str, year: int, week: int, horizon: int, level: str) -> str:
    raw = f"{district}|{year}|W{week:02d}|h{horizon}|{level}"
    return f"ALT-{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def generate_alerts(
    db: Session,
    *,
    run_key: Optional[str] = None,
    horizons: Optional[Iterable[int]] = None,
    min_level: str = "HIGH",
    actor: str = "system",
) -> List[Dict[str, Any]]:
    """Crea alertas a partir de los pronósticos persistidos, sin duplicar las existentes."""
    stmt = select(Prediction).order_by(Prediction.generated_at.desc())
    if run_key:
        stmt = stmt.where(Prediction.run_key == run_key)
    else:
        latest = db.scalar(select(func.max(Prediction.generated_at)))
        if latest is None:
            raise ValidationError("No hay pronósticos registrados. Genere una inferencia primero.")
        stmt = stmt.where(Prediction.generated_at == latest)
    if horizons:
        stmt = stmt.where(Prediction.horizon_weeks.in_(list(horizons)))

    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    threshold_weight = LEVEL_WEIGHT.get(min_level.upper(), 3)
    created: List[Dict[str, Any]] = []

    for prediction in db.scalars(stmt):
        if LEVEL_WEIGHT.get(prediction.risk_level, 0) < threshold_weight:
            continue
        key = _alert_key(
            prediction.district_code, prediction.target_year, prediction.target_week,
            prediction.horizon_weeks, prediction.risk_level,
        )
        existing = db.scalar(select(Alert).where(Alert.alert_key == key))
        if existing is not None:
            if existing.status == "CLOSED":
                continue
            created.append(_payload(existing, units))
            continue
        unit = units.get(prediction.district_code)
        alert = Alert(
            alert_key=key,
            district_code=prediction.district_code,
            epi_year=prediction.target_year,
            epi_week=prediction.target_week,
            horizon_weeks=prediction.horizon_weeks,
            risk_level=prediction.risk_level,
            risk_score=prediction.risk_score,
            predicted_cases=prediction.predicted_cases,
            lower_bound=prediction.lower_bound,
            upper_bound=prediction.upper_bound,
            incidence_per_100k=prediction.incidence_per_100k,
            factors={"district": unit.name if unit else prediction.district_code},
            rationale=(
                f"Riesgo {prediction.risk_level} para {unit.name if unit else prediction.district_code} "
                f"en la semana {prediction.target_week} de {prediction.target_year}: "
                f"{prediction.predicted_cases:.1f} casos pronosticados "
                f"(IC 90 %: {prediction.lower_bound:.1f}–{prediction.upper_bound:.1f})."
            ),
            status="OPEN",
            model_version_id=prediction.model_version_id,
            prediction_run_key=prediction.run_key,
            is_synthetic_data=prediction.is_synthetic_data,
            notification_channels={"channels": ["in_app"], "sent": False},
        )
        db.add(alert)
        db.flush()
        created.append(_payload(alert, units))

    audit_service.record(
        db, event_type=audit_service.AuditEventType.CREATE, module="M11", action="generate_alerts",
        result="SUCCESS", user_email=actor, resource_type="alert",
        description=f"Se generaron {len(created)} alertas con nivel mínimo {min_level}.",
        context={"min_level": min_level, "run_key": run_key},
    )
    db.commit()
    return created


def _payload(alert: Alert, units: Dict[str, GeographicUnit]) -> Dict[str, Any]:
    unit = units.get(alert.district_code)
    return {
        "id": alert.id,
        "alert_key": alert.alert_key,
        "district_code": alert.district_code,
        "district_name": unit.name if unit else alert.district_code,
        "epi_year": alert.epi_year,
        "epi_week": alert.epi_week,
        "horizon_weeks": alert.horizon_weeks,
        "risk_level": alert.risk_level,
        "risk_score": alert.risk_score,
        "predicted_cases": alert.predicted_cases,
        "lower_bound": alert.lower_bound,
        "upper_bound": alert.upper_bound,
        "incidence_per_100k": alert.incidence_per_100k,
        "rationale": alert.rationale,
        "status": alert.status,
        "acknowledged_by": alert.acknowledged_by,
        "acknowledged_at": alert.acknowledged_at,
        "closed_by": alert.closed_by,
        "closed_at": alert.closed_at,
        "notes": alert.notes,
        "generated_at": alert.generated_at,
        "is_synthetic_data": alert.is_synthetic_data,
        "factors": alert.factors,
    }


def list_alerts(
    db: Session,
    *,
    status: Optional[str] = None,
    level: Optional[str] = None,
    district: Optional[str] = None,
    page: int = 1,
    size: int = 50,
) -> Dict[str, Any]:
    stmt = select(Alert)
    if status:
        stmt = stmt.where(Alert.status == status.upper())
    if level:
        stmt = stmt.where(Alert.risk_level == level.upper())
    if district:
        stmt = stmt.where(Alert.district_code == district)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    rows = [
        _payload(alert, units)
        for alert in db.scalars(stmt.order_by(Alert.generated_at.desc()).offset((page - 1) * size).limit(size))
    ]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def update_alert(db: Session, alert_id: int, action: str, notes: str, *, actor: str) -> Dict[str, Any]:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise NotFoundError("La alerta no existe.")
    now = datetime.now(timezone.utc)
    if action == "acknowledge":
        if alert.status not in ("OPEN",):
            raise ValidationError("Solo se pueden reconocer alertas abiertas.")
        alert.status = "ACKNOWLEDGED"
        alert.acknowledged_by = actor
        alert.acknowledged_at = now
    elif action == "close":
        if alert.status == "CLOSED":
            raise ValidationError("La alerta ya está cerrada.")
        alert.status = "CLOSED"
        alert.closed_by = actor
        alert.closed_at = now
    elif action == "reopen":
        alert.status = "OPEN"
        alert.closed_by = ""
        alert.closed_at = None
    else:
        raise ValidationError("Acción no válida. Use acknowledge, close o reopen.")
    if notes:
        alert.notes = f"{alert.notes}\n{notes}".strip() if alert.notes else notes
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M11", action=f"alert_{action}",
        result="SUCCESS", user_email=actor, resource_type="alert", resource_id=alert.id,
        description=f"Alerta {action} para {alert.district_code} (S{alert.epi_week}).",
        context={"notes": notes[:300]},
    )
    db.commit()
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    return _payload(alert, units)


def alert_summary(db: Session) -> Dict[str, Any]:
    by_status = {
        status: int(count)
        for status, count in db.execute(select(Alert.status, func.count()).group_by(Alert.status)).all()
    }
    by_level = {
        level: int(count)
        for level, count in db.execute(select(Alert.risk_level, func.count()).group_by(Alert.risk_level)).all()
    }
    return {"by_status": by_status, "by_level": by_level, "total": sum(by_status.values())}
