"""Routers de administración, configuración y estado del sistema (M15)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import AuthContext, get_auth_context, get_db, require_permissions
from app.core.errors import NotFoundError, ValidationError
from app.models.epi import DataSource, DengueCase, ImportJob
from app.models.ml import Experiment, ModelVersion
from app.models.ops import Alert, Report, WorkflowExecution
from app.models.security import AuditEvent, SystemConfig, User
from app.schemas.common import ConfigValue, HealthStatus, Message
from app.services import audit as audit_service
from app.services import user_service

router = APIRouter(tags=["M15 Administración · Estado del sistema"])


@router.get("/health", response_model=HealthStatus, summary="Estado de la plataforma")
def health(db: Session = Depends(get_db)) -> HealthStatus:
    from app.ml.reproducibility import torch_info

    limitations: List[str] = []
    if settings.is_sqlite:
        limitations.append(
            "Base de datos SQLite para ejecución local sin infraestructura; el esquema es compatible con "
            "PostgreSQL/PostGIS (ver docs/architecture/base-de-datos.md)."
        )
    if not settings.POSTGIS_ENABLED:
        limitations.append("PostGIS deshabilitado: la geometría se sirve como GeoJSON almacenado.")
    from app.ml.features import resolve_data_mode

    xgb = _xgboost()
    torch_state = torch_info()
    if not torch_state.get("available"):
        limitations.append("PyTorch no disponible: los modelos neuronales (LSTM, ConvLSTM, CNN-LSTM, ConvLSTM-Attention) quedan inhabilitados.")
    if db is not None and resolve_data_mode(db) == "synthetic":
        limitations.append(
            "Los datos cargados por defecto son SINTÉTICOS de demostración y están identificados como tales."
        )

    return HealthStatus(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        database="ok" if db is not None else "error",
        database_url_scheme=settings.DATABASE_URL.split(":", 1)[0],
        postgis=settings.POSTGIS_ENABLED,
        ml_backend={"torch": torch_state, "xgboost": xgb},
        integrations={
            "nasa_power": "live" if settings.LIVE_INTEGRATIONS_ENABLED else "adaptador listo (desactivado)",
            "llm_provider": settings.LLM_PROVIDER,
            "mlflow": "instalado" if _has_mlflow() else "registro interno (sin MLflow)",
        },
        timestamp=datetime.now(timezone.utc),
        limitations=limitations,
    )


def _has_mlflow() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except Exception:
        return False


def _xgboost() -> Dict[str, Any]:
    try:
        import xgboost

        return {"available": True, "version": xgboost.__version__}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


@router.get("/admin/config", response_model=List[ConfigValue], summary="Configuración del sistema")
def list_config(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("admin:config"))):
    rows = list(db.scalars(select(SystemConfig).order_by(SystemConfig.key)))
    return [
        ConfigValue(
            key=row.key,
            value=row.value,
            description=row.description,
            updated_at=row.updated_at,
            updated_by=row.updated_by,
        )
        for row in rows
    ]


@router.put("/admin/config/{key}", response_model=ConfigValue, summary="Actualizar configuración")
def update_config(
    key: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("admin:config")),
):
    record = db.scalar(select(SystemConfig).where(SystemConfig.key == key))
    if record is None:
        raise NotFoundError("La clave de configuración no existe.")
    new_value = payload.get("value", payload)
    if key == "risk_thresholds":
        for field in ("low", "moderate", "high"):
            if field not in new_value:
                raise ValidationError(f"El umbral '{field}' es obligatorio.")
        if not (new_value["low"] < new_value["moderate"] < new_value["high"]):
            raise ValidationError("Los umbrales deben cumplir low < moderate < high.")
    previous = record.value
    record.value = new_value
    record.updated_by = auth.user.email
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M15", action="update_config",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        resource_type="config", resource_id=key,
        description=f"Configuración '{key}' actualizada.",
        context={"previous": previous, "new": new_value},
    )
    db.commit()
    return ConfigValue(
        key=record.key, value=record.value, description=record.description,
        updated_at=record.updated_at, updated_by=record.updated_by,
    )


@router.get("/admin/stats", summary="Estadísticas globales")
def stats(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("admin:config"))):
    def count(model) -> int:
        return int(db.scalar(select(func.count()).select_from(model)) or 0)

    return {
        "users": count(User),
        "active_users": int(db.scalar(select(func.count()).select_from(User).where(User.is_active.is_(True))) or 0),
        "roles": len(user_service.list_roles(db)),
        "sessions_active": user_service.active_session_count(db),
        "audit_events": count(AuditEvent),
        "cases": count(DengueCase),
        "imports": count(ImportJob),
        "data_sources": count(DataSource),
        "experiments": count(Experiment),
        "models": count(ModelVersion),
        "models_in_production": int(
            db.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.is_production.is_(True))) or 0
        ),
        "alerts": count(Alert),
        "reports": count(Report),
        "workflow_executions": count(WorkflowExecution),
    }
