"""Routers del laboratorio de IA (M07, M08, M09)."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.deps import AuthContext, get_auth_context, get_db, require_permissions
from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.ml.explain import explain_district
from app.ml.models.registry import algorithm_metadata, get_algorithm, get_search_space
from app.ml.trainer import is_running, submit_experiment
from app.ml.reproducibility import torch_info
from app.models.ml import Experiment, MetricResult, ModelVersion
from app.schemas.common import Message, Page
from app.schemas.ml import (
    AlgorithmInfo,
    ExperimentCreate,
    ExperimentOut,
    ExplanationResponse,
    MetricRow,
    ModelComparison,
    ModelVersionOut,
)
from app.services import audit as audit_service
from app.services import comparison_service

router = APIRouter(prefix="/ml", tags=["M07 Laboratorio de IA · M08 · M09"])


@router.get("/algorithms", response_model=List[AlgorithmInfo], summary="Algoritmos disponibles")
def algorithms(auth: AuthContext = Depends(require_permissions("ml:read"))):
    return algorithm_metadata()


@router.get("/backend", summary="Estado del backend de Deep Learning")
def backend(auth: AuthContext = Depends(require_permissions("ml:read"))):
    return {
        "torch": torch_info(),
        "xgboost": _xgboost_info(),
        "note": (
            "Los modelos neuronales requieren PyTorch. XGBoost opera de forma independiente."
            if not torch_info()["available"]
            else "PyTorch disponible en CPU: los cinco algoritmos son ejecutables."
        ),
    }


def _xgboost_info() -> Dict[str, Any]:
    try:
        import xgboost

        return {"available": True, "version": xgboost.__version__}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc)}


@router.post("/experiments", response_model=ExperimentOut, status_code=202, summary="Lanzar un experimento")
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:run")),
):
    metadata = {item["key"]: item for item in algorithm_metadata()}
    meta = metadata.get(payload.algorithm)
    if meta is None:
        raise ValidationError(f"Algoritmo no válido: {payload.algorithm}")
    if meta["status"] != "READY":
        raise ValidationError(
            f"{meta['name']} no está disponible en este entorno: {meta['status_reason']}",
            details={"algorithm": payload.algorithm},
        )
    if payload.n_trials > settings.ML_MAX_TRIALS:
        raise ValidationError(f"El máximo de trials permitido es {settings.ML_MAX_TRIALS}.")
    if not payload.horizons:
        raise ValidationError("Debe indicar al menos un horizonte de pronóstico.")

    run_id = f"exp-{uuid.uuid4().hex[:12]}"
    experiment = Experiment(
        run_id=run_id,
        name=payload.name or f"{meta['name']} · h={payload.horizons}",
        algorithm=payload.algorithm,
        status="QUEUED",
        stage="en_cola",
        n_trials=payload.n_trials,
        validation_strategy=payload.validation_strategy,
        horizons={"values": payload.horizons},
        hyperparameters={},
        search_space=get_search_space(payload.algorithm),
        feature_config={"max_lag": settings.ML_MAX_LAG},
        notes=payload.notes,
        created_by=auth.user.email,
        is_synthetic_data=True,
    )
    db.add(experiment)
    db.commit()

    options = {
        "horizons": payload.horizons,
        "n_trials": payload.n_trials,
        "strategy": payload.validation_strategy,
        "max_epochs": 40,
    }
    if not submit_experiment(experiment.id, options):
        raise ConflictError("Ya hay un experimento en ejecución con el mismo identificador.")

    audit_service.record(
        db, event_type=audit_service.AuditEventType.MODEL_EXECUTION, module="M07", action="start_experiment",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        resource_type="experiment", resource_id=experiment.id,
        description=f"Experimento {payload.algorithm} iniciado con {payload.n_trials} trials y horizontes {payload.horizons}.",
    )
    db.commit()
    return _experiment_payload(experiment)


def _experiment_payload(experiment: Experiment) -> Dict[str, Any]:
    return {
        "id": experiment.id,
        "run_id": experiment.run_id,
        "name": experiment.name,
        "algorithm": experiment.algorithm,
        "status": experiment.status,
        "stage": experiment.stage,
        "progress": experiment.progress,
        "n_trials": experiment.n_trials,
        "trials_completed": experiment.trials_completed,
        "validation_strategy": experiment.validation_strategy,
        "horizons": experiment.horizons,
        "hyperparameters": experiment.hyperparameters,
        "metrics": experiment.metrics,
        "best_trial": experiment.best_trial,
        "mlflow_run_id": experiment.mlflow_run_id,
        "is_synthetic_data": experiment.is_synthetic_data,
        "notes": experiment.notes,
        "error_message": experiment.error_message,
        "started_at": experiment.started_at,
        "finished_at": experiment.finished_at,
        "duration_seconds": experiment.duration_seconds,
        "created_by": experiment.created_by,
        "running": is_running(experiment.id),
    }


@router.get("/experiments", response_model=Page[ExperimentOut], summary="Listar experimentos")
def list_experiments(
    algorithm: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:read")),
):
    stmt = select(Experiment)
    if algorithm:
        stmt = stmt.where(Experiment.algorithm == algorithm)
    if status:
        stmt = stmt.where(Experiment.status == status)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(db.scalars(stmt.order_by(Experiment.started_at.desc()).offset((page - 1) * size).limit(size)))
    return Page[ExperimentOut](
        items=[_experiment_payload(row) for row in rows], total=total, page=page, size=size,
        pages=max(1, (total + size - 1) // size),
    )


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut, summary="Detalle de un experimento")
def get_experiment(experiment_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("ml:read"))):
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        raise NotFoundError("El experimento no existe.")
    return _experiment_payload(experiment)


@router.get("/models", response_model=List[ModelVersionOut], summary="Versiones de modelo")
def list_models(
    algorithm: Optional[str] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:read")),
):
    stmt = select(ModelVersion)
    if algorithm:
        stmt = stmt.where(ModelVersion.algorithm == algorithm)
    rows = list(db.scalars(stmt.order_by(ModelVersion.created_at.desc()).limit(200)))
    return [
        {
            "id": row.id,
            "algorithm": row.algorithm,
            "version": row.version,
            "status": row.status,
            "is_production": row.is_production,
            "training_weeks": row.training_weeks,
            "metrics": row.metrics,
            "hyperparameters": row.hyperparameters,
            "promoted_at": row.promoted_at,
            "promoted_by": row.promoted_by,
            "created_at": row.created_at,
            "artifact_checksum": row.artifact_checksum,
        }
        for row in rows
    ]


@router.post("/models/{version_id}/promote", response_model=Message, summary="Promover modelo a producción")
def promote(
    version_id: int,
    horizon: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:promote")),
):
    version = db.get(ModelVersion, version_id)
    if version is None:
        raise NotFoundError("La versión de modelo no existe.")
    target_horizon = horizon or int((version.feature_names or {}).get("horizon") or 1)
    previous = []
    # Solo puede haber un modelo en producción por horizonte: se archivan todos los demás,
    # sin importar el algoritmo, para que la inferencia sea determinista.
    for other in db.scalars(select(ModelVersion).where(ModelVersion.is_production.is_(True))):
        if other.id == version.id:
            continue
        if int((other.feature_names or {}).get("horizon") or -1) == target_horizon:
            other.is_production = False
            other.status = "ARCHIVED"
            previous.append(f"{other.algorithm}:{other.version}")
    version.is_production = True
    version.status = "PRODUCTION"
    from app.db.base import utcnow

    version.promoted_at = utcnow()
    version.promoted_by = auth.user.email
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M07", action="promote_model",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        resource_type="model_version", resource_id=version.id,
        description=f"Versión {version.version} ({version.algorithm}) promovida a producción para horizonte {target_horizon}.",
        context={"archived": previous},
    )
    db.commit()
    return Message(message=f"Modelo {version.algorithm} {version.version} promovido a producción (h={target_horizon}).")


@router.get("/metrics", response_model=List[MetricRow], summary="Matriz de métricas")
def metrics(
    horizon: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:read")),
):
    return comparison_service.metric_matrix(db, horizon=horizon)


@router.get("/compare", response_model=ModelComparison, summary="Comparación estadística de modelos")
def compare(
    horizon: int = Query(1, ge=1, le=12),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("ml:read")),
):
    return comparison_service.compare(db, horizon=horizon)


@router.get("/explain/{district_code}", response_model=ExplanationResponse, summary="Explicación de la predicción por distrito")
def explain(
    district_code: str,
    horizon: int = Query(1, ge=1, le=12),
    model_version_id: Optional[int] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("explain:read")),
):
    result = explain_district(db, district_code, horizon=horizon, model_version_id=model_version_id)
    return ExplanationResponse(**result)
