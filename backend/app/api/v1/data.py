"""Routers de datos, fuentes, importaciones, calidad y conjuntos de datos (M05, M06, M16)."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, get_auth_context, get_context, get_db, require_permissions, RequestContext
from app.core.errors import ValidationError
from app.ml.features import build_features, load_panel
from app.models.epi import Dataset
from app.schemas.common import Message, Page
from app.schemas.data import (
    CaseOut,
    ClimateOut,
    DataDictionaryEntry,
    DataSourceOut,
    DatasetBuildRequest,
    DatasetOut,
    DataQualityOut,
    ImportJobOut,
    ImportPreview,
    ImportRowErrorOut,
    IntegrationProbeResult,
    QualitySummary,
)
from app.services import audit as audit_service
from app.services import data_service, dataset_service

router = APIRouter(tags=["M05 Datos · M06 Pipeline · M16 Reproducibilidad"])


@router.get("/data/sources", response_model=List[DataSourceOut], summary="Catálogo de fuentes de datos")
def list_sources(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:read"))):
    return data_service.list_sources(db)


@router.post("/data/sources/{code}/probe", response_model=IntegrationProbeResult, summary="Verificar disponibilidad de una fuente")
def probe_source(code: str, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:import"))):
    from app.integrations.nasa_power import probe_nasa_power

    if code.upper() == "NASA_POWER":
        result = probe_nasa_power()
        return IntegrationProbeResult(
            source_code="NASA_POWER",
            reachable=bool(result.get("reachable")),
            status="OK" if result.get("reachable") else "UNAVAILABLE",
            message=result.get("message", ""),
            latency_ms=result.get("latency_ms"),
            sample=result.get("sample"),
        )
    return IntegrationProbeResult(
        source_code=code.upper(),
        reachable=False,
        status="MANUAL",
        message=(
            "Esta fuente se integra mediante carga de archivos estructurados (CSV/JSON). "
            "No expone un endpoint público verificado por la plataforma; use la importación manual."
        ),
    )


@router.post("/data/import/preview", response_model=ImportPreview, summary="Previsualizar archivo antes de importar")
async def preview_import(
    file: UploadFile = File(...),
    kind: str = Form("epidemiological"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:import")),
):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise ValidationError("El archivo supera el límite de 20 MB.")
    return data_service.preview(raw, file.filename or "dataset.csv", kind)


@router.post("/data/import/cases", response_model=ImportJobOut, summary="Importar casos de dengue")
async def import_cases(
    file: UploadFile = File(...),
    source_code: Optional[str] = Form(None),
    is_synthetic: bool = Form(False),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:import")),
    ctx: RequestContext = Depends(get_context),
):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise ValidationError("El archivo supera el límite de 20 MB.")
    return data_service.import_cases(
        db, raw=raw, filename=file.filename or "casos.csv", source_code=source_code,
        actor=auth.user.email, is_synthetic=is_synthetic, ip_address=ctx.ip_address, request_id=ctx.request_id,
    )


@router.post("/data/import/climate", response_model=ImportJobOut, summary="Importar variables climáticas")
async def import_climate(
    file: UploadFile = File(...),
    source_code: Optional[str] = Form(None),
    is_synthetic: bool = Form(False),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:import")),
    ctx: RequestContext = Depends(get_context),
):
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise ValidationError("El archivo supera el límite de 20 MB.")
    return data_service.import_climate(
        db, raw=raw, filename=file.filename or "clima.csv", source_code=source_code,
        actor=auth.user.email, is_synthetic=is_synthetic, ip_address=ctx.ip_address, request_id=ctx.request_id,
    )


@router.get("/data/imports", summary="Historial de importaciones")
def list_imports(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    kind: Optional[str] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return data_service.list_imports(db, page=page, size=size, kind=kind)


@router.get("/data/imports/{job_id}/errors", response_model=List[ImportRowErrorOut], summary="Errores de una importación")
def import_errors(job_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:read"))):
    return data_service.import_errors(db, job_id)


@router.get("/data/quality", response_model=QualitySummary, summary="Resumen de calidad de datos")
def quality(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:read"))):
    data = data_service.quality_summary(db)
    return QualitySummary(**data)


@router.get("/data/cases", summary="Consultar casos")
def cases(
    district: Optional[str] = None,
    start_week: Optional[date] = None,
    end_week: Optional[date] = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return data_service.list_cases(db, district=district, start_week=start_week, end_week=end_week, page=page, size=size)


@router.get("/data/climate", summary="Consultar variables climáticas")
def climate(
    district: Optional[str] = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return data_service.list_climate(db, district=district, page=page, size=size)


@router.post("/datasets/build", response_model=DatasetOut, summary="Construir y versionar un conjunto de datos")
def build_dataset(
    payload: DatasetBuildRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("datasets:write")),
):
    return dataset_service.build_and_register(
        db, name=payload.name, description=payload.description, max_lag=payload.max_lag, actor=auth.user.email
    )


@router.get("/datasets", response_model=List[DatasetOut], summary="Listar conjuntos de datos")
def list_datasets(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("datasets:read"))):
    return dataset_service.list_datasets(db)


@router.get("/datasets/{dataset_id}/dictionary", response_model=List[DataDictionaryEntry], summary="Diccionario de datos")
def dataset_dictionary(dataset_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("datasets:read"))):
    return dataset_service.data_dictionary(db, dataset_id)


@router.get("/datasets/{dataset_id}", response_model=DatasetOut, summary="Detalle de un conjunto de datos")
def get_dataset(dataset_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("datasets:read"))):
    return dataset_service.get_dataset(db, dataset_id)
