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
    DatasetStatisticsResponse,
    DataQualityOut,
    DictionaryUpdateEntry,
    GroupValidationResponse,
    ImportJobOut,
    ImportPreview,
    ImportRowErrorOut,
    IntegrationProbeResult,
    PanelSummary,
    QualitySummary,
    RealIngestRequest,
    RealIngestResponse,
)
from app.services import audit as audit_service
from app.services import data_service, dataset_service, real_data_service, statistics_service, validation_service


router = APIRouter(tags=["M05 Datos · M06 Pipeline · M16 Reproducibilidad"])


@router.get("/data/sources", response_model=List[DataSourceOut], summary="Catálogo de fuentes de datos")
def list_sources(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:read"))):
    return data_service.list_sources(db)


@router.post("/data/sources/{code}/probe", response_model=IntegrationProbeResult, summary="Verificar disponibilidad de una fuente")
def probe_source(code: str, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:import"))):
    from app.integrations.minsa_dengue import probe_minsa_dengue
    from app.integrations.nasa_power import probe_nasa_power

    normalized = code.upper()
    if normalized == "NASA_POWER":
        result = probe_nasa_power()
        return IntegrationProbeResult(
            source_code="NASA_POWER",
            reachable=bool(result.get("reachable")),
            status="OK" if result.get("reachable") else "UNAVAILABLE",
            message=result.get("message", ""),
            latency_ms=result.get("latency_ms"),
            sample=result.get("sample"),
        )
    if normalized in {"MINSA_VIGILANCIA", "MINSA_DENGUE"}:
        result = probe_minsa_dengue()
        return IntegrationProbeResult(
            source_code="MINSA_VIGILANCIA",
            reachable=bool(result.get("reachable")),
            status="OK" if result.get("reachable") else "UNAVAILABLE",
            message=result.get("message", ""),
            latency_ms=result.get("latency_ms"),
            sample={
                "status": result.get("status"),
                "content_length": result.get("content_length"),
                "content_type": result.get("content_type"),
                "endpoint": result.get("endpoint"),
            },
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


@router.post(
    "/data/ingest/real",
    response_model=RealIngestResponse,
    summary="Ingerir datos reales MINSA/CDC Perú + NASA POWER",
)
def ingest_real(
    payload: RealIngestRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:import")),
    ctx: RequestContext = Depends(get_context),
):
    result = real_data_service.ingest_real_data(
        db,
        actor=auth.user.email,
        start_year=payload.start_year,
        end_year=payload.end_year,
        district_codes=payload.district_codes,
        include_cases=payload.include_cases,
        include_climate=payload.include_climate,
        force_download=payload.force_download,
    )
    audit_service.record(
        db,
        event_type=audit_service.AuditEventType.DATA_IMPORT,
        module="M05",
        action="ingest_real",
        result="SUCCESS" if result.get("effective_data_mode") == "real" else "PARTIAL",
        user_email=auth.user.email,
        resource_type="real_ingestion",
        description="Ingesta real MINSA/CDC Perú + NASA POWER.",
        ip_address=ctx.ip_address,
        request_id=ctx.request_id,
        context={
            "requested": result.get("requested"),
            "steps": {
                name: {key: value for key, value in step.items() if key in {"status", "inserted", "updated", "rows_zero_filled"}}
                for name, step in result.get("steps", {}).items()
            },
        },
    )
    db.commit()
    return result


@router.get("/data/provenance", summary="Procedencia auditable de los datos (real vs. sintético)")
def provenance(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("data:read"))):
    return real_data_service.provenance(db)


@router.get("/data/panel", response_model=PanelSummary, summary="Panel distrito-semana que usaría el modelado")
def panel_summary(
    mode: Optional[str] = Query(None, pattern="^(auto|synthetic|real)$"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return real_data_service.panel_summary(db, mode=mode)


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


@router.put(
    "/datasets/{dataset_id}/dictionary/{column}",
    response_model=DataDictionaryEntry,
    summary="Actualizar metadatos de una variable del diccionario",
)
def update_dictionary_entry(
    dataset_id: int,
    column: str,
    payload: DictionaryUpdateEntry,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("datasets:write")),
):
    return dataset_service.update_data_dictionary_entry(
        db,
        dataset_id=dataset_id,
        column=column,
        description=payload.description,
        unit=payload.unit,
        source=payload.source,
        actor=auth.user.email,
    )


@router.get(
    "/data/group-validation",
    response_model=GroupValidationResponse,
    summary="Auditoría de validación de grupos y prevención de fuga de información (Panel Activo)",
)
def group_validation_active_panel(
    mode: Optional[str] = Query(None, pattern="^(auto|synthetic|real)$"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return validation_service.audit_group_validation(db, mode=mode)


@router.get(
    "/datasets/{dataset_id}/group-validation",
    response_model=GroupValidationResponse,
    summary="Auditoría de validación de grupos y prevención de fuga de información (Conjunto Versionado)",
)
def group_validation_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("datasets:read")),
):
    return validation_service.audit_group_validation(db, dataset_id=dataset_id)


@router.get(
    "/data/statistics",
    response_model=DatasetStatisticsResponse,
    summary="Estadística descriptiva multivariable y explicabilidad en lenguaje natural (Panel Activo)",
)
def dataset_statistics_active_panel(
    mode: Optional[str] = Query(None, pattern="^(auto|synthetic|real)$"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("data:read")),
):
    return statistics_service.compute_dataset_statistics(db, mode=mode)


@router.get(
    "/datasets/{dataset_id}/statistics",
    response_model=DatasetStatisticsResponse,
    summary="Estadística descriptiva multivariable y explicabilidad en lenguaje natural (Conjunto Versionado)",
)
def dataset_statistics_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("datasets:read")),
):
    return statistics_service.compute_dataset_statistics(db, dataset_id=dataset_id)

