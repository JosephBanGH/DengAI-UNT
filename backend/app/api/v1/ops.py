"""Routers de alertas, reportes, asistente y workflows (M11 a M14)."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, RequestContext, get_auth_context, get_context, get_db, require_permissions
from app.schemas.common import Message
from app.schemas.ops import (
    AlertGenerateRequest,
    AlertOut,
    AlertUpdate,
    AssistantAnswer,
    AssistantAskRequest,
    ReportCreate,
    ReportOut,
    ReportTypeInfo,
    WorkflowExecutionOut,
    WorkflowInfo,
)
from app.services import alerts_service, assistant_service, reports_service, workflows_service

alerts_router = APIRouter(prefix="/alerts", tags=["M11 Alertas tempranas"])
reports_router = APIRouter(prefix="/reports", tags=["M14 Reportes"])
assistant_router = APIRouter(prefix="/assistant", tags=["M13 Asistente inteligente"])
workflows_router = APIRouter(prefix="/workflows", tags=["M12 Orquestación"])


@alerts_router.get("", summary="Listar alertas")
def list_alerts(
    status: Optional[str] = None,
    level: Optional[str] = None,
    district: Optional[str] = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("alerts:read")),
):
    return alerts_service.list_alerts(db, status=status, level=level, district=district, page=page, size=size)


@alerts_router.get("/summary", summary="Resumen de alertas")
def alert_summary(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("alerts:read"))):
    return alerts_service.alert_summary(db)


@alerts_router.post("/generate", summary="Generar alertas desde pronósticos")
def generate(
    payload: AlertGenerateRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("alerts:write")),
):
    created = alerts_service.generate_alerts(
        db, horizons=payload.horizons, min_level=payload.min_level, actor=auth.user.email
    )
    return {"created": len(created), "items": created}


@alerts_router.post("/{alert_id}", response_model=AlertOut, summary="Actualizar alerta")
def update_alert(
    alert_id: int,
    payload: AlertUpdate,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("alerts:write")),
):
    return alerts_service.update_alert(db, alert_id, payload.action, payload.notes, actor=auth.user.email)


@reports_router.get("/types", response_model=List[ReportTypeInfo], summary="Catálogo de reportes")
def report_types(auth: AuthContext = Depends(require_permissions("reports:read"))):
    return reports_service.report_type_catalog()


@reports_router.get("", summary="Historial de reportes")
def list_reports(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("reports:read")),
):
    return reports_service.list_reports(db, page=page, size=size, category=category)


@reports_router.post("", response_model=ReportOut, status_code=201, summary="Generar reporte")
def generate_report(
    payload: ReportCreate,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("reports:generate")),
    ctx: RequestContext = Depends(get_context),
):
    return reports_service.generate_report(
        db,
        report_type=payload.report_type,
        fmt=payload.format,
        title=payload.title,
        filters=payload.filters,
        period_start=payload.period_start,
        period_end=payload.period_end,
        actor=auth.user.email,
        ip_address=ctx.ip_address,
        request_id=ctx.request_id,
    )


@reports_router.get("/{report_id}/download", summary="Descargar reporte")
def download_report(
    report_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("reports:read")),
):
    path, record = reports_service.report_file(db, report_id)
    media = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }.get(record.format, "application/octet-stream")
    from app.services import audit as audit_service

    audit_service.record(
        db, event_type=audit_service.AuditEventType.EXPORT, module="M14", action="download_report",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        resource_type="report", resource_id=record.id,
        description=f"Descarga del reporte {record.report_key} ({record.format}).",
    )
    db.commit()
    return FileResponse(path=str(path), media_type=media, filename=f"{record.report_key}.{record.format}")


@reports_router.delete("/{report_id}", response_model=Message, summary="Eliminar reporte")
def delete_report(
    report_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("reports:generate")),
):
    reports_service.delete_report(db, report_id, actor=auth.user.email)
    return Message(message="Reporte eliminado.")


@assistant_router.post("/ask", response_model=AssistantAnswer, summary="Consultar al asistente")
def ask(
    payload: AssistantAskRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("assistant:ask")),
):
    return assistant_service.ask(
        db, question=payload.question, district_code=payload.district_code, top_k=payload.top_k,
        user_email=auth.user.email,
    )


@assistant_router.get("/documents", summary="Documentos indexados")
def documents(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("assistant:ask"))):
    return assistant_service.documents_catalog(db)


@assistant_router.get("/history", summary="Historial de consultas")
def history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("assistant:ask")),
):
    return assistant_service.history(db, limit=limit)


@workflows_router.get("", response_model=List[WorkflowInfo], summary="Workflows disponibles")
def list_workflows(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("workflows:read"))):
    return workflows_service.catalog(db)


@workflows_router.get("/files", summary="Archivos de workflow para n8n")
def workflow_files(auth: AuthContext = Depends(require_permissions("workflows:read"))):
    return workflows_service.workflow_files()


@workflows_router.get("/executions", summary="Historial de ejecuciones")
def executions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("workflows:read")),
):
    return workflows_service.list_executions(db, page=page, size=size)


@workflows_router.post("/{code}/run", response_model=WorkflowExecutionOut, summary="Ejecutar workflow")
def run_workflow(
    code: str,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("workflows:run")),
):
    return workflows_service.run_workflow(db, code.upper(), actor=auth.user.email, trigger="manual")
