"""Generación de reportes operacionales, científicos y gerenciales (M14).

Formatos: PDF (reportlab), DOCX (python-docx) y XLSX (openpyxl). Cada reporte incluye
fecha de generación, filtros aplicados, periodo, fuentes, versión de modelo y notas
metodológicas, tal como exige el protocolo.
"""
from __future__ import annotations

import hashlib
import io
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.models.epi import DataQualityRecord, DataSource, DengueCase, ImportJob
from app.models.geo import GeographicUnit
from app.models.ml import Experiment, ModelVersion, Prediction
from app.models.ops import Alert, Report, WorkflowExecution
from app.models.security import AuditEvent, UserSession
from app.services import audit as audit_service

logger = logging.getLogger("dengai.reports")

REPORT_TYPES: Dict[str, Dict[str, Any]] = {
    "estado_importaciones": {"category": "operational", "title": "Estado de importaciones", "requires_period": False},
    "calidad_datos": {"category": "operational", "title": "Calidad de los datos", "requires_period": False},
    "ejecuciones_workflows": {"category": "operational", "title": "Ejecuciones de workflows", "requires_period": False},
    "entrenamiento_modelos": {"category": "operational", "title": "Entrenamiento de modelos", "requires_period": True},
    "resultados_validacion": {"category": "operational", "title": "Resultados de validación", "requires_period": True},
    "predicciones_generadas": {"category": "operational", "title": "Predicciones generadas", "requires_period": True},
    "alertas_emitidas": {"category": "operational", "title": "Alertas emitidas", "requires_period": True},
    "auditoria_sistema": {"category": "operational", "title": "Auditoría del sistema", "requires_period": True},
    "resumen_epidemiologico": {"category": "managerial", "title": "Resumen epidemiológico", "requires_period": True},
    "distribucion_territorial": {"category": "managerial", "title": "Distribución territorial del riesgo", "requires_period": True},
    "distritos_prioritarios": {"category": "managerial", "title": "Distritos prioritarios", "requires_period": True},
    "pronosticos_casos": {"category": "managerial", "title": "Pronósticos de casos", "requires_period": True},
    "tendencias_historicas": {"category": "managerial", "title": "Tendencias históricas", "requires_period": True},
    "evolucion_alertas": {"category": "managerial", "title": "Evolución de las alertas", "requires_period": True},
    "indicadores_gestion": {"category": "managerial", "title": "Indicadores clave de gestión", "requires_period": False},
    "comparacion_periodos": {"category": "managerial", "title": "Comparación de periodos", "requires_period": True},
}


def report_type_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "key": key,
            "category": meta["category"],
            "title": meta["title"],
            "description": _description(key),
            "formats": ["pdf", "docx", "xlsx"],
            "requires_period": meta["requires_period"],
        }
        for key, meta in REPORT_TYPES.items()
    ]


def _description(key: str) -> str:
    texts = {
        "estado_importaciones": "Detalle de importaciones, filas procesadas y errores.",
        "calidad_datos": "Chequeos de valores faltantes, duplicados y anomalías.",
        "ejecuciones_workflows": "Historial de WF-01 a WF-04 con estado y duración.",
        "entrenamiento_modelos": "Experimentos, hiperparámetros y versiones generadas.",
        "resultados_validacion": "Métricas de prueba, intervalos y pruebas estadísticas.",
        "predicciones_generadas": "Pronósticos por distrito y horizonte.",
        "alertas_emitidas": "Alertas generadas, niveles y estado de seguimiento.",
        "auditoria_sistema": "Eventos de bitácora con su cadena de integridad.",
        "resumen_epidemiologico": "Situación actual, tendencia y carga de enfermedad.",
        "distribucion_territorial": "Riesgo e incidencia por distrito y provincia.",
        "distritos_prioritarios": "Distritos ordenados por prioridad de intervención.",
        "pronosticos_casos": "Pronósticos multihorizonte con intervalos de credibilidad.",
        "tendencias_historicas": "Serie histórica semanal y comparación interanual.",
        "evolucion_alertas": "Evolución temporal de alertas por nivel.",
        "indicadores_gestion": "Indicadores operativos de la plataforma.",
        "comparacion_periodos": "Comparación de periodos seleccionados.",
    }
    return texts.get(key, "")


def _collect(db: Session, report_type: str, filters: Dict[str, Any], period: Optional[tuple]) -> Dict[str, Any]:
    """Construye las secciones del reporte a partir de datos reales de la base."""
    sections: List[Dict[str, Any]] = []
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    synthetic = bool(db.scalar(select(func.count()).select_from(DengueCase).where(DengueCase.is_synthetic.is_(True))) or 0)

    def add(title: str, kind: str, content: Any) -> None:
        sections.append({"title": title, "kind": kind, "content": content})

    if report_type == "estado_importaciones":
        jobs = list(db.scalars(select(ImportJob).order_by(ImportJob.started_at.desc()).limit(50)))
        add("Importaciones recientes", "table", {
            "columns": ["Clave", "Archivo", "Estado", "Total", "Insertados", "Rechazados", "Fecha"],
            "rows": [
                [j.job_key, j.file_name, j.status, j.rows_total, j.rows_inserted, j.rows_rejected, str(j.started_at.date())]
                for j in jobs
            ],
        })
    elif report_type == "calidad_datos":
        checks = list(db.scalars(select(DataQualityRecord).order_by(DataQualityRecord.created_at.desc()).limit(50)))
        add("Chequeos de calidad", "table", {
            "columns": ["Conjunto", "Chequeo", "Severidad", "Revisados", "Marcados", "Fecha"],
            "rows": [[c.dataset_name, c.check_name, c.severity, c.rows_checked, c.rows_flagged, str(c.created_at.date())] for c in checks],
        })
    elif report_type == "ejecuciones_workflows":
        runs = list(db.scalars(select(WorkflowExecution).order_by(WorkflowExecution.started_at.desc()).limit(50)))
        add("Ejecuciones", "table", {
            "columns": ["Workflow", "Disparador", "Estado", "Inicio", "Duración (ms)", "Ítems"],
            "rows": [[r.workflow_code, r.trigger, r.status, str(r.started_at.replace(microsecond=0)), r.duration_ms or "", r.items_processed] for r in runs],
        })
    elif report_type == "entrenamiento_modelos":
        experiments = list(db.scalars(select(Experiment).order_by(Experiment.started_at.desc()).limit(30)))
        add("Experimentos", "table", {
            "columns": ["Run", "Algoritmo", "Estado", "Trials", "Duración (s)", "Creado por"],
            "rows": [[e.run_id[:12], e.algorithm, e.status, e.n_trials, e.duration_seconds or "", e.created_by] for e in experiments],
        })
    elif report_type == "resultados_validacion":
        versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(30)))
        rows = []
        for v in versions:
            m = v.metrics or {}
            rows.append([v.algorithm, v.version, m.get("MAE"), m.get("RMSE"), m.get("MASE"), m.get("coverage"), m.get("AUC_PR")])
        add("Métricas de prueba", "table", {
            "columns": ["Algoritmo", "Versión", "MAE", "RMSE", "MASE", "Cobertura", "AUC-PR"],
            "rows": [[_fmt(x) for x in row] for row in rows],
        })
    elif report_type == "predicciones_generadas":
        latest = db.scalar(select(func.max(Prediction.generated_at)))
        rows = []
        if latest:
            for p in db.scalars(select(Prediction).where(Prediction.generated_at == latest).limit(200)):
                rows.append([
                    units[p.district_code].name if p.district_code in units else p.district_code,
                    f"{p.target_year}-W{p.target_week:02d}", p.horizon_weeks,
                    round(p.predicted_cases, 1), round(p.lower_bound or 0, 1), round(p.upper_bound or 0, 1), p.risk_level,
                ])
        add("Pronósticos", "table", {
            "columns": ["Distrito", "Semana", "Horizonte", "Casos", "IC inf.", "IC sup.", "Riesgo"],
            "rows": rows,
        })
    elif report_type == "alertas_emitidas":
        alerts = list(db.scalars(select(Alert).order_by(Alert.generated_at.desc()).limit(100)))
        add("Alertas", "table", {
            "columns": ["Distrito", "Semana", "Nivel", "Casos pron.", "Estado", "Generada"],
            "rows": [
                [
                    units[a.district_code].name if a.district_code in units else a.district_code,
                    f"{a.epi_year}-W{a.epi_week:02d}", a.risk_level,
                    round(a.predicted_cases or 0, 1), a.status, str(a.generated_at.date()),
                ]
                for a in alerts
            ],
        })
    elif report_type == "auditoria_sistema":
        events = list(db.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(200)))
        add("Eventos de auditoría", "table", {
            "columns": ["Fecha", "Tipo", "Módulo", "Usuario", "Resultado", "Descripción"],
            "rows": [
                [str(e.occurred_at.replace(microsecond=0)), e.event_type, e.module, e.user_email, e.result, e.description[:80]]
                for e in events
            ],
        })
    elif report_type in ("resumen_epidemiologico", "tendencias_historicas", "comparacion_periodos"):
        end = period[1] if period else db.scalar(select(func.max(DengueCase.week_start)))
        start = period[0] if period else (end - timedelta(weeks=26) if end else None)
        stmt = (
            select(DengueCase.week_start, func.sum(DengueCase.confirmed_cases))
            .where(DengueCase.week_start >= start, DengueCase.week_start <= end)
            .group_by(DengueCase.week_start)
            .order_by(DengueCase.week_start)
        )
        series = [(str(w), int(c or 0)) for w, c in db.execute(stmt).all()]
        total = sum(value for _, value in series)
        add("Indicadores del periodo", "kpi", [
            {"label": "Casos confirmados", "value": total},
            {"label": "Semanas analizadas", "value": len(series)},
            {"label": "Promedio semanal", "value": round(total / len(series), 1) if series else 0},
        ])
        add("Serie semanal", "table", {"columns": ["Semana", "Casos"], "rows": [[w, c] for w, c in series]})
    elif report_type in ("distribucion_territorial", "distritos_prioritarios"):
        stmt = (
            select(DengueCase.district_code, func.sum(DengueCase.confirmed_cases))
            .group_by(DengueCase.district_code)
            .order_by(func.sum(DengueCase.confirmed_cases).desc())
        )
        rows = []
        for code, total in db.execute(stmt).all()[:20]:
            unit = units.get(code)
            population = float(unit.population or 0) if unit else 0
            rate = (float(total) / population * 100_000) if population else None
            rows.append([
                unit.name if unit else code, unit.province if unit else "", int(total or 0),
                round(rate, 1) if rate else "",
            ])
        add("Distritos por carga de enfermedad", "table", {
            "columns": ["Distrito", "Provincia", "Casos acumulados", "Incidencia/100k"],
            "rows": rows,
        })
    elif report_type == "pronosticos_casos":
        latest = db.scalar(select(func.max(Prediction.generated_at)))
        rows = []
        if latest:
            for p in db.scalars(select(Prediction).where(Prediction.generated_at == latest).limit(200)):
                rows.append([
                    units[p.district_code].name if p.district_code in units else p.district_code,
                    f"{p.target_year}-W{p.target_week:02d}", p.horizon_weeks,
                    round(p.predicted_cases, 1), p.risk_level,
                ])
        add("Pronósticos vigentes", "table", {
            "columns": ["Distrito", "Semana objetivo", "Horizonte", "Casos", "Riesgo"],
            "rows": rows,
        })
    elif report_type == "evolucion_alertas":
        alerts = list(db.scalars(select(Alert).order_by(Alert.generated_at)))
        buckets: Dict[str, Dict[str, int]] = {}
        for alert in alerts:
            key = f"{alert.epi_year}-W{alert.epi_week:02d}"
            buckets.setdefault(key, {}).setdefault(alert.risk_level, 0)
            buckets[key][alert.risk_level] += 1
        rows = [[key, values.get("CRITICAL", 0), values.get("HIGH", 0), values.get("MODERATE", 0), values.get("LOW", 0)] for key, values in sorted(buckets.items())]
        add("Alertas por semana y nivel", "table", {
            "columns": ["Semana", "Crítico", "Alto", "Medio", "Bajo"],
            "rows": rows,
        })
    elif report_type == "indicadores_gestion":
        add("Indicadores operativos", "kpi", [
            {"label": "Usuarios activos", "value": int(db.scalar(select(func.count()).select_from(UserSession).where(UserSession.revoked_at.is_(None))) or 0)},
            {"label": "Experimentos registrados", "value": int(db.scalar(select(func.count()).select_from(Experiment)) or 0)},
            {"label": "Modelos en producción", "value": int(db.scalar(select(func.count()).select_from(ModelVersion).where(ModelVersion.is_production.is_(True))) or 0)},
            {"label": "Alertas abiertas", "value": int(db.scalar(select(func.count()).select_from(Alert).where(Alert.status == "OPEN")) or 0)},
            {"label": "Eventos de auditoría", "value": int(db.scalar(select(func.count()).select_from(AuditEvent)) or 0)},
            {"label": "Reportes generados", "value": int(db.scalar(select(func.count()).select_from(Report)) or 0)},
        ])
    else:
        add("Sin datos", "text", "El tipo de reporte solicitado no tiene secciones definidas.")

    sources = [
        {"code": s.code, "name": s.name, "license": s.license, "synthetic": s.is_synthetic}
        for s in db.scalars(select(DataSource))
    ]
    version = db.scalar(select(ModelVersion).where(ModelVersion.is_production.is_(True)).order_by(ModelVersion.created_at.desc()))
    return {
        "sections": sections,
        "sources": sources,
        "model_version": f"{version.algorithm} {version.version}" if version else "sin modelo en producción",
        "is_synthetic": synthetic,
    }


def _fmt(value: Any) -> Any:
    if value is None:
        return "-"
    if isinstance(value, float):
        return round(value, 3)
    return value


def _render_pdf(title: str, meta: Dict[str, Any], sections: List[Dict[str, Any]]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=colors.HexColor("#555555"))
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * cm)]
    for key, value in meta.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", small))
    story.append(Spacer(1, 0.4 * cm))

    for section in sections:
        story.append(Paragraph(section["title"], styles["Heading2"]))
        kind = section["kind"]
        content = section["content"]
        if kind == "text":
            story.append(Paragraph(str(content), body))
        elif kind == "kpi":
            data = [[item["label"], str(item["value"])] for item in content]
            table = Table(data, colWidths=[9 * cm, 5 * cm])
            table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
            story.append(table)
        elif kind == "table":
            columns = content.get("columns", [])
            rows = content.get("rows", [])[:80]
            data = [columns] + [[str(cell) for cell in row] for row in rows]
            if data:
                available = 17 * cm
                width = available / max(1, len(columns))
                table = Table(data, colWidths=[width] * len(columns), repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C5C")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.append(table)
        story.append(Spacer(1, 0.35 * cm))

    doc.build(story)
    return buffer.getvalue()


def _render_docx(title: str, meta: Dict[str, Any], sections: List[Dict[str, Any]]) -> bytes:
    from docx import Document
    from docx.shared import Pt

    document = Document()
    document.add_heading(title, level=0)
    for key, value in meta.items():
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"{key}: ")
        run.bold = True
        paragraph.add_run(str(value))
        paragraph.runs[0].font.size = Pt(9)
    for section in sections:
        document.add_heading(section["title"], level=2)
        content = section["content"]
        if section["kind"] == "text":
            document.add_paragraph(str(content))
        elif section["kind"] == "kpi":
            table = document.add_table(rows=0, cols=2)
            table.style = "Light Grid Accent 1"
            for item in content:
                cells = table.add_row().cells
                cells[0].text = str(item["label"])
                cells[1].text = str(item["value"])
        elif section["kind"] == "table":
            columns = content.get("columns", [])
            rows = content.get("rows", [])[:200]
            table = document.add_table(rows=1, cols=len(columns))
            table.style = "Light Grid Accent 1"
            for index, column in enumerate(columns):
                table.rows[0].cells[index].text = str(column)
            for row in rows:
                cells = table.add_row().cells
                for index, value in enumerate(row):
                    cells[index].text = str(value)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _render_xlsx(title: str, meta: Dict[str, Any], sections: List[Dict[str, Any]]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Resumen"
    summary["A1"] = title
    summary["A1"].font = Font(size=14, bold=True)
    row = 3
    for key, value in meta.items():
        summary.cell(row=row, column=1, value=key).font = Font(bold=True)
        summary.cell(row=row, column=2, value=str(value))
        row += 1
    header_fill = PatternFill("solid", fgColor="0F4C5C")
    for index, section in enumerate(sections, start=1):
        sheet = workbook.create_sheet(title=f"T{index}"[:31])
        sheet["A1"] = section["title"]
        sheet["A1"].font = Font(bold=True, size=12)
        content = section["content"]
        if section["kind"] == "table":
            columns = content.get("columns", [])
            for col_index, column in enumerate(columns, start=1):
                cell = sheet.cell(row=3, column=col_index, value=column)
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
            for row_index, data_row in enumerate(content.get("rows", [])[:5000], start=4):
                for col_index, value in enumerate(data_row, start=1):
                    sheet.cell(row=row_index, column=col_index, value=value if not isinstance(value, (list, dict)) else str(value))
        elif section["kind"] == "kpi":
            for row_index, item in enumerate(content, start=3):
                sheet.cell(row=row_index, column=1, value=item["label"])
                sheet.cell(row=row_index, column=2, value=item["value"])
        else:
            sheet["A3"] = str(content)
    for sheet in workbook.worksheets:
        for column in sheet.columns:
            length = max((len(str(cell.value)) for cell in column if cell.value is not None), default=10)
            sheet.column_dimensions[column[0].column_letter].width = min(60, max(12, length + 2))
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def generate_report(
    db: Session,
    *,
    report_type: str,
    fmt: str,
    title: Optional[str],
    filters: Dict[str, Any],
    period_start: Optional[date],
    period_end: Optional[date],
    actor: str,
    ip_address: str = "",
    request_id: str = "",
) -> Dict[str, Any]:
    if report_type not in REPORT_TYPES:
        raise ValidationError(f"Tipo de reporte no válido. Opciones: {', '.join(REPORT_TYPES)}")
    if fmt not in ("pdf", "docx", "xlsx"):
        raise ValidationError("Formato no soportado. Use pdf, docx o xlsx.")
    meta_type = REPORT_TYPES[report_type]
    report_title = title or meta_type["title"]
    report_key = f"rep-{uuid.uuid4().hex[:12]}"
    started = datetime.now(timezone.utc)
    record = Report(
        report_key=report_key,
        title=report_title,
        category=meta_type["category"],
        report_type=report_type,
        format=fmt,
        status="GENERATING",
        filters=filters,
        period_start=period_start,
        period_end=period_end,
        generated_by=actor,
        generated_at=started,
        access_roles={"roles": ["admin_sistema", "admin_epidemiologico", "epidemiologo", "auditor", "investigador", "analista_datos", "consulta"]},
    )
    db.add(record)
    db.flush()

    try:
        payload = _collect(db, report_type, filters, (period_start, period_end) if period_start and period_end else None)
        meta = {
            "Fecha de generación": started.strftime("%Y-%m-%d %H:%M UTC"),
            "Periodo analizado": f"{period_start} a {period_end}" if period_start and period_end else "según disponibilidad de datos",
            "Filtros aplicados": filters or "ninguno",
            "Fuentes de datos": ", ".join(source["name"] for source in payload["sources"][:6]),
            "Versión del modelo": payload["model_version"],
            "Datos sintéticos": "SÍ (demostración)" if payload["is_synthetic"] else "No",
            "Notas metodológicas": (
                "Reporte generado automáticamente por SP-01 Dengue Intelligence. Los indicadores provienen "
                "exclusivamente de datos almacenados en la plataforma. Las cifras son de apoyo a la decisión "
                "y no constituyen diagnóstico clínico."
            ),
        }
        if fmt == "pdf":
            content = _render_pdf(report_title, meta, payload["sections"])
        elif fmt == "docx":
            content = _render_docx(report_title, meta, payload["sections"])
        else:
            content = _render_xlsx(report_title, meta, payload["sections"])

        filename = f"{report_key}.{fmt}"
        path = settings.reports_dir / filename
        path.write_bytes(content)
        record.file_path = str(path)
        record.file_size = len(content)
        record.checksum = hashlib.sha256(content).hexdigest()
        record.status = "READY"
        record.model_version = payload["model_version"]
        record.data_sources = {"sources": payload["sources"]}
        record.methodology_notes = meta["Notas metodológicas"]
        record.is_synthetic_data = payload["is_synthetic"]
        record.duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
        db.commit()
        audit_service.record(
            db, event_type=audit_service.AuditEventType.REPORT_GENERATED, module="M14", action="generate_report",
            result="SUCCESS", user_email=actor, resource_type="report", resource_id=record.id,
            ip_address=ip_address, request_id=request_id,
            description=f"Reporte '{report_title}' generado en formato {fmt.upper()} ({record.file_size} bytes).",
            duration_ms=record.duration_ms, context={"report_type": report_type, "filters": filters},
        )
    except Exception as exc:
        record.status = "FAILED"
        record.error_message = f"{type(exc).__name__}: {exc}"[:1000]
        db.commit()
        logger.exception("Fallo al generar el reporte %s", report_key)
        raise ValidationError(f"No fue posible generar el reporte: {exc}") from exc

    return _payload(record)


def _payload(record: Report) -> Dict[str, Any]:
    return {
        "id": record.id,
        "report_key": record.report_key,
        "title": record.title,
        "category": record.category,
        "report_type": record.report_type,
        "format": record.format,
        "status": record.status,
        "file_size": record.file_size,
        "filters": record.filters,
        "period_start": record.period_start,
        "period_end": record.period_end,
        "model_version": record.model_version,
        "generated_by": record.generated_by,
        "generated_at": record.generated_at,
        "is_synthetic_data": record.is_synthetic_data,
        "methodology_notes": record.methodology_notes,
        "error_message": record.error_message,
    }


def list_reports(db: Session, *, page: int = 1, size: int = 20, category: Optional[str] = None) -> Dict[str, Any]:
    stmt = select(Report)
    if category:
        stmt = stmt.where(Report.category == category)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = [_payload(r) for r in db.scalars(stmt.order_by(Report.generated_at.desc()).offset((page - 1) * size).limit(size))]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def report_file(db: Session, report_id: int) -> tuple:
    record = db.get(Report, report_id)
    if record is None:
        raise NotFoundError("El reporte no existe.")
    if record.status != "READY" or not record.file_path:
        raise ValidationError("El reporte no está disponible para descarga.")
    path = Path(record.file_path)
    if not path.exists():
        raise NotFoundError("El archivo del reporte no se encuentra en el almacenamiento.")
    return path, record


def delete_report(db: Session, report_id: int, *, actor: str) -> None:
    record = db.get(Report, report_id)
    if record is None:
        raise NotFoundError("El reporte no existe.")
    if record.file_path:
        try:
            Path(record.file_path).unlink(missing_ok=True)
        except OSError:
            logger.warning("No se pudo eliminar el archivo %s", record.file_path)
    db.delete(record)
    audit_service.record(
        db, event_type=audit_service.AuditEventType.DELETE, module="M14", action="delete_report",
        result="SUCCESS", user_email=actor, resource_type="report", resource_id=report_id,
        description=f"Reporte eliminado: {record.title}.",
    )
    db.commit()
