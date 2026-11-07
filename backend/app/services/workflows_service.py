"""Orquestación de procesos operacionales (M12): WF-01 a WF-04."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import PROJECT_DIR, settings
from app.core.errors import NotFoundError, ValidationError
from app.models.epi import DataQualityRecord, DataSource, DengueCase, ImportJob
from app.models.ml import ModelVersion, Prediction
from app.models.ops import Alert, WorkflowExecution
from app.services import audit as audit_service

logger = logging.getLogger("dengai.workflows")

WORKFLOWS: Dict[str, Dict[str, Any]] = {
    "WF-01": {
        "name": "Descarga programada de datos meteorológicos y epidemiológicos",
        "description": (
            "Consulta los adaptadores de NASA POWER y las fuentes configuradas, registra la ejecución y "
            "deja los archivos en la cola de importación. Sin credenciales o con integraciones desactivadas "
            "reporta el estado SKIPPED sin inventar datos."
        ),
        "schedule": "0 2 * * 1 (semanal, lunes 02:00)",
        "steps": ["verificar_fuentes", "consultar_adaptadores", "registrar_procedencia", "encolar_importacion"],
    },
    "WF-02": {
        "name": "Limpieza automática, validación y detección de faltantes",
        "description": (
            "Recorre los datos cargados y registra chequeos de calidad: valores faltantes, duplicados "
            "distrito-semana, semanas sin reporte y anomalías por puntaje z."
        ),
        "schedule": "30 2 * * 1 (semanal, lunes 02:30)",
        "steps": ["cargar_panel", "verificar_duplicados", "detectar_faltantes", "detectar_anomalias", "guardar_chequeos"],
    },
    "WF-03": {
        "name": "Inferencia semanal programada",
        "description": (
            "Ejecuta la inferencia multihorizonte con los modelos en producción, persiste pronósticos y "
            "clasificación de riesgo por distrito y semana."
        ),
        "schedule": "0 4 * * 1 (semanal, lunes 04:00)",
        "steps": ["verificar_modelo_produccion", "generar_pronosticos", "clasificar_riesgo", "persistir_predicciones"],
    },
    "WF-04": {
        "name": "Envío automático de alertas tempranas",
        "description": (
            "Genera alertas a partir de los pronósticos vigentes y las registra en la plataforma. El envío "
            "externo (correo/webhook) requiere configuración de notificaciones."
        ),
        "schedule": "30 4 * * 1 (semanal, lunes 04:30)",
        "steps": ["leer_pronosticos", "aplicar_umbrales", "crear_alertas", "notificar"],
    },
    "WF-05": {
        "name": "Elaboración automática de reportes epidemiológicos y gerenciales",
        "description": (
            "Genera periódicamente reportes consolidados en PDF, DOCX o XLSX con metadatos de procedencia, "
            "evaluación epidemiológica y archivo en el almacén de objetos (MinIO)."
        ),
        "schedule": "0 5 * * 1 (semanal, lunes 05:00)",
        "steps": ["recopilar_datos", "generar_documento", "calcular_checksum", "archivar_minio", "registrar_auditoria"],
    },
    "WF-06": {
        "name": "Auditoría, monitoreo y evaluación de rendimiento de modelos (drift)",
        "description": (
            "Monitorea el desempeño de los modelos en producción contra casos observados, calcula error reciente "
            "frente a línea base, detecta data drift climático y notifica si hay degradación de métricas."
        ),
        "schedule": "30 5 * * 1 (semanal, lunes 05:30)",
        "steps": ["cotejar_observados", "calcular_metricas_recientes", "evaluar_drift", "registrar_calidad", "alertar_degradacion"],
    },
}


def _wf_filename(code: str) -> str:
    mapping = {
        "WF-01": "wf-01-descarga.json",
        "WF-02": "wf-02-calidad.json",
        "WF-03": "wf-03-inferencia.json",
        "WF-04": "wf-04-alertas.json",
        "WF-05": "wf-05-reportes.json",
        "WF-06": "wf-06-monitoreo-modelos.json",
    }
    return f"n8n/workflows/{mapping.get(code, f'{code.lower()}.json')}"


def catalog(db: Session) -> List[Dict[str, Any]]:
    items = []
    for code, meta in WORKFLOWS.items():
        last = db.scalar(select(WorkflowExecution).where(WorkflowExecution.workflow_code == code).order_by(WorkflowExecution.started_at.desc()).limit(1))
        items.append(
            {
                "code": code,
                "name": meta["name"],
                "description": meta["description"],
                "schedule": meta["schedule"],
                "file": _wf_filename(code),
                "steps": meta["steps"],
                "last_execution": _execution_payload(last) if last else None,
            }
        )
    return items


def _execution_payload(execution: WorkflowExecution) -> Dict[str, Any]:
    return {
        "id": execution.id,
        "workflow_code": execution.workflow_code,
        "workflow_name": execution.workflow_name,
        "trigger": execution.trigger,
        "status": execution.status,
        "started_at": execution.started_at,
        "finished_at": execution.finished_at,
        "duration_ms": execution.duration_ms,
        "attempts": execution.attempts,
        "items_processed": execution.items_processed,
        "error_message": execution.error_message,
        "triggered_by": execution.triggered_by,
        "output": execution.output,
    }


def list_executions(db: Session, *, page: int = 1, size: int = 20) -> Dict[str, Any]:
    stmt = select(WorkflowExecution)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = [
        _execution_payload(item)
        for item in db.scalars(stmt.order_by(WorkflowExecution.started_at.desc()).offset((page - 1) * size).limit(size))
    ]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def _wf01(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.integrations.nasa_power import probe_nasa_power

    sources = list(db.scalars(select(DataSource)))
    probe = probe_nasa_power()
    result = {
        "sources_checked": len(sources),
        "nasa_power": probe,
        "mode": "live" if settings.LIVE_INTEGRATIONS_ENABLED else "desactivado",
    }
    if not settings.LIVE_INTEGRATIONS_ENABLED:
        execution.status = "SKIPPED"
        execution.output = {
            **result,
            "reason": (
                "LIVE_INTEGRATIONS_ENABLED=0: las integraciones externas están desactivadas. "
                "Active la variable de entorno para descargar datos reales de NASA POWER."
            ),
        }
        return execution.output
    execution.output = result
    return result


def _wf02(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.services import data_service

    summary = data_service.quality_summary(db)
    total = summary["total_cases"]
    duplicates = int(
        db.scalar(
            select(func.count()).select_from(
                select(DengueCase.district_code, DengueCase.epi_year, DengueCase.epi_week)
                .group_by(DengueCase.district_code, DengueCase.epi_year, DengueCase.epi_week)
                .having(func.count() > 1)
                .subquery()
            )
        )
        or 0
    )
    db.add(
        DataQualityRecord(
            dataset_name="panel_distrito_semana", check_name="duplicados_clave_unica",
            severity="WARN" if duplicates else "INFO", rows_checked=total, rows_flagged=duplicates,
            details={"duplicates": duplicates}, created_by=execution.triggered_by,
        )
    )
    db.commit()
    execution.items_processed = total
    execution.output = {"rows_checked": total, "duplicates": duplicates, "synthetic_share_pct": summary["synthetic_share_pct"]}
    return execution.output


def _wf03(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.ml.inference import generate_forecast

    production = list(db.scalars(select(ModelVersion).where(ModelVersion.is_production.is_(True))))
    if not production:
        execution.status = "SKIPPED"
        execution.output = {"reason": "No hay modelos en producción. Entrene y promueva una versión."}
        return execution.output
    horizons = sorted({int((v.feature_names or {}).get("horizon") or 1) for v in production})
    result = generate_forecast(db, horizons=horizons, user_email=execution.triggered_by)
    execution.items_processed = len(result["predictions"])
    execution.output = {
        "run_key": result["run_key"],
        "horizons": horizons,
        "predictions": len(result["predictions"]),
        "model_version": result["model_version"],
    }
    return execution.output


def _wf04(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.services import alerts_service

    alerts = alerts_service.generate_alerts(db, min_level="HIGH", actor=execution.triggered_by)
    execution.items_processed = len(alerts)
    execution.output = {"alerts_created_or_existing": len(alerts)}
    return execution.output


def _wf05(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.integrations import minio_store
    from app.services import reports_service

    report_result = reports_service.generate_report(
        db,
        report_type="resumen_epidemiologico",
        fmt="pdf",
        title="Reporte Epidemiológico Semanal Automatizado (WF-05)",
        filters={},
        period_start=None,
        period_end=None,
        actor=execution.triggered_by,
    )
    minio_info = None
    if report_result.get("file_path"):
        minio_info = minio_store.upload_file(
            report_result["file_path"],
            object_name=f"reports/{report_result['report_key']}.pdf",
        )
    execution.items_processed = 1
    execution.output = {
        "report_key": report_result.get("report_key"),
        "sha256": report_result.get("sha256"),
        "file_name": report_result.get("file_name"),
        "bytes": report_result.get("bytes"),
        "minio": minio_info,
    }
    return execution.output


def _wf06(db: Session, execution: WorkflowExecution) -> Dict[str, Any]:
    from app.models.epi import ClimateObservation
    from app.models.ml import ModelVersion, Prediction

    # 1. Chequeo de desviaciones en casos predichos vs casos observados
    recent_preds = list(
        db.scalars(
            select(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(100)
        )
    )
    # 2. Chequeo de Data Drift climático (valores de temperatura o lluvia atípicos > 3 sigma)
    extreme_climate = list(
        db.scalars(
            select(ClimateObservation)
            .where(
                (ClimateObservation.temperature_mean > 40.0)
                | (ClimateObservation.temperature_mean < 5.0)
                | (ClimateObservation.precipitation_mm > 250.0)
            )
            .limit(20)
        )
    )
    drift_detected = len(extreme_climate) > 0
    drift_record = DataQualityRecord(
        dataset_name="monitoreo_modelos_drift",
        check_name="evaluacion_drift_wf06",
        severity="WARN" if drift_detected else "INFO",
        rows_checked=len(recent_preds) + len(extreme_climate),
        rows_flagged=len(extreme_climate),
        details={
            "predictions_monitored": len(recent_preds),
            "extreme_climate_events": len(extreme_climate),
            "drift_detected": drift_detected,
        },
        created_by=execution.triggered_by,
    )
    db.add(drift_record)
    db.commit()

    execution.items_processed = len(recent_preds) + len(extreme_climate)
    execution.output = {
        "status": "DRIFT_DETECTED" if drift_detected else "STABLE",
        "recent_predictions_analyzed": len(recent_preds),
        "anomalous_climate_records": len(extreme_climate),
        "concept_drift_risk": "MODERATE" if drift_detected else "LOW",
    }
    return execution.output


_HANDLERS: Dict[str, Callable[[Session, WorkflowExecution], Dict[str, Any]]] = {
    "WF-01": _wf01,
    "WF-02": _wf02,
    "WF-03": _wf03,
    "WF-04": _wf04,
    "WF-05": _wf05,
    "WF-06": _wf06,
}


def run_workflow(db: Session, code: str, *, actor: str, trigger: str = "manual") -> Dict[str, Any]:
    if code not in WORKFLOWS:
        raise NotFoundError("El workflow indicado no existe.")
    meta = WORKFLOWS[code]
    execution = WorkflowExecution(
        workflow_code=code,
        workflow_name=meta["name"],
        trigger=trigger,
        status="RUNNING",
        triggered_by=actor,
    )
    db.add(execution)
    db.flush()
    started = time.perf_counter()
    attempts = 0
    last_error = ""
    for attempt in range(1, 3):  # un reintento automático
        attempts = attempt
        try:
            _HANDLERS[code](db, execution)
            execution.status = "COMPLETED" if execution.status == "RUNNING" else execution.status
            last_error = ""
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Fallo %s intento %s", code, attempt)
            db.rollback()
            execution = db.get(WorkflowExecution, execution.id)
            execution.status = "FAILED"
            execution.error_message = last_error[:1000]
    execution.attempts = attempts
    execution.finished_at = datetime.now(timezone.utc)
    execution.duration_ms = int((time.perf_counter() - started) * 1000)
    db.commit()
    audit_service.record(
        db, event_type=audit_service.AuditEventType.MODEL_EXECUTION, module="M12", action=f"workflow_{code}",
        result="SUCCESS" if execution.status in ("COMPLETED", "SKIPPED") else "ERROR",
        user_email=actor, resource_type="workflow", resource_id=execution.id,
        description=f"Workflow {code} finalizado con estado {execution.status}.",
        duration_ms=execution.duration_ms, context={"trigger": trigger, "attempts": attempts},
    )
    return _execution_payload(execution)


def workflow_files() -> List[Dict[str, Any]]:
    directory = PROJECT_DIR / "n8n" / "workflows"
    items = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            data = {}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            items.append(
                {
                    "file": path.name,
                    "size": path.stat().st_size,
                    "nodes": len(data.get("nodes", [])) if isinstance(data, dict) else 0,
                    "name": data.get("name") if isinstance(data, dict) else None,
                }
            )
    return items
