"""Bitácora de auditoría encadenada y consultable.

Cada evento incluye ``hash = sha256(prev_hash || payload)``, de modo que alterar o
eliminar una fila rompe la cadena y ``verify_chain`` lo detecta. La tabla no se expone
para escritura a través de la API: solo el servicio interno inserta registros.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.security import AuditEvent

logger = logging.getLogger("dengai.audit")
_CHAIN_LOCK = threading.Lock()
_GENESIS = "0" * 64


class AuditEventType:
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    MODULE_ENTER = "MODULE_ENTER"
    MODULE_EXIT = "MODULE_EXIT"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXPORT = "EXPORT"
    REPORT_GENERATED = "REPORT_GENERATED"
    DATA_IMPORT = "DATA_IMPORT"
    MODEL_EXECUTION = "MODEL_EXECUTION"
    ACCESS_DENIED = "ACCESS_DENIED"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    SESSION_REVOKED = "SESSION_REVOKED"


def _canonical_dt(value: Optional[datetime]) -> Optional[str]:
    """Normaliza la marca temporal para que el hash sobreviva al viaje de ida y vuelta a la BD.

    SQLite devuelve ``DateTime(timezone=True)`` como fecha ingenua, por lo que se fija
    siempre la representación UTC sin zona antes de calcular o verificar el hash.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.replace(microsecond=0).isoformat()


def _payload(event: AuditEvent) -> Dict[str, Any]:
    return {
        "occurred_at": _canonical_dt(event.occurred_at),
        "event_type": event.event_type,
        "module": event.module,
        "action": event.action,
        "result": event.result,
        "user_id": event.user_id,
        "user_email": event.user_email,
        "session_id": event.session_id,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "description": event.description,
        "context": event.context or {},
    }


def _digest(prev_hash: str, payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{prev_hash}|{canonical}".encode("utf-8")).hexdigest()


def last_hash(db: Session) -> str:
    value = db.scalar(select(AuditEvent.hash).order_by(AuditEvent.id.desc()).limit(1))
    return value or _GENESIS


def record(
    db: Session,
    *,
    event_type: str,
    module: str = "",
    action: str = "",
    result: str = "SUCCESS",
    user_id: Optional[int] = None,
    user_email: str = "",
    user_role: str = "",
    session_id: str = "",
    resource_type: str = "",
    resource_id: Any = "",
    description: str = "",
    ip_address: str = "",
    user_agent: str = "",
    request_id: str = "",
    duration_ms: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[AuditEvent]:
    """Registra un evento. Nunca interrumpe la operación de negocio si falla."""
    try:
        with _CHAIN_LOCK:
            event = AuditEvent(
                occurred_at=datetime.now(timezone.utc),
                event_type=event_type,
                module=module,
                action=action,
                result=result,
                user_id=user_id,
                user_email=user_email,
                user_role=user_role,
                session_id=session_id,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id is not None else "",
                description=description[:2000],
                ip_address=ip_address[:64],
                user_agent=user_agent[:255],
                request_id=request_id[:64],
                duration_ms=duration_ms,
                context=context or {},
            )
            event.prev_hash = last_hash(db)
            event.hash = _digest(event.prev_hash, _payload(event))
            db.add(event)
            db.flush()
            return event
    except Exception:  # pragma: no cover - la auditoría no debe tumbar la petición
        logger.exception("No se pudo registrar el evento de auditoría %s", event_type)
        return None


def verify_chain(db: Session, limit: Optional[int] = None) -> Tuple[bool, int, Optional[int]]:
    """Recalcula la cadena completa y devuelve (válida, verificados, primer id roto)."""
    stmt = select(AuditEvent).order_by(AuditEvent.id.asc())
    if limit:
        stmt = stmt.limit(limit)
    prev = _GENESIS
    checked = 0
    for event in db.scalars(stmt):
        expected = _digest(prev, _payload(event))
        if event.prev_hash != prev or event.hash != expected:
            return False, checked, event.id
        prev = event.hash
        checked += 1
    return True, checked, None


def query_events(
    db: Session,
    *,
    event_type: Optional[str] = None,
    module: Optional[str] = None,
    result: Optional[str] = None,
    user_email: Optional[str] = None,
    session_id: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    search: Optional[str] = None,
    page: int = 1,
    size: int = 50,
) -> Tuple[List[AuditEvent], int]:
    stmt = select(AuditEvent)
    if event_type:
        stmt = stmt.where(AuditEvent.event_type == event_type)
    if module:
        stmt = stmt.where(AuditEvent.module == module)
    if result:
        stmt = stmt.where(AuditEvent.result == result)
    if user_email:
        stmt = stmt.where(AuditEvent.user_email.ilike(f"%{user_email}%"))
    if session_id:
        stmt = stmt.where(AuditEvent.session_id == session_id)
    if start:
        stmt = stmt.where(AuditEvent.occurred_at >= start)
    if end:
        stmt = stmt.where(AuditEvent.occurred_at <= end)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(
            AuditEvent.description.ilike(like)
            | AuditEvent.action.ilike(like)
            | AuditEvent.resource_id.ilike(like)
        )
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(db.scalars(stmt.order_by(AuditEvent.id.desc()).offset((page - 1) * size).limit(size)))
    return rows, total


def export_csv(events: Iterable[AuditEvent]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id", "fecha_hora", "tipo_evento", "modulo", "accion", "resultado",
            "usuario", "rol", "sesion", "recurso", "recurso_id", "descripcion",
            "ip", "duracion_ms", "hash",
        ]
    )
    for event in events:
        writer.writerow(
            [
                event.id,
                event.occurred_at.isoformat() if event.occurred_at else "",
                event.event_type, event.module, event.action, event.result,
                event.user_email, event.user_role, event.session_id,
                event.resource_type, event.resource_id, event.description,
                event.ip_address, event.duration_ms or "", event.hash,
            ]
        )
    return buffer.getvalue().encode("utf-8-sig")


def stats(db: Session) -> Dict[str, Any]:
    def _group(column) -> List[dict]:
        rows = db.execute(
            select(column, func.count()).group_by(column).order_by(func.count().desc()).limit(20)
        ).all()
        return [{"key": key or "(vacío)", "count": count} for key, count in rows]

    total = db.scalar(select(func.count()).select_from(AuditEvent)) or 0
    last_at = db.scalar(select(func.max(AuditEvent.occurred_at)))
    first_at = db.scalar(select(func.min(AuditEvent.occurred_at)))
    return {
        "total_events": total,
        "by_type": _group(AuditEvent.event_type),
        "by_module": _group(AuditEvent.module),
        "by_result": _group(AuditEvent.result),
        "last_event_at": last_at,
        "range_start": first_at,
        "range_end": last_at,
    }
