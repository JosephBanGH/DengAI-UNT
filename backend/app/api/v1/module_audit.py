"""Eventos de navegación por módulo (M03): entrada, salida y duración de permanencia."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, RequestContext, get_auth_context, get_context, get_db
from app.core.errors import ValidationError
from app.models.security import AuditEvent
from app.schemas.common import Message
from app.services import audit as audit_service

router = APIRouter(prefix="/audit", tags=["M03 Auditoría"])

ALLOWED = {"MODULE_ENTER", "MODULE_EXIT"}


class ModuleEventRequest(BaseModel):
    event_type: str = Field(pattern="^(MODULE_ENTER|MODULE_EXIT)$")
    module: str = Field(min_length=1, max_length=40)
    action: str = ""


@router.post("/module-event", response_model=Message, summary="Registrar entrada o salida de un módulo")
def module_event(
    payload: ModuleEventRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
    ctx: RequestContext = Depends(get_context),
) -> Message:
    if payload.event_type not in ALLOWED:
        raise ValidationError("Tipo de evento no permitido.")

    duration_ms: Optional[int] = None
    if payload.event_type == "MODULE_EXIT":
        # Se busca la entrada más reciente sin salida registrada para esta sesión y módulo.
        enters = list(
            db.scalars(
                select(AuditEvent)
                .where(
                    AuditEvent.event_type == "MODULE_ENTER",
                    AuditEvent.session_id == auth.session_id,
                    AuditEvent.module == payload.module,
                )
                .order_by(AuditEvent.occurred_at.desc())
                .limit(20)
            )
        )
        exits = list(
            db.scalars(
                select(AuditEvent).where(
                    AuditEvent.event_type == "MODULE_EXIT",
                    AuditEvent.session_id == auth.session_id,
                    AuditEvent.module == payload.module,
                )
            )
        )
        used = {event.occurred_at for event in exits}
        for enter in enters:
            if enter.occurred_at not in used:
                entered = enter.occurred_at
                if entered.tzinfo is None:
                    entered = entered.replace(tzinfo=timezone.utc)
                duration_ms = max(0, int((datetime.now(timezone.utc) - entered).total_seconds() * 1000))
                break

    audit_service.record(
        db,
        event_type=payload.event_type,
        module=payload.module[:40],
        action=(payload.action or "navigate")[:80],
        result="SUCCESS",
        user_id=auth.user.id,
        user_email=auth.user.email,
        user_role=",".join(auth.user.role_slugs),
        session_id=auth.session_id,
        ip_address=ctx.ip_address,
        user_agent=ctx.user_agent,
        request_id=ctx.request_id,
        duration_ms=duration_ms,
        description=(
            f"Entrada al módulo {payload.module}."
            if payload.event_type == "MODULE_ENTER"
            else f"Salida del módulo {payload.module}"
            + (f" tras {round(duration_ms / 1000, 1)} s." if duration_ms is not None else ".")
        ),
    )
    db.commit()
    return Message(message="Evento registrado.")
