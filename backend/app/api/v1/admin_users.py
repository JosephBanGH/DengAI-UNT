"""Routers de usuarios, roles, permisos y auditoría (M02 y M03)."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.orm import Session

from app.core.deps import AuthContext, get_auth_context, get_db, require_permissions
from app.schemas.common import Page
from app.schemas.users import (
    AuditEventOut,
    AuditStats,
    AuditVerifyResponse,
    PermissionOut,
    RoleCreate,
    RoleOut,
    RoleUpdate,
    UserCreate,
    UserOut,
    UserStatusUpdate,
    UserUpdate,
)
from app.schemas.common import Message
from app.services import audit as audit_service
from app.services import user_service

users_router = APIRouter(prefix="/users", tags=["M02 Usuarios y roles"])
roles_router = APIRouter(prefix="/roles", tags=["M02 Usuarios y roles"])
audit_router = APIRouter(prefix="/audit", tags=["M03 Auditoría"])


@users_router.get("", response_model=Page[UserOut], summary="Listar usuarios")
def list_users(
    search: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    page: int = Query(1, ge=1),
    size: int = Query(25, ge=1, le=200),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("users:read")),
):
    items, total = user_service.list_users(db, search=search, role=role, is_active=is_active, page=page, size=size)
    return Page[UserOut](items=items, total=total, page=page, size=size, pages=max(1, (total + size - 1) // size))


@users_router.post("", response_model=UserOut, status_code=201, summary="Crear usuario")
def create_user(payload: UserCreate, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("users:write"))):
    return user_service.create_user(db, payload.model_dump(), actor=auth.user.email)


@users_router.get("/{user_id}", response_model=UserOut, summary="Consultar usuario")
def get_user(user_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("users:read"))):
    return user_service._user_payload(user_service.get_user(db, user_id))


@users_router.patch("/{user_id}", response_model=UserOut, summary="Actualizar usuario")
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("users:write"))):
    return user_service.update_user(db, user_id, payload.model_dump(exclude_none=True), actor=auth.user.email)


@users_router.post("/{user_id}/status", response_model=UserOut, summary="Activar o desactivar usuario")
def set_status(user_id: int, payload: UserStatusUpdate, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("users:write"))):
    return user_service.set_user_status(db, user_id, payload.is_active, payload.reason, actor=auth.user.email)


@users_router.post("/{user_id}/force-logout", response_model=Message, summary="Revocar todas las sesiones del usuario")
def force_logout(user_id: int, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("users:write"))):
    count = user_service.force_logout_user(db, user_id, actor=auth.user.email)
    return Message(message=f"Se revocaron {count} sesiones.")


@roles_router.get("", response_model=List[RoleOut], summary="Listar roles")
def list_roles(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("roles:read"))):
    return user_service.list_roles(db)


@roles_router.post("", response_model=RoleOut, status_code=201, summary="Crear rol")
def create_role(payload: RoleCreate, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("roles:write"))):
    return user_service.create_role(db, payload.model_dump(), actor=auth.user.email)


@roles_router.patch("/{role_id}", response_model=RoleOut, summary="Actualizar rol")
def update_role(role_id: int, payload: RoleUpdate, db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("roles:write"))):
    return user_service.update_role(db, role_id, payload.model_dump(exclude_none=True), actor=auth.user.email)


@roles_router.get("/permissions/catalog", response_model=List[PermissionOut], summary="Catálogo de permisos")
def permission_catalog(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("roles:read"))):
    return user_service.list_permissions(db)


@audit_router.get("", response_model=Page[AuditEventOut], summary="Consultar bitácora")
def list_events(
    event_type: Optional[str] = None,
    module: Optional[str] = None,
    result: Optional[str] = None,
    user_email: Optional[str] = None,
    session_id: Optional[str] = None,
    search: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("audit:read")),
):
    rows, total = audit_service.query_events(
        db, event_type=event_type, module=module, result=result, user_email=user_email,
        session_id=session_id, start=start, end=end, search=search, page=page, size=size,
    )
    items = [
        AuditEventOut(
            id=row.id, occurred_at=row.occurred_at, event_type=row.event_type, module=row.module,
            action=row.action, result=row.result, user_email=row.user_email, user_role=row.user_role,
            session_id=row.session_id, resource_type=row.resource_type, resource_id=row.resource_id,
            description=row.description, ip_address=row.ip_address, duration_ms=row.duration_ms, hash=row.hash,
        )
        for row in rows
    ]
    return Page[AuditEventOut](items=items, total=total, page=page, size=size, pages=max(1, (total + size - 1) // size))


@audit_router.get("/stats", response_model=AuditStats, summary="Estadísticas de la bitácora")
def audit_stats(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("audit:read"))):
    data = audit_service.stats(db)
    data["active_sessions"] = user_service.active_session_count(db)
    return AuditStats(**data)


@audit_router.get("/verify", response_model=AuditVerifyResponse, summary="Verificar integridad de la cadena")
def verify(db: Session = Depends(get_db), auth: AuthContext = Depends(require_permissions("audit:read"))):
    valid, checked, broken = audit_service.verify_chain(db)
    return AuditVerifyResponse(
        valid=valid,
        checked=checked,
        first_broken_id=broken,
        message=(
            f"Cadena intacta: {checked} eventos verificados."
            if valid
            else f"Se detectó una alteración en el evento {broken}."
        ),
    )


@audit_router.get("/export", summary="Exportar bitácora (CSV)")
def export_events(
    event_type: Optional[str] = None,
    module: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permissions("audit:export")),
):
    rows, _total = audit_service.query_events(
        db, event_type=event_type, module=module, start=start, end=end, page=1, size=5000
    )
    payload = audit_service.export_csv(rows)
    audit_service.record(
        db, event_type=audit_service.AuditEventType.EXPORT, module="M03", action="export_audit",
        result="SUCCESS", user_id=auth.user.id, user_email=auth.user.email, session_id=auth.session_id,
        description=f"Exportación de {len(rows)} eventos de auditoría.",
    )
    db.commit()
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="bitacora_auditoria.csv"'},
    )
