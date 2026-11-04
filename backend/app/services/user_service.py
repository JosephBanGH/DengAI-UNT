"""Gestión de usuarios, roles y permisos (M02)."""
from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.core.permissions import ALL_PERMISSION_CODES
from app.core.security import hash_password, validate_password_policy
from app.models.security import Permission, Role, User, UserSession
from app.services import audit as audit_service


def _role_payload(db: Session, role: Role) -> Dict[str, object]:
    count = db.scalar(
        select(func.count()).select_from(User).where(User.roles.any(Role.id == role.id))
    ) or 0
    return {
        "id": role.id,
        "slug": role.slug,
        "name": role.name,
        "description": role.description,
        "is_system": role.is_system,
        "permissions": [
            {"code": p.code, "description": p.description, "module": p.module}
            for p in sorted(role.permissions, key=lambda item: item.code)
        ],
        "user_count": int(count),
    }


def _user_payload(user: User, *, include_roles: bool = True) -> Dict[str, object]:
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "position": user.position,
        "organization": user.organization,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "must_change_password": user.must_change_password,
        "last_login_at": user.last_login_at,
        "created_at": user.created_at,
        "locked_until": user.locked_until,
        "roles": [
            {"id": r.id, "slug": r.slug, "name": r.name, "description": r.description, "is_system": r.is_system, "permissions": [], "user_count": 0}
            for r in (user.roles if include_roles else [])
        ],
    }


def list_users(db: Session, *, search: Optional[str] = None, role: Optional[str] = None, is_active: Optional[bool] = None, page: int = 1, size: int = 25):
    stmt = select(User)
    if search:
        like = f"%{search}%"
        stmt = stmt.where(User.email.ilike(like) | User.full_name.ilike(like))
    if is_active is not None:
        stmt = stmt.where(User.is_active.is_(is_active))
    if role:
        stmt = stmt.where(User.roles.any(Role.slug == role))
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = list(db.scalars(stmt.order_by(User.created_at.desc()).offset((page - 1) * size).limit(size)))
    return [_user_payload(user) for user in rows], total


def get_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("El usuario no existe.")
    return user


def create_user(db: Session, payload: Dict[str, object], *, actor: str = "system") -> Dict[str, object]:
    email = str(payload["email"]).lower().strip()
    if db.scalar(select(User).where(User.email == email)):
        raise ConflictError("Ya existe un usuario con ese correo.")
    password = str(payload["password"])
    validate_password_policy(password)
    roles = _resolve_roles(db, payload.get("roles") or [])
    user = User(
        email=email,
        full_name=str(payload["full_name"]),
        password_hash=hash_password(password),
        position=str(payload.get("position") or ""),
        organization=str(payload.get("organization") or ""),
        is_active=bool(payload.get("is_active", True)),
        must_change_password=True,
        roles=roles,
    )
    db.add(user)
    db.flush()
    audit_service.record(
        db, event_type=audit_service.AuditEventType.CREATE, module="M02", action="create_user",
        result="SUCCESS", user_email=actor, resource_type="user", resource_id=user.id,
        description=f"Usuario creado: {user.email} con roles {[r.slug for r in roles]}.",
    )
    db.commit()
    return _user_payload(user)


def _resolve_roles(db: Session, slugs: List[str]) -> List[Role]:
    roles = []
    for slug in slugs:
        role = db.scalar(select(Role).where(Role.slug == slug))
        if role is None:
            raise ValidationError(f"El rol '{slug}' no existe.")
        roles.append(role)
    return roles


def update_user(db: Session, user_id: int, payload: Dict[str, object], *, actor: str = "system") -> Dict[str, object]:
    user = get_user(db, user_id)
    changes = {}
    for field in ("full_name", "position", "organization"):
        if payload.get(field) is not None:
            setattr(user, field, payload[field])
            changes[field] = payload[field]
    if payload.get("roles") is not None:
        user.roles = _resolve_roles(db, payload["roles"])
        changes["roles"] = payload["roles"]
    if payload.get("is_active") is not None:
        user.is_active = bool(payload["is_active"])
        changes["is_active"] = user.is_active
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M02", action="update_user",
        result="SUCCESS", user_email=actor, resource_type="user", resource_id=user.id,
        description=f"Usuario actualizado {user.email}.", context={"changes": changes},
    )
    db.commit()
    return _user_payload(user)


def set_user_status(db: Session, user_id: int, is_active: bool, reason: str, *, actor: str = "system") -> Dict[str, object]:
    user = get_user(db, user_id)
    user.is_active = is_active
    if not is_active:
        for session in list(user.sessions):
            if session.revoked_at is None:
                session.revoked_at = user.updated_at
                session.revoke_reason = f"usuario desactivado: {reason}"[:120]
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M02", action="user_status",
        result="SUCCESS", user_email=actor, resource_type="user", resource_id=user.id,
        description=f"Usuario {'activado' if is_active else 'desactivado'}: {user.email}. Motivo: {reason or 'no indicado'}.",
    )
    db.commit()
    return _user_payload(user)


def list_roles(db: Session) -> List[Dict[str, object]]:
    return [_role_payload(db, role) for role in db.scalars(select(Role).order_by(Role.id))]


def list_permissions(db: Session) -> List[Dict[str, object]]:
    rows = list(db.scalars(select(Permission).order_by(Permission.module, Permission.code)))
    return [{"code": p.code, "description": p.description, "module": p.module} for p in rows] or [
        {"code": code, "description": "", "module": ""} for code in ALL_PERMISSION_CODES
    ]


def create_role(db: Session, payload: Dict[str, object], *, actor: str = "system") -> Dict[str, object]:
    slug = str(payload["slug"])
    if db.scalar(select(Role).where(Role.slug == slug)):
        raise ConflictError("Ya existe un rol con ese identificador.")
    permissions = _resolve_permissions(db, payload.get("permissions") or [])
    role = Role(slug=slug, name=str(payload["name"]), description=str(payload.get("description") or ""), is_system=False)
    role.permissions = permissions
    db.add(role)
    db.flush()
    audit_service.record(
        db, event_type=audit_service.AuditEventType.CREATE, module="M02", action="create_role",
        result="SUCCESS", user_email=actor, resource_type="role", resource_id=role.id,
        description=f"Rol creado: {role.slug} con {len(permissions)} permisos.",
    )
    db.commit()
    return _role_payload(db, role)


def _resolve_permissions(db: Session, codes: List[str]) -> List[Permission]:
    resolved = []
    for code in codes:
        permission = db.scalar(select(Permission).where(Permission.code == code))
        if permission is None:
            raise ValidationError(f"El permiso '{code}' no existe.")
        resolved.append(permission)
    return resolved


def update_role(db: Session, role_id: int, payload: Dict[str, object], *, actor: str = "system") -> Dict[str, object]:
    role = db.get(Role, role_id)
    if role is None:
        raise NotFoundError("El rol no existe.")
    if role.slug == "admin_sistema" and payload.get("permissions") is not None:
        raise ValidationError("El rol de administrador del sistema no puede perder permisos.")
    if payload.get("name"):
        role.name = str(payload["name"])
    if payload.get("description") is not None:
        role.description = str(payload["description"])
    if payload.get("permissions") is not None:
        role.permissions = _resolve_permissions(db, payload["permissions"])
    audit_service.record(
        db, event_type=audit_service.AuditEventType.UPDATE, module="M02", action="update_role",
        result="SUCCESS", user_email=actor, resource_type="role", resource_id=role.id,
        description=f"Rol actualizado: {role.slug}.",
    )
    db.commit()
    return _role_payload(db, role)


def active_session_count(db: Session) -> int:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return int(
        db.scalar(
            select(func.count())
            .select_from(UserSession)
            .where(UserSession.revoked_at.is_(None), UserSession.expires_at > now)
        )
        or 0
    )


def force_logout_user(db: Session, user_id: int, *, actor: str = "system") -> int:
    from datetime import datetime, timezone

    user = get_user(db, user_id)
    now = datetime.now(timezone.utc)
    revoked = 0
    for session in user.sessions:
        if session.revoked_at is None:
            session.revoked_at = now
            session.revoke_reason = "cierre forzado por administrador"
            revoked += 1
    audit_service.record(
        db, event_type=audit_service.AuditEventType.SESSION_REVOKED, module="M02", action="force_logout",
        result="SUCCESS", user_email=actor, resource_type="user", resource_id=user.id,
        description=f"Se revocaron {revoked} sesiones del usuario {user.email}.",
    )
    db.commit()
    return revoked
