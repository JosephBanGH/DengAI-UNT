"""Autenticación, sesiones y recuperación de contraseña (M01)."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AuthError, RateLimitError, ValidationError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    new_opaque_token,
    validate_password_policy,
    verify_password,
)
from app.models.security import LoginAttempt, SystemConfig, User, UserSession
from app.services import audit as audit_service

logger = logging.getLogger("dengai.auth")
RESET_TOKEN_KEY = "password_reset_tokens"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def register_failed_attempt(db: Session, user: Optional[User], email: str, reason: str, ip: str, user_agent: str) -> None:
    db.add(
        LoginAttempt(
            email=email, user_id=user.id if user else None, success=False,
            reason=reason[:120], ip_address=ip[:64], user_agent=user_agent[:255],
        )
    )
    if user is not None:
        user.failed_login_count = int(user.failed_login_count or 0) + 1
        if user.failed_login_count >= settings.MAX_LOGIN_ATTEMPTS:
            user.locked_until = _utcnow() + timedelta(minutes=settings.LOGIN_LOCK_MINUTES)
    db.flush()


def login(
    db: Session,
    *,
    email: str,
    password: str,
    ip_address: str = "",
    user_agent: str = "",
    request_id: str = "",
) -> Dict[str, object]:
    user = db.scalar(select(User).where(User.email == email.lower().strip()))
    if user is None:
        register_failed_attempt(db, None, email, "USER_NOT_FOUND", ip_address, user_agent)
        audit_service.record(
            db, event_type=audit_service.AuditEventType.ACCESS_DENIED, module="M01", action="login",
            result="DENIED", user_email=email, ip_address=ip_address, user_agent=user_agent,
            request_id=request_id, description="Intento de acceso con usuario inexistente.",
        )
        db.commit()
        raise AuthError("Credenciales inválidas.")

    if user.locked_until and _aware(user.locked_until) > _utcnow():
        db.commit()
        raise RateLimitError(
            "Cuenta bloqueada temporalmente por intentos fallidos.",
            details={"locked_until": _aware(user.locked_until).isoformat()},
        )
    if not user.is_active:
        register_failed_attempt(db, user, email, "USER_INACTIVE", ip_address, user_agent)
        db.commit()
        raise AuthError("El usuario está inactivo. Contacte al administrador.")

    if not verify_password(password, user.password_hash):
        register_failed_attempt(db, user, email, "BAD_PASSWORD", ip_address, user_agent)
        audit_service.record(
            db, event_type=audit_service.AuditEventType.ACCESS_DENIED, module="M01", action="login",
            result="DENIED", user_id=user.id, user_email=user.email, user_role=",".join(user.role_slugs),
            ip_address=ip_address, user_agent=user_agent, request_id=request_id,
            description="Contraseña incorrecta.",
        )
        db.commit()
        raise AuthError("Credenciales inválidas.")

    # Éxito: se limpian bloqueos y contadores.
    user.failed_login_count = 0
    user.locked_until = None
    user.last_login_at = _utcnow()
    db.add(LoginAttempt(email=user.email, user_id=user.id, success=True, reason="OK", ip_address=ip_address[:64], user_agent=user_agent[:255]))

    session_public = f"sess-{new_opaque_token(12)}"
    refresh_token, refresh_expires = create_refresh_token(user.email, session_id=session_public)
    session = UserSession(
        public_id=session_public,
        user_id=user.id,
        refresh_token_hash=_token_hash(refresh_token),
        expires_at=refresh_expires,
        ip_address=ip_address[:64],
        user_agent=user_agent[:255],
    )
    db.add(session)
    db.flush()

    access_token, access_expires = create_access_token(
        user.email, session_id=session_public, roles=user.role_slugs, permissions=user.permission_codes
    )
    audit_service.record(
        db, event_type=audit_service.AuditEventType.LOGIN, module="M01", action="login",
        result="SUCCESS", user_id=user.id, user_email=user.email,
        user_role=",".join(user.role_slugs), session_id=session_public,
        ip_address=ip_address, user_agent=user_agent, request_id=request_id,
        description="Inicio de sesión correcto.",
    )
    db.commit()
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": int((access_expires - _utcnow()).total_seconds()),
        "must_change_password": bool(user.must_change_password),
        "session_id": session_public,
    }


def refresh(db: Session, refresh_token: str, *, ip_address: str = "", user_agent: str = "") -> Dict[str, object]:
    payload = decode_token(refresh_token, expected_type="refresh")
    session = db.scalar(select(UserSession).where(UserSession.public_id == payload.get("sid", "")))
    if session is None or session.revoked_at is not None:
        raise AuthError("La sesión fue revocada o no existe.")
    if _aware(session.expires_at) < _utcnow():
        raise AuthError("La sesión expiró. Inicie sesión nuevamente.")
    if not session.is_active:
        raise AuthError("La sesión no está activa.")
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise AuthError("Usuario inactivo.")
    access_token, access_expires = create_access_token(
        user.email, session_id=session.public_id, roles=user.role_slugs, permissions=user.permission_codes
    )
    session.last_seen_at = _utcnow()
    db.commit()
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": int((access_expires - _utcnow()).total_seconds()),
    }


def logout(db: Session, session_public_id: str, *, user: Optional[User] = None, ip_address: str = "", user_agent: str = "", reason: str = "logout") -> None:
    session = db.scalar(select(UserSession).where(UserSession.public_id == session_public_id))
    if session is not None and session.revoked_at is None:
        session.revoked_at = _utcnow()
        session.revoke_reason = reason
        audit_service.record(
            db, event_type=audit_service.AuditEventType.LOGOUT, module="M01", action="logout",
            result="SUCCESS", user_id=session.user_id, user_email=user.email if user else "",
            user_role=",".join(user.role_slugs) if user else "", session_id=session.public_id,
            ip_address=ip_address, user_agent=user_agent, description=f"Cierre de sesión ({reason}).",
        )
    db.commit()


def change_password(db: Session, user: User, current_password: str, new_password: str, *, user_role: str = "") -> None:
    if not verify_password(current_password, user.password_hash):
        raise AuthError("La contraseña actual no coincide.")
    validate_password_policy(new_password)
    if verify_password(new_password, user.password_hash):
        raise ValidationError("La nueva contraseña debe ser distinta de la actual.")
    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    audit_service.record(
        db, event_type=audit_service.AuditEventType.PASSWORD_CHANGE, module="M01", action="change_password",
        result="SUCCESS", user_id=user.id, user_email=user.email, user_role=user_role,
        description="Cambio de contraseña exitoso.",
    )
    db.commit()


def request_password_reset(db: Session, email: str) -> Dict[str, object]:
    """Genera un token de un solo uso. Sin SMTP configurado se devuelve para entrega manual."""
    user = db.scalar(select(User).where(User.email == email.lower().strip()))
    if user is None:
        return {
            "message": "Si el correo está registrado, se generó un enlace de restablecimiento.",
            "delivery": "manual",
        }
    token = new_opaque_token(24)
    expires_at = _utcnow() + timedelta(minutes=30)
    record = db.scalar(select(SystemConfig).where(SystemConfig.key == RESET_TOKEN_KEY))
    payload = dict(record.value or {}) if record else {}
    payload[_token_hash(token)] = {"email": user.email, "expires_at": expires_at.isoformat(), "used": False}
    if record is None:
        db.add(SystemConfig(key=RESET_TOKEN_KEY, value=payload, description="Tokens de restablecimiento de contraseña.", updated_by="system"))
    else:
        record.value = payload
    audit_service.record(
        db, event_type=audit_service.AuditEventType.PASSWORD_CHANGE, module="M01", action="request_reset",
        result="SUCCESS", user_id=user.id, user_email=user.email,
        description="Solicitud de restablecimiento de contraseña.",
    )
    db.commit()
    return {
        "message": (
            "No hay servicio de correo configurado: entregue este token por un canal seguro "
            "al usuario verificado."
        ),
        "reset_token": token,
        "expires_at": expires_at,
        "delivery": "manual",
    }


def reset_password(db: Session, token: str, new_password: str) -> None:
    validate_password_policy(new_password)
    record = db.scalar(select(SystemConfig).where(SystemConfig.key == RESET_TOKEN_KEY))
    if record is None:
        raise ValidationError("El token no es válido.")
    payload = dict(record.value or {})
    entry = payload.get(_token_hash(token))
    if not entry or entry.get("used"):
        raise ValidationError("El token no es válido o ya fue utilizado.")
    if _aware(datetime.fromisoformat(entry["expires_at"])) < _utcnow():
        raise ValidationError("El token expiró. Solicite uno nuevo.")
    user = db.scalar(select(User).where(User.email == entry["email"]))
    if user is None:
        raise ValidationError("El token no es válido.")
    user.password_hash = hash_password(new_password)
    user.must_change_password = False
    user.failed_login_count = 0
    user.locked_until = None
    entry["used"] = True
    record.value = payload
    audit_service.record(
        db, event_type=audit_service.AuditEventType.PASSWORD_CHANGE, module="M01", action="reset_password",
        result="SUCCESS", user_id=user.id, user_email=user.email,
        description="Restablecimiento de contraseña mediante token.",
    )
    db.commit()


def list_sessions(db: Session, user: User, current_session_id: str) -> list:
    rows = []
    for session in db.scalars(select(UserSession).where(UserSession.user_id == user.id).order_by(UserSession.created_at.desc())):
        rows.append(
            {
                "id": session.id,
                "public_id": session.public_id,
                "created_at": session.created_at,
                "last_seen_at": session.last_seen_at,
                "expires_at": session.expires_at,
                "revoked_at": session.revoked_at,
                "ip_address": session.ip_address,
                "user_agent": session.user_agent,
                "current": session.public_id == current_session_id,
            }
        )
    return rows


def revoke_session(db: Session, user: User, public_id: str, *, reason: str = "revocada por el usuario") -> None:
    session = db.scalar(select(UserSession).where(UserSession.public_id == public_id, UserSession.user_id == user.id))
    if session is None:
        raise ValidationError("La sesión indicada no existe.")
    session.revoked_at = _utcnow()
    session.revoke_reason = reason
    audit_service.record(
        db, event_type=audit_service.AuditEventType.SESSION_REVOKED, module="M01", action="revoke_session",
        result="SUCCESS", user_id=user.id, user_email=user.email, session_id=public_id,
        description=f"Sesión revocada: {reason}.",
    )
    db.commit()
