"""Dependencias de FastAPI: sesión de BD, usuario autenticado, RBAC y contexto de petición."""
from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import Depends, Header, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AuthError, PermissionError_
from app.core.security import decode_token
from app.db.session import SessionLocal
from app.models.security import User, UserSession

bearer_scheme = HTTPBearer(auto_error=False)


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@dataclass
class RequestContext:
    request_id: str = ""
    ip_address: str = ""
    user_agent: str = ""


async def get_context(
    request: Request,
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
) -> RequestContext:
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else ""
    return RequestContext(
        request_id=x_request_id or f"req-{uuid.uuid4().hex[:12]}",
        ip_address=client_ip,
        user_agent=(request.headers.get("User-Agent") or "")[:255],
    )


@dataclass
class AuthContext:
    user: User
    session: UserSession
    token_payload: dict = field(default_factory=dict)

    @property
    def permissions(self) -> List[str]:
        return self.user.permission_codes

    @property
    def session_id(self) -> str:
        return self.session.public_id


def _load_session(db: Session, session_public_id: str) -> Optional[UserSession]:
    return db.scalar(select(UserSession).where(UserSession.public_id == session_public_id))


async def get_auth_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> AuthContext:
    if credentials is None or not credentials.credentials:
        raise AuthError("Se requiere autenticación.")
    payload = decode_token(credentials.credentials, expected_type="access")
    session = _load_session(db, payload.get("sid", ""))
    if session is None:
        raise AuthError("La sesión no existe o fue revocada.")
    if session.revoked_at is not None:
        raise AuthError("La sesión fue revocada.")
    expires = session.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if expires < datetime.now(timezone.utc):
        raise AuthError("La sesión expiró.")
    user = db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise AuthError("El usuario está inactivo.")
    # Tocar la sesión para medir actividad (se confirma en el commit del request).
    session.last_seen_at = datetime.now(timezone.utc)
    return AuthContext(user=user, session=session, token_payload=payload)


def require_permissions(*required: str):
    """Dependencia que exige todos los permisos indicados."""

    async def _dependency(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        granted = set(ctx.permissions)
        if "*" in granted:
            return ctx
        missing = [code for code in required if code not in granted]
        if missing:
            raise PermissionError_(
                "No cuenta con permisos para esta operación.",
                details={"required": list(required), "missing": missing},
            )
        return ctx

    return _dependency


def require_any_permission(*required: str):
    async def _dependency(ctx: AuthContext = Depends(get_auth_context)) -> AuthContext:
        granted = set(ctx.permissions)
        if "*" in granted or any(code in granted for code in required):
            return ctx
        raise PermissionError_(
            "No cuenta con permisos para esta operación.",
            details={"required_any": list(required)},
        )

    return _dependency


CurrentAuth = Depends(get_auth_context)
