"""Hashing de contraseñas, emisión/validación de JWT y política de contraseñas."""
from __future__ import annotations

import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import bcrypt
import jwt

from app.core.config import settings
from app.core.errors import AuthError, ValidationError

ALGORITHM = settings.JWT_ALGORITHM
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

SPECIAL_CHARS = "!@#$%^&*()-_=+[]{};:,.?/|"


def hash_password(password: str) -> str:
    """bcrypt limita la entrada a 72 bytes; se rechaza antes para no truncar en silencio."""
    if len(password.encode("utf-8")) > 72:
        raise ValidationError("La contraseña supera el máximo de 72 bytes admitido por bcrypt.")
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def validate_password_policy(password: str) -> None:
    problems = []
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        problems.append(f"debe tener al menos {settings.PASSWORD_MIN_LENGTH} caracteres")
    if not re.search(r"[A-Z]", password):
        problems.append("debe incluir al menos una mayúscula")
    if not re.search(r"[a-z]", password):
        problems.append("debe incluir al menos una minúscula")
    if not re.search(r"\d", password):
        problems.append("debe incluir al menos un dígito")
    if not any(ch in SPECIAL_CHARS for ch in password):
        problems.append("debe incluir al menos un carácter especial")
    if problems:
        raise ValidationError("Contraseña insegura: " + "; ".join(problems))


def _encode(payload: Dict[str, Any], token_type: str, expires_delta: timedelta) -> Tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    expires_at = now + expires_delta
    body = {
        **payload,
        "type": token_type,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": secrets.token_urlsafe(16),
        "iss": settings.APP_SHORT_NAME,
    }
    return jwt.encode(body, settings.SECRET_KEY, algorithm=ALGORITHM), expires_at


def create_access_token(subject: str, *, session_id: str, roles: list[str], permissions: list[str]) -> Tuple[str, datetime]:
    return _encode(
        {"sub": subject, "sid": session_id, "roles": roles, "perms": permissions},
        TOKEN_TYPE_ACCESS,
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(subject: str, *, session_id: str) -> Tuple[str, datetime]:
    return _encode(
        {"sub": subject, "sid": session_id},
        TOKEN_TYPE_REFRESH,
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


def decode_token(token: str, *, expected_type: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM], issuer=settings.APP_SHORT_NAME)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("El token expiró. Inicie sesión nuevamente.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Token inválido.") from exc
    if payload.get("type") != expected_type:
        raise AuthError("Tipo de token incorrecto.")
    return payload


def new_opaque_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def constant_time_equals(left: Optional[str], right: Optional[str]) -> bool:
    if left is None or right is None:
        return False
    return secrets.compare_digest(left, right)
