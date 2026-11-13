"""Esquemas de autenticación y seguridad."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

try:
    import email_validator as _ev  # noqa: F401
    from pydantic import EmailStr
except ImportError:  # pragma: no cover
    EmailStr = str  # type: ignore[misc,assignment]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    must_change_password: bool = False


class RefreshRequest(BaseModel):
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=10, max_length=72)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str
    reset_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    delivery: str = "manual"


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=10, max_length=72)


class RoleSummary(BaseModel):
    slug: str
    name: str


class MeResponse(BaseModel):
    id: int
    email: str
    full_name: str
    position: str
    organization: str
    roles: List[RoleSummary]
    permissions: List[str]
    is_superuser: bool
    must_change_password: bool
    last_login_at: Optional[datetime] = None


class SessionInfo(BaseModel):
    id: int
    public_id: str
    created_at: datetime
    last_seen_at: datetime
    expires_at: datetime
    revoked_at: Optional[datetime]
    ip_address: str
    user_agent: str
    current: bool = False
