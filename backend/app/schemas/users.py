"""Esquemas de usuarios, roles y permisos."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class PermissionOut(BaseModel):
    code: str
    description: str
    module: str


class RoleOut(BaseModel):
    id: int
    slug: str
    name: str
    description: str
    is_system: bool
    permissions: List[PermissionOut] = Field(default_factory=list)
    user_count: int = 0


class RoleCreate(BaseModel):
    slug: str = Field(pattern=r"^[a-z0-9_]{3,64}$")
    name: str = Field(min_length=3, max_length=64)
    description: str = ""
    permissions: List[str] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    position: str
    organization: str
    is_active: bool
    is_superuser: bool
    must_change_password: bool
    last_login_at: Optional[datetime]
    created_at: datetime
    roles: List[RoleOut] = Field(default_factory=list)
    locked_until: Optional[datetime] = None


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=3, max_length=160)
    password: str = Field(min_length=10, max_length=72)
    position: str = ""
    organization: str = ""
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    position: Optional[str] = None
    organization: Optional[str] = None
    is_active: Optional[bool] = None
    roles: Optional[List[str]] = None


class UserStatusUpdate(BaseModel):
    is_active: bool
    reason: str = ""


class AuditEventOut(BaseModel):
    id: int
    occurred_at: datetime
    event_type: str
    module: str
    action: str
    result: str
    user_email: str
    user_role: str
    session_id: str
    resource_type: str
    resource_id: str
    description: str
    ip_address: str
    duration_ms: Optional[int]
    hash: str


class AuditVerifyResponse(BaseModel):
    valid: bool
    checked: int
    first_broken_id: Optional[int] = None
    message: str


class AuditStats(BaseModel):
    total_events: int
    by_type: List[dict]
    by_module: List[dict]
    by_result: List[dict]
    active_sessions: int
    last_event_at: Optional[datetime]
    range_start: Optional[datetime]
    range_end: Optional[datetime]
