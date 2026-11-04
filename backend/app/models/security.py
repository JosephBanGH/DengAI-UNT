"""Modelos de seguridad, identidad y auditoría."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin, utcnow


class Permission(Base, IdMixin):
    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    module: Mapped[str] = mapped_column(String(32), nullable=False, index=True)


role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    Column("permission_id", ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True),
)

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("role_id", ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
)


class Role(Base, IdMixin, TimestampMixin):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    permissions: Mapped[List[Permission]] = relationship(
        secondary=role_permissions, lazy="selectin", backref="roles"
    )
    users: Mapped[List["User"]] = relationship(secondary=user_roles, back_populates="roles", viewonly=True)


class User(Base, IdMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(160), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    organization: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    locked_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    roles: Mapped[List[Role]] = relationship(secondary=user_roles, lazy="selectin", back_populates="users")
    sessions: Mapped[List["UserSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    @property
    def role_slugs(self) -> List[str]:
        return sorted({role.slug for role in self.roles})

    @property
    def permission_codes(self) -> List[str]:
        if self.is_superuser:
            return ["*"]
        codes = {perm.code for role in self.roles for perm in role.permissions}
        return sorted(codes)


class UserSession(Base, IdMixin):
    """Sesión activa. El identificador público es un UUID opaco, no el id numérico."""

    __tablename__ = "user_sessions"

    public_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    refresh_token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoke_reason: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    user: Mapped[User] = relationship(back_populates="sessions")

    @property
    def is_active(self) -> bool:
        if self.revoked_at is not None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=utcnow().tzinfo)
        return expires > utcnow()


class LoginAttempt(Base, IdMixin):
    __tablename__ = "login_attempts"

    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reason: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class AuditEvent(Base, IdMixin):
    """Bitácora encadenada: cada fila incluye el hash de la anterior (evidencia de manipulación)."""

    __tablename__ = "audit_events"

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    module: Mapped[str] = mapped_column(String(40), default="", nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    result: Mapped[str] = mapped_column(String(20), default="SUCCESS", nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    user_email: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    user_role: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    resource_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ip_address: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    user_agent: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    prev_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    __table_args__ = (
        Index("ix_audit_events_type_time", "event_type", "occurred_at"),
        Index("ix_audit_events_user_time", "user_id", "occurred_at"),
    )


class SystemConfig(Base, IdMixin, TimestampMixin):
    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    value: Mapped[dict] = mapped_column(JSON, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    updated_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    __table_args__ = (UniqueConstraint("key", name="uq_system_config_key"),)
