"""Esquema inicial de SP-01 Dengue Intelligence.

Genera todas las tablas declaradas en los modelos ORM (M01 a M15). Es idempotente en
bases nuevas; para cambios posteriores cree una revisión con ``alembic revision``.
"""
from __future__ import annotations

from alembic import op

from app.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
