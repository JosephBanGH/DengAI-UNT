"""Alertas tempranas, reportes generados y ejecuciones de automatización."""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin, utcnow


class Alert(Base, IdMixin, TimestampMixin):
    __tablename__ = "alerts"

    alert_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    epi_year: Mapped[int] = mapped_column(Integer, nullable=False)
    epi_week: Mapped[int] = mapped_column(Integer, nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_cases: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lower_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    incidence_per_100k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    factors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="OPEN", nullable=False, index=True)
    acknowledged_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    prediction_run_key: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notification_channels: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    __table_args__ = (Index("ix_alerts_district_week", "district_code", "epi_year", "epi_week"),)


class Report(Base, IdMixin, TimestampMixin):
    __tablename__ = "reports"

    report_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    report_type: Mapped[str] = mapped_column(String(60), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(400), default="", nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    checksum: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    filters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    period_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    period_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    model_version: Mapped[str] = mapped_column(String(60), default="", nullable=False)
    data_sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    methodology_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    generated_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    access_roles: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)


class WorkflowExecution(Base, IdMixin, TimestampMixin):
    """Ejecuciones de los workflows WF-01..WF-04 (n8n o planificador interno)."""

    __tablename__ = "workflow_executions"

    workflow_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    workflow_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    trigger: Mapped[str] = mapped_column(String(30), default="manual", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="RUNNING", nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    items_processed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    triggered_by: Mapped[str] = mapped_column(String(255), default="system", nullable=False)


class AssistantQuery(Base, IdMixin, TimestampMixin):
    """Consultas al asistente RAG, con las fuentes efectivamente recuperadas."""

    __tablename__ = "assistant_queries"

    user_email: Mapped[str] = mapped_column(String(255), default="", nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, default="", nullable=False)
    sources: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    grounded: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(40), default="grounded_template", nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    blocked_reason: Mapped[str] = mapped_column(String(160), default="", nullable=False)
