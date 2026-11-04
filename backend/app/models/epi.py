"""Modelos de vigilancia epidemiológica, clima, importación y calidad de datos."""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin, utcnow


class DataSource(Base, IdMixin, TimestampMixin):
    __tablename__ = "data_sources"

    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    kind: Mapped[str] = mapped_column(String(30), default="epidemiological", nullable=False)
    provider: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    url: Mapped[str] = mapped_column(String(400), default="", nullable=False)
    license: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    license_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    access_method: Mapped[str] = mapped_column(String(30), default="file", nullable=False)
    integration_status: Mapped[str] = mapped_column(String(30), default="NOT_CONFIGURED", nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    refresh_frequency: Mapped[str] = mapped_column(String(30), default="weekly", nullable=False)
    last_success_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class ImportJob(Base, IdMixin, TimestampMixin):
    __tablename__ = "import_jobs"

    job_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    dataset_name: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    kind: Mapped[str] = mapped_column(String(30), default="epidemiological", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    rows_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_rejected: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_duplicated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    initiated_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    log: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    errors: Mapped[List["ImportRowError"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class ImportRowError(Base, IdMixin):
    __tablename__ = "import_row_errors"

    job_id: Mapped[int] = mapped_column(ForeignKey("import_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    column: Mapped[str] = mapped_column(String(80), default="", nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    raw_value: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    job: Mapped[ImportJob] = relationship(back_populates="errors")


class DengueCase(Base, IdMixin, TimestampMixin):
    """Casos confirmados por unidad geográfica y semana epidemiológica (unidad de análisis)."""

    __tablename__ = "dengue_cases"

    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    epi_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    epi_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week_start: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    confirmed_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    probable_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    hospitalized: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deaths: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="", nullable=False)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    import_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("import_jobs.id", ondelete="SET NULL"), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    data_quality_flag: Mapped[str] = mapped_column(String(20), default="OK", nullable=False)
    reported_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    district: Mapped["GeographicUnit"] = relationship(back_populates="cases")

    __table_args__ = (
        UniqueConstraint("district_code", "epi_year", "epi_week", name="uq_dengue_cases_district_week"),
        Index("ix_dengue_cases_week", "week_start"),
    )


class ClimateObservation(Base, IdMixin, TimestampMixin):
    """Variables climáticas semanales (NASA POWER / CHIRPS o carga estructurada)."""

    __tablename__ = "climate_observations"

    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    epi_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    epi_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week_start: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    temperature_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_max: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_min: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipitation_mm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solar_radiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    provider: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    source_id: Mapped[Optional[int]] = mapped_column(ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    import_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("import_jobs.id", ondelete="SET NULL"), nullable=True)
    data_quality_flag: Mapped[str] = mapped_column(String(20), default="OK", nullable=False)

    district: Mapped["GeographicUnit"] = relationship(back_populates="climate")

    __table_args__ = (
        UniqueConstraint("district_code", "epi_year", "epi_week", name="uq_climate_district_week"),
        Index("ix_climate_week", "week_start"),
    )


class EnvironmentalVariable(Base, IdMixin, TimestampMixin):
    """Índices ambientales/poblacionales adicionales (vegetación, densidad, movilidad)."""

    __tablename__ = "environmental_variables"

    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    epi_year: Mapped[int] = mapped_column(Integer, nullable=False)
    epi_week: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(60), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(30), default="", nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        UniqueConstraint("district_code", "epi_year", "epi_week", "name", name="uq_env_var_district_week_name"),
    )


class DataQualityRecord(Base, IdMixin, TimestampMixin):
    __tablename__ = "data_quality_records"

    import_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("import_jobs.id", ondelete="SET NULL"), nullable=True)
    dataset_name: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    check_name: Mapped[str] = mapped_column(String(80), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="INFO", nullable=False)
    rows_checked: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_flagged: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), default="system", nullable=False)


class Dataset(Base, IdMixin, TimestampMixin):
    """Conjunto de datos versionado listo para modelamiento."""

    __tablename__ = "datasets"

    slug: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    version: Mapped[str] = mapped_column(String(20), default="1.0.0", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    rows: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    districts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    weeks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    start_week: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    end_week: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    source_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    data_dictionary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="READY", nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), default="system", nullable=False)
