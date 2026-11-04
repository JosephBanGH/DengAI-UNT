"""Modelos geoespaciales y de calendario epidemiológico.

La geometría se persiste como GeoJSON en ``geometry_geojson``. Cuando la base de
datos es PostgreSQL con PostGIS habilitado, la migración ``0002_postgis`` añade la
columna nativa ``geom geometry(MultiPolygon, 4326)``, su índice GiST y el trigger de
sincronización; la aplicación detecta el soporte y sirve las consultas espaciales
nativas (ver ``app/geospatial/postgis.py``).
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, IdMixin, TimestampMixin


class GeographicUnit(Base, IdMixin, TimestampMixin):
    __tablename__ = "geographic_units"

    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    level: Mapped[str] = mapped_column(String(20), default="district", nullable=False)
    parent_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    province: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    department: Mapped[str] = mapped_column(String(120), default="", nullable=False)
    altitude_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    area_km2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    population: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    population_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    centroid_lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    centroid_lon: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    geometry_geojson: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    cases: Mapped[List["DengueCase"]] = relationship(back_populates="district")
    climate: Mapped[List["ClimateObservation"]] = relationship(back_populates="district")

    __table_args__ = (
        UniqueConstraint("code", name="uq_geographic_units_code"),
        Index("ix_geographic_units_parent_level", "parent_code", "level"),
    )


class EpiWeek(Base, IdMixin, TimestampMixin):
    __tablename__ = "epi_weeks"

    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    season: Mapped[str] = mapped_column(String(20), default="", nullable=False)

    __table_args__ = (
        UniqueConstraint("year", "week", name="uq_epi_weeks_year_week"),
        Index("ix_epi_weeks_start", "start_date"),
    )

    @property
    def iso_week_key(self) -> str:
        return f"{self.year}-W{self.week:02d}"
