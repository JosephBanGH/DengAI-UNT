"""Esquemas Pydantic compartidos."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    message: str
    detail: Optional[str] = None


class Page(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int = 1
    size: int = 25
    pages: int = 1


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class GeoUnitRef(ORMModel):
    code: str
    name: str
    province: str = ""
    department: str = ""


class Periodo(BaseModel):
    start: Optional[date] = None
    end: Optional[date] = None


class KPI(BaseModel):
    key: str
    label: str
    value: float
    unit: str = ""
    change_pct: Optional[float] = None
    trend: Optional[str] = None
    hint: Optional[str] = None


class SeriesPoint(BaseModel):
    x: str
    y: float
    label: Optional[str] = None


class HealthStatus(BaseModel):
    status: str
    version: str
    environment: str
    database: str
    database_url_scheme: str
    postgis: bool
    ml_backend: dict
    integrations: dict
    timestamp: datetime
    limitations: List[str] = Field(default_factory=list)


class ConfigValue(BaseModel):
    key: str
    value: Any
    description: str = ""
    updated_at: Optional[datetime] = None
    updated_by: str = ""
