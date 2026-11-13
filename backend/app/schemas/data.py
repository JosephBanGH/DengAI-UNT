"""Esquemas de datos, fuentes, importaciones y conjuntos de datos."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DataSourceOut(BaseModel):
    id: int
    code: str
    name: str
    kind: str
    provider: str
    url: str
    license: str
    access_method: str
    integration_status: str
    is_synthetic: bool
    refresh_frequency: str
    last_success_at: Optional[datetime]
    last_error: str = ""
    notes: str = ""


class DataSourceUpdate(BaseModel):
    integration_status: Optional[str] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


class IntegrationProbeResult(BaseModel):
    source_code: str
    reachable: bool
    status: str
    message: str
    latency_ms: Optional[int] = None
    sample: Optional[Dict[str, Any]] = None


class ImportJobOut(BaseModel):
    id: int
    job_key: str
    source_code: Optional[str] = None
    dataset_name: str
    kind: str
    status: str
    file_name: str
    rows_total: int
    rows_inserted: int
    rows_updated: int
    rows_rejected: int
    rows_duplicated: int
    started_at: datetime
    finished_at: Optional[datetime]
    duration_ms: Optional[int]
    initiated_by: str
    is_synthetic: bool
    error_message: str = ""


class ImportRowErrorOut(BaseModel):
    row_number: int
    column: str
    reason: str
    raw_value: str


class ImportPreview(BaseModel):
    kind: str
    columns: List[str]
    sample_rows: List[Dict[str, Any]]
    detected_rows: int
    validation: List[Dict[str, Any]]
    ready: bool


class DataQualityOut(BaseModel):
    id: int
    dataset_name: str
    check_name: str
    severity: str
    rows_checked: int
    rows_flagged: int
    details: Optional[Dict[str, Any]]
    created_at: datetime


class CaseOut(BaseModel):
    district_code: str
    district_name: str
    epi_year: int
    epi_week: int
    week_start: date
    confirmed_cases: int
    probable_cases: int
    hospitalized: int
    deaths: int
    is_synthetic: bool
    data_quality_flag: str


class ClimateOut(BaseModel):
    district_code: str
    epi_year: int
    epi_week: int
    week_start: date
    temperature_mean: Optional[float]
    temperature_max: Optional[float]
    temperature_min: Optional[float]
    humidity_mean: Optional[float]
    precipitation_mm: Optional[float]
    provider: str
    is_synthetic: bool


class DatasetOut(BaseModel):
    id: int
    slug: str
    name: str
    version: str
    description: str
    rows: int
    districts: int
    weeks: int
    start_week: Optional[date]
    end_week: Optional[date]
    is_synthetic: bool
    status: str
    checksum: str
    created_at: datetime
    created_by: str
    source_summary: Optional[Dict[str, Any]] = None
    feature_schema: Optional[Dict[str, Any]] = None


class DatasetBuildRequest(BaseModel):
    name: str = Field(min_length=3, max_length=160)
    description: str = ""
    include_synthetic: bool = True
    max_lag: int = Field(default=12, ge=1, le=24)


class DataDictionaryEntry(BaseModel):
    column: str
    dtype: str
    description: str
    unit: str = ""
    source: str = ""
    nullable: bool = True


class RealIngestRequest(BaseModel):
    start_year: Optional[int] = Field(default=None, ge=2000, le=2100)
    end_year: Optional[int] = Field(default=None, ge=2000, le=2100)
    district_codes: Optional[List[str]] = None
    include_cases: bool = True
    include_climate: bool = True
    force_download: bool = False


class RealIngestResponse(BaseModel):
    requested: Dict[str, Any]
    steps: Dict[str, Any]
    effective_data_mode: str
    provenance: Dict[str, Any]


class PanelSummary(BaseModel):
    data_mode: str
    is_synthetic: bool
    rows: int
    districts: int
    weeks: int
    start_week: Optional[str] = None
    end_week: Optional[str] = None
    source_summary: Dict[str, Any] = Field(default_factory=dict)
    per_district: List[Dict[str, Any]] = Field(default_factory=list)


class QualitySummary(BaseModel):
    total_cases: int
    total_climate: int
    districts_covered: int
    weeks_covered: int
    missing_climate_pct: float
    duplicate_rows: int
    anomalies: int
    synthetic_share_pct: float
    checks: List[DataQualityOut] = Field(default_factory=list)


class DictionaryUpdateEntry(BaseModel):
    description: str = Field(..., min_length=2, max_length=500)
    unit: str = Field(default="", max_length=50)
    source: str = Field(default="", max_length=100)


class GroupValidationResponse(BaseModel):
    audited_at: str
    dataset_id: Optional[int] = None
    dataset_name: str
    total_records: int
    is_synthetic: bool
    data_mode: str
    groups_summary: Dict[str, Any]
    partitions_audit: Dict[str, Any]
    leakage_risk: Dict[str, Any]
    methodological_recommendations: List[str]


class DatasetStatisticsResponse(BaseModel):
    generated_at: str
    dataset_id: Optional[int] = None
    dataset_name: str
    total_records: int
    is_synthetic: bool
    univariate_statistics: List[Dict[str, Any]]
    correlations: List[Dict[str, Any]]
    district_breakdown: List[Dict[str, Any]]
    automated_explanations: List[Dict[str, Any]]

