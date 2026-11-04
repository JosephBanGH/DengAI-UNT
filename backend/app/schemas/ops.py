"""Esquemas de alertas, reportes, asistente y automatización."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AlertOut(BaseModel):
    id: int
    alert_key: str
    district_code: str
    district_name: str = ""
    epi_year: int
    epi_week: int
    horizon_weeks: int
    risk_level: str
    risk_score: Optional[float]
    predicted_cases: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    incidence_per_100k: Optional[float]
    rationale: str
    status: str
    acknowledged_by: str
    acknowledged_at: Optional[datetime]
    closed_by: str
    closed_at: Optional[datetime]
    notes: str
    generated_at: datetime
    is_synthetic_data: bool
    factors: Optional[Dict[str, Any]] = None


class AlertUpdate(BaseModel):
    action: str = Field(pattern="^(acknowledge|close|reopen)$")
    notes: str = ""


class AlertGenerateRequest(BaseModel):
    horizons: List[int] = Field(default_factory=lambda: [1, 2])
    min_level: str = "HIGH"


class ReportCreate(BaseModel):
    report_type: str
    format: str = Field(pattern="^(pdf|docx|xlsx)$")
    title: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    period_start: Optional[date] = None
    period_end: Optional[date] = None


class ReportOut(BaseModel):
    id: int
    report_key: str
    title: str
    category: str
    report_type: str
    format: str
    status: str
    file_size: int
    filters: Optional[Dict[str, Any]] = None
    period_start: Optional[date]
    period_end: Optional[date]
    model_version: str
    generated_by: str
    generated_at: datetime
    is_synthetic_data: bool
    methodology_notes: str = ""
    error_message: str = ""


class ReportTypeInfo(BaseModel):
    key: str
    category: str
    title: str
    description: str
    formats: List[str]
    requires_period: bool = False


class AssistantAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    district_code: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=10)


class AssistantSource(BaseModel):
    document: str
    section: str
    score: float
    snippet: str


class AssistantAnswer(BaseModel):
    answer: str
    grounded: bool
    provider: str
    sources: List[AssistantSource] = Field(default_factory=list)
    facts: Dict[str, Any] = Field(default_factory=dict)
    caveat: str = ""
    latency_ms: Optional[int] = None


class WorkflowInfo(BaseModel):
    code: str
    name: str
    description: str
    schedule: str
    file: str
    steps: List[str]
    last_execution: Optional[Dict[str, Any]] = None


class WorkflowExecutionOut(BaseModel):
    id: int
    workflow_code: str
    workflow_name: str
    trigger: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    duration_ms: Optional[int]
    attempts: int
    items_processed: int
    error_message: str
    triggered_by: str
    output: Optional[Dict[str, Any]] = None


class DashboardOverview(BaseModel):
    generated_at: datetime
    data_freshness: Optional[datetime]
    is_synthetic_data: bool
    kpis: List[Dict[str, Any]]
    weekly_series: List[Dict[str, Any]]
    districts_top_incidence: List[Dict[str, Any]]
    districts_top_risk: List[Dict[str, Any]]
    risk_distribution: List[Dict[str, Any]]
    active_alerts: int
    alerts_by_level: List[Dict[str, Any]]
    climate_summary: List[Dict[str, Any]]
    model_performance: List[Dict[str, Any]]
    integration_status: List[Dict[str, Any]]
    data_quality: Dict[str, Any] = Field(default_factory=dict)
