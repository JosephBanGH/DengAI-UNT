"""Esquemas del laboratorio de IA, pronósticos y explicabilidad."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AlgorithmInfo(BaseModel):
    key: str
    name: str
    family: str
    role: str
    description: str
    spatial: bool
    temporal: bool
    attention: bool
    backend: str
    status: str
    default_horizons: List[int] = Field(default_factory=list)


class ExperimentCreate(BaseModel):
    algorithm: str
    name: Optional[str] = None
    dataset_id: Optional[int] = None
    n_trials: int = Field(default=50, ge=1, le=150)
    horizons: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    validation_strategy: str = "walk_forward"
    notes: str = ""


class ExperimentOut(BaseModel):
    id: int
    run_id: str
    name: str
    algorithm: str
    status: str
    stage: str
    progress: float
    n_trials: int
    trials_completed: int
    validation_strategy: str
    horizons: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    best_trial: Optional[Dict[str, Any]] = None
    mlflow_run_id: str = ""
    is_synthetic_data: bool
    notes: str = ""
    error_message: str = ""
    started_at: datetime
    finished_at: Optional[datetime]
    duration_seconds: Optional[float]
    created_by: str


class ModelVersionOut(BaseModel):
    id: int
    algorithm: str
    version: str
    status: str
    is_production: bool
    training_weeks: int
    metrics: Optional[Dict[str, Any]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    promoted_at: Optional[datetime]
    promoted_by: str
    created_at: datetime
    artifact_checksum: str = ""


class MetricRow(BaseModel):
    algorithm: str
    split: str
    horizon_weeks: int
    metric: str
    value: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    n_observations: int = 0


class PredictionOut(BaseModel):
    district_code: str
    district_name: str
    target_week_start: date
    target_year: int
    target_week: int
    horizon_weeks: int
    predicted_cases: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    risk_level: str
    risk_score: Optional[float]
    incidence_per_100k: Optional[float]


class ForecastRequest(BaseModel):
    district_codes: Optional[List[str]] = None
    horizons: List[int] = Field(default_factory=lambda: [1, 2, 4])
    model_version_id: Optional[int] = None


class ForecastResponse(BaseModel):
    run_key: str
    model_version: str
    algorithm: str
    generated_at: datetime
    origin_week: str
    is_synthetic_data: bool
    predictions: List[PredictionOut]
    disclaimer: str


class RiskRow(BaseModel):
    district_code: str
    district_name: str
    level: str
    score: float
    predicted_cases: Optional[float]
    incidence_per_100k: Optional[float]
    population: Optional[int]
    rationale: str
    factors: Optional[Dict[str, Any]] = None


class RiskMatrixResponse(BaseModel):
    generated_at: datetime
    method: str
    thresholds: Dict[str, float]
    rows: List[RiskRow]
    summary: Dict[str, int]


class ShapFactor(BaseModel):
    feature: str
    label: str
    contribution: float
    direction: str
    value: Optional[float] = None


class ExplanationResponse(BaseModel):
    district_code: str
    district_name: str
    model_version: str
    algorithm: str
    method: str
    base_value: Optional[float] = None
    prediction: Optional[float] = None
    factors: List[ShapFactor]
    temporal_importance: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: str
    caveat: str


class ModelComparison(BaseModel):
    rows: List[MetricRow]
    ranking: List[Dict[str, Any]]
    tests: List[Dict[str, Any]]
    winner: Optional[str] = None
    winner_basis: str = ""
