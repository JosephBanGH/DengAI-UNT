"""Modelos del laboratorio de IA: experimentos, versiones, predicciones y métricas."""
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

ALGORITHM_CHOICES = (
    "xgboost",
    "lstm",
    "convlstm",
    "cnn_lstm",
    "convlstm_attention",
)


class Experiment(Base, IdMixin, TimestampMixin):
    __tablename__ = "experiments"

    run_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(40), default="", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    n_trials: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    trials_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    validation_strategy: Mapped[str] = mapped_column(String(60), default="walk_forward", nullable=False)
    horizons: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    search_space: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    best_trial: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    mlflow_run_id: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    error_message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_by: Mapped[str] = mapped_column(String(255), default="system", nullable=False, index=True)

    model_versions: Mapped[List["ModelVersion"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class ModelVersion(Base, IdMixin, TimestampMixin):
    __tablename__ = "model_versions"

    experiment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(30), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(400), default="", nullable=False)
    artifact_checksum: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="CANDIDATE", nullable=False, index=True)
    is_production: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    training_weeks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    feature_names: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    promoted_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    experiment: Mapped[Optional[Experiment]] = relationship(back_populates="model_versions")
    predictions: Mapped[List["Prediction"]] = relationship(back_populates="model_version")

    __table_args__ = (UniqueConstraint("algorithm", "version", name="uq_model_versions_algorithm_version"),)


class Prediction(Base, IdMixin, TimestampMixin):
    __tablename__ = "predictions"

    run_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    origin_year: Mapped[int] = mapped_column(Integer, nullable=False)
    origin_week: Mapped[int] = mapped_column(Integer, nullable=False)
    target_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    target_week: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    target_week_start: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    horizon_weeks: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    predicted_cases: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    interval_level: Mapped[float] = mapped_column(Float, default=0.9, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), default="UNKNOWN", nullable=False, index=True)
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    incidence_per_100k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    is_synthetic_data: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    model_version: Mapped[Optional["ModelVersion"]] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_district_target", "district_code", "target_week_start"),
        Index("ix_predictions_run_horizon", "run_key", "horizon_weeks"),
    )


class RiskClassification(Base, IdMixin, TimestampMixin):
    __tablename__ = "risk_classifications"

    district_code: Mapped[str] = mapped_column(
        ForeignKey("geographic_units.code", ondelete="CASCADE"), nullable=False, index=True
    )
    epi_year: Mapped[int] = mapped_column(Integer, nullable=False)
    epi_week: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(40), default="incidence_threshold", nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="", nullable=False)
    predicted_cases: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    incidence_per_100k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    factors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("district_code", "epi_year", "epi_week", name="uq_risk_district_week"),
    )


class MetricResult(Base, IdMixin, TimestampMixin):
    """Métricas por modelo, horizonte y partición, con IC bootstrap."""

    __tablename__ = "metric_results"

    experiment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=True, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(20), default="test", nullable=False)
    horizon_weeks: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    metric: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    ci_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ci_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_observations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class StatisticalTest(Base, IdMixin, TimestampMixin):
    __tablename__ = "statistical_tests"

    experiment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=True, index=True
    )
    test_name: Mapped[str] = mapped_column(String(60), nullable=False)
    comparison: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    statistic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_value_adjusted: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    significant: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    effect_size: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
