"""Registro de todos los modelos ORM."""
from app.db.base import Base
from app.models.epi import (
    ClimateObservation,
    DataQualityRecord,
    DataSource,
    Dataset,
    DengueCase,
    EnvironmentalVariable,
    ImportJob,
    ImportRowError,
)
from app.models.geo import EpiWeek, GeographicUnit
from app.models.ml import (
    ALGORITHM_CHOICES,
    Experiment,
    MetricResult,
    ModelVersion,
    Prediction,
    RiskClassification,
    StatisticalTest,
)
from app.models.ops import Alert, AssistantQuery, Report, WorkflowExecution
from app.models.security import (
    AuditEvent,
    LoginAttempt,
    Permission,
    Role,
    SystemConfig,
    User,
    UserSession,
    role_permissions,
    user_roles,
)

__all__ = [
    "Base",
    "ALGORITHM_CHOICES",
    "Alert",
    "AssistantQuery",
    "AuditEvent",
    "ClimateObservation",
    "DataQualityRecord",
    "DataSource",
    "Dataset",
    "DengueCase",
    "EnvironmentalVariable",
    "EpiWeek",
    "Experiment",
    "GeographicUnit",
    "ImportJob",
    "ImportRowError",
    "LoginAttempt",
    "MetricResult",
    "ModelVersion",
    "Permission",
    "Prediction",
    "Report",
    "RiskClassification",
    "Role",
    "StatisticalTest",
    "SystemConfig",
    "User",
    "UserSession",
    "WorkflowExecution",
    "role_permissions",
    "user_roles",
]
