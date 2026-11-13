"""Configuración central de la aplicación (12-factor, vía variables de entorno)."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - fallback when pydantic-settings is not installed
    import os

    class BaseSettings(BaseModel):  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for name in self.__class__.model_fields:
                if name not in kwargs and name in os.environ:
                    kwargs[name] = os.environ[name]
            super().__init__(**kwargs)

    def SettingsConfigDict(**kwargs):  # type: ignore[misc]
        return ConfigDict(extra="ignore")

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


def _default_database_url() -> str:
    """SQLite es el valor por defecto para el arranque local sin infraestructura."""
    data_dir = PROJECT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(data_dir / 'dengai.db').as_posix()}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Aplicación ---
    APP_NAME: str = "SP-01 Dengue Intelligence"
    APP_SHORT_NAME: str = "DengAI-UNT"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "test", "production"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    BASE_URL: str = "http://127.0.0.1:8000"

    # --- Seguridad ---
    SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_MIN_LENGTH: int = 10
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCK_MINUTES: int = 15

    # --- Base de datos ---
    DATABASE_URL: str = Field(default_factory=_default_database_url)
    DATABASE_ECHO: bool = False
    POSTGIS_ENABLED: bool = False

    # --- CORS ---
    CORS_ORIGINS: List[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:8080", "http://localhost:8080", "http://127.0.0.1:8001", "http://localhost:8001",
    ]

    # --- Bootstrap (se usan solo si la BD está vacía) ---
    BOOTSTRAP_ADMIN_EMAIL: str = "admin@dengai.unt.edu.pe"
    BOOTSTRAP_ADMIN_PASSWORD: str = "Admin#Dengue2025"
    SEED_DEMO_USERS: bool = True
    DEMO_USER_PASSWORD: str = "Demo#Dengue2025"

    # --- Machine Learning ---
    ML_DEFAULT_TRIALS: int = 50
    ML_MAX_TRIALS: int = 150
    ML_TEST_WEEKS: int = 26
    ML_VALIDATION_WEEKS: int = 26
    ML_MAX_LAG: int = 12
    ML_MODELS_DIR: str = "data/models"
    MLFLOW_TRACKING_URI: str = ""
    MLFLOW_ENABLED: bool = False

    # --- Datos externos ---
    # NASA POWER no expone un endpoint semanal: se descarga la serie diaria y se
    # agrega a semanas ISO en app/integrations/nasa_power.py.
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    NASA_POWER_TIMEOUT: float = 45.0
    NASA_POWER_START_YEAR: int = 2015
    # 2024 es el último año completo en la fuente MINSA (2025 está parcial).
    NASA_POWER_END_YEAR: int = 2024
    LIVE_INTEGRATIONS_ENABLED: bool = False

    # MINSA/CDC Perú · datos abiertos de vigilancia de dengue (microdatos por caso).
    MINSA_DENGUE_URL: str = (
        "https://www.datosabiertos.gob.pe/sites/default/files/"
        "datos_abiertos_vigilancia_dengue_2000_2025_0.csv"
    )
    MINSA_START_YEAR: int = 2015
    # 2025 se excluye por defecto: el CSV oficial aún está parcial y sesgaría el
    # conjunto de prueba temporal.
    MINSA_END_YEAR: int = 2024
    RAW_DATA_DIR: str = "data/raw"

    # Modo de datos del pipeline: auto usa datos reales si existen; synthetic fuerza
    # el conjunto de demostración; real exige datos reales.
    DATA_MODE: Literal["auto", "synthetic", "real"] = "auto"

    # --- Almacenamiento de objetos (MinIO) ---
    MINIO_ENABLED: bool = False
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "dengai"
    MINIO_SECRET_KEY: str = "dengai-minio"
    MINIO_BUCKET: str = "dengai"
    MINIO_SECURE: bool = False

    # --- MLflow ---
    MLFLOW_EXPERIMENT_NAME: str = "SP01-Dengue-Intelligence"
    MLFLOW_ARTIFACT_LOCATION: str = ""

    # --- Capa de lenguaje (RAG / explicaciones) ---
    LLM_PROVIDER: Literal["grounded_template", "openai_compatible", "gemini"] = "grounded_template"
    GEMINI_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT: float = 30.0
    RAG_TOP_K: int = 5

    # --- Reportes ---
    REPORTS_DIR: str = "data/reports"
    REVIEWER_EMAIL: str = "comite@dengai.unt.edu.pe"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _validate_database_url(cls, value: Any) -> str:
        if not value or not str(value).strip():
            return _default_database_url()
        url = str(value).strip()
        if "@db:" in url and not Path("/.dockerenv").exists():
            return _default_database_url()
        return url

    @field_validator("NASA_POWER_BASE_URL", mode="before")
    @classmethod
    def _validate_nasa_power_url(cls, value: Any) -> str:
        if not value or not str(value).strip():
            return "https://power.larc.nasa.gov/api/temporal/daily/point"
        url = str(value).strip()
        return url.replace("/temporal/weekly/", "/temporal/daily/")

    @field_validator("SECRET_KEY")
    @classmethod
    def _ensure_secret(cls, value: str) -> str:
        if value:
            return value
        env_file = PROJECT_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("SECRET_KEY="):
                    candidate = line.split("=", 1)[1].strip()
                    if candidate:
                        return candidate
        generated = secrets.token_urlsafe(48)
        with env_file.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# Clave generada automáticamente en el primer arranque\nSECRET_KEY={generated}\n")
        return generated

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def models_dir(self) -> Path:
        path = PROJECT_DIR / self.ML_MODELS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_dir(self) -> Path:
        path = PROJECT_DIR / self.REPORTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        path = self.models_dir.parent
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_dir(self) -> Path:
        path = PROJECT_DIR / self.RAW_DATA_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
