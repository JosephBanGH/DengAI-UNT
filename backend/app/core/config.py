"""Configuración central de la aplicación (12-factor, vía variables de entorno)."""
from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    CORS_ORIGINS: List[str] = ["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000"]

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
    NASA_POWER_BASE_URL: str = "https://power.larc.nasa.gov/api/temporal/weekly/point"
    NASA_POWER_TIMEOUT: float = 20.0
    LIVE_INTEGRATIONS_ENABLED: bool = False

    # --- Capa de lenguaje (RAG / explicaciones) ---
    LLM_PROVIDER: Literal["grounded_template", "openai_compatible"] = "grounded_template"
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT: float = 30.0
    RAG_TOP_K: int = 5

    # --- Reportes ---
    REPORTS_DIR: str = "data/reports"
    REVIEWER_EMAIL: str = "comite@dengai.unt.edu.pe"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
