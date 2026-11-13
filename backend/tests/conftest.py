"""Configuración de pruebas: base de datos temporal aislada y cliente de la API."""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_DIR / "backend"
LIBS = PROJECT_DIR / "libs"

sys.path.insert(0, str(BACKEND_DIR))
if LIBS.is_dir():
    # Entorno local: dependencias instaladas con `pip --target libs`.
    # En Docker se usan las del site-packages del intérprete, que no deben filtrarse.
    sys.path.insert(0, str(LIBS))
    sys.path = [
        entry
        for entry in sys.path
        if not entry.lower().replace("\\", "/").rstrip("/").endswith("site-packages")
    ]

_TMP_DB = pathlib.Path(tempfile.gettempdir()) / f"dengai_tests_{os.getpid()}.db"
if _TMP_DB.exists():
    try:
        _TMP_DB.unlink()
    except Exception:
        pass
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"
os.environ["ENVIRONMENT"] = "test"
os.environ["SEED_DEMO_USERS"] = "true"
os.environ["ML_TEST_WEEKS"] = "12"
os.environ["ML_VALIDATION_WEEKS"] = "12"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production-use"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db.session import SessionLocal, engine  # noqa: E402
from app.db.seed import run_seed  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Base  # noqa: E402

ADMIN_EMAIL = "admin@dengai.unt.edu.pe"
ADMIN_PASSWORD = "Admin#Dengue2025"
CONSULT_EMAIL = "consulta@dengai.unt.edu.pe"
DEMO_PASSWORD = "Demo#Dengue2025"


@pytest.fixture(scope="session", autouse=True)
def _prepare_database():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        run_seed(session)
    finally:
        session.close()
    yield


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, email: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture(scope="session")
def admin_token(client: TestClient) -> str:
    return _login(client, ADMIN_EMAIL, ADMIN_PASSWORD)


@pytest.fixture(scope="session")
def consult_token(client: TestClient) -> str:
    return _login(client, CONSULT_EMAIL, DEMO_PASSWORD)


@pytest.fixture()
def admin_headers(admin_token: str) -> dict:
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture()
def consult_headers(consult_token: str) -> dict:
    return {"Authorization": f"Bearer {consult_token}"}
