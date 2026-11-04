"""Pruebas de autenticación, RBAC y sesiones (M01/M02)."""
from __future__ import annotations

from backend.tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, CONSULT_EMAIL, DEMO_PASSWORD


def test_health_is_public(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert any("SQLite" in limitation or "PostgreSQL" in limitation for limitation in body["limitations"])


def test_login_rejects_invalid_password(client):
    response = client.post("/api/v1/auth/login", json={"email": ADMIN_EMAIL, "password": "incorrecta"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_ERROR"


def test_login_accepts_valid_credentials_and_returns_profile(client, admin_headers):
    response = client.get("/api/v1/auth/me", headers=admin_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == ADMIN_EMAIL
    assert body["is_superuser"] is True
    assert "*" in body["permissions"]


def test_protected_endpoint_requires_token(client):
    response = client.get("/api/v1/dashboard/overview")
    assert response.status_code == 401


def test_refresh_token_rotates_access_token(client):
    login = client.post("/api/v1/auth/login", json={"email": CONSULT_EMAIL, "password": DEMO_PASSWORD})
    tokens = login.json()
    refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"]


def test_consultation_role_cannot_manage_users(client, consult_headers):
    response = client.get("/api/v1/users", headers=consult_headers)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "ACCESS_DENIED"


def test_consultation_role_can_read_dashboard(client, consult_headers):
    response = client.get("/api/v1/dashboard/overview", headers=consult_headers)
    assert response.status_code == 200


def test_logout_revokes_session(client):
    login = client.post("/api/v1/auth/login", json={"email": CONSULT_EMAIL, "password": DEMO_PASSWORD})
    access = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {access}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200
    assert client.post("/api/v1/auth/logout", headers=headers).status_code == 200
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
