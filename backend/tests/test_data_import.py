"""Pruebas de importación, validación y calidad de datos (M05)."""
from __future__ import annotations

import io

CSV_VALIDO = (
    "district_code,epi_year,epi_week,confirmed_cases\n"
    "PE-LL-TRU,2026,40,120\n"
    "PE-LL-TRU,2026,41,135\n"
    "PE-LL-VLH,2026,40,18\n"
)

CSV_CON_ERRORES = (
    "district_code,epi_year,epi_week,confirmed_cases\n"
    "PE-LL-XXX,2026,40,10\n"      # distrito inexistente
    "PE-LL-TRU,2026,99,5\n"        # semana fuera de rango
    "PE-LL-TRU,2026,42,-3\n"       # casos negativos
    "PE-LL-TRU,2026,43,22\n"       # válido
)


def test_preview_detects_columns(client, admin_headers):
    response = client.post(
        "/api/v1/data/import/preview",
        headers=admin_headers,
        files={"file": ("casos.csv", io.BytesIO(CSV_VALIDO.encode("utf-8")), "text/csv")},
        data={"kind": "epidemiological"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["detected_rows"] == 3
    assert "district_code" in body["columns"]


def test_preview_rejects_missing_columns(client, admin_headers):
    response = client.post(
        "/api/v1/data/import/preview",
        headers=admin_headers,
        files={"file": ("malo.csv", io.BytesIO(b"foo,bar\n1,2\n"), "text/csv")},
        data={"kind": "epidemiological"},
    )
    assert response.status_code == 200
    assert response.json()["ready"] is False


def test_import_inserts_valid_rows_and_rejects_invalid(client, admin_headers):
    response = client.post(
        "/api/v1/data/import/cases",
        headers=admin_headers,
        files={"file": ("casos.csv", io.BytesIO(CSV_VALIDO.encode("utf-8")), "text/csv")},
        data={"is_synthetic": "false"},
    )
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "COMPLETED"
    # Las semanas pueden existir ya en el conjunto sembrado: se contabilizan como actualización.
    assert job["rows_inserted"] + job["rows_updated"] >= 2
    assert job["rows_total"] == 3


def test_import_rejects_rows_with_invalid_values(client, admin_headers):
    response = client.post(
        "/api/v1/data/import/cases",
        headers=admin_headers,
        files={"file": ("errores.csv", io.BytesIO(CSV_CON_ERRORES.encode("utf-8")), "text/csv")},
        data={"is_synthetic": "true"},
    )
    assert response.status_code == 200
    job = response.json()
    assert job["rows_rejected"] >= 3
    details = client.get(f"/api/v1/data/imports/{job['id']}/errors", headers=admin_headers).json()
    assert len(details) >= 3


def test_import_requires_permission(client, consult_headers):
    response = client.post(
        "/api/v1/data/import/cases",
        headers=consult_headers,
        files={"file": ("casos.csv", io.BytesIO(CSV_VALIDO.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 403


def test_quality_summary_reports_synthetic_share(client, admin_headers):
    response = client.get("/api/v1/data/quality", headers=admin_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total_cases"] > 0
    assert 0 <= body["synthetic_share_pct"] <= 100
