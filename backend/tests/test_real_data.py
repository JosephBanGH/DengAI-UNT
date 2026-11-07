"""Pruebas de la capa real: raw inmutable, agregación NASA y MINSA, modos de datos."""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.integrations import minsa_dengue, nasa_power, raw_store
from app.ml.features import resolve_data_mode
from app.models.epi import ClimateObservation, DengueCase
from app.models.geo import GeographicUnit


def _daily_rows(start: date, days: int) -> list[dict]:
    rows = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        rows.append(
            {
                "date": day,
                "temperature_mean": 20.0 + offset,
                "temperature_max": 26.0 + offset,
                "temperature_min": 14.0 + offset,
                "humidity_mean": 70.0,
                "precipitation_mm": 1.0,
                "wind_speed": 2.0,
                "provider": "NASA_POWER",
            }
        )
    return rows


def test_nasa_weekly_aggregation_uses_iso_weeks():
    # 2024-01-01 es lunes de la semana ISO 2024-W01.
    rows = _daily_rows(date(2024, 1, 1), 10)
    weekly = nasa_power.aggregate_weekly(rows)
    assert len(weekly) == 2
    first = weekly[0]
    assert (first["epi_year"], first["epi_week"]) == (2024, 1)
    assert first["week_start"] == date(2024, 1, 1)
    assert first["days_observed"] == 7
    assert first["temperature_mean"] == pytest.approx(23.0)
    assert first["precipitation_mm"] == pytest.approx(7.0)
    assert weekly[1]["days_observed"] == 3


def test_minsa_aggregates_and_zero_fills(tmp_path):
    csv_path = tmp_path / "minsa.csv"
    pd.DataFrame(
        [
            ["LA LIBERTAD", "TRUJILLO", "TRUJILLO", "DENGUE SIN SIGNOS DE ALARMA", 2020, 1, "C", "130101"],
            ["LA LIBERTAD", "TRUJILLO", "TRUJILLO", "DENGUE CON SIGNOS DE ALARMA", 2020, 1, "P", "130101"],
            ["LA LIBERTAD", "TRUJILLO", "TRUJILLO", "DENGUE CON SIGNOS DE ALARMA", 2020, 2, "C", "130101"],
            ["LA LIBERTAD", "ASCOPE", "MOCHE", "DENGUE GRAVE", 2020, 1, "C", "130102"],
        ],
        columns=["departamento", "provincia", "distrito", "enfermedad", "ano", "semana", "tipo_dx", "ubigeo"],
    ).to_csv(csv_path, index=False)

    panel, metadata = minsa_dengue.build_district_week(csv_path, start_year=2020, end_year=2020)
    assert metadata["source"] == "MINSA_VIGILANCIA"
    assert set(metadata["districts"]) == {"PE-LL-TRU", "PE-LL-MOC"}
    assert metadata["rows_observed"] == 3
    assert metadata["rows_zero_filled"] > 0

    trujillo_w1 = panel[(panel["district_code"] == "PE-LL-TRU") & (panel["semana"] == 1)].iloc[0]
    assert trujillo_w1["confirmed_cases"] == 1
    assert trujillo_w1["probable_cases"] == 1
    assert trujillo_w1["severity"] == "CON_SIGNOS_ALARMA"
    moche_w1 = panel[(panel["district_code"] == "PE-LL-MOC") & (panel["semana"] == 1)].iloc[0]
    assert moche_w1["severity"] == "GRAVE"
    assert (panel["data_quality_flag"] == "REAL_ZERO_FILLED").any()


def test_raw_store_manifest_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path / "raw"))
    content = b"a,b\n1,2\n"
    entry = raw_store.save_raw_bytes(
        source="TEST_SOURCE",
        filename="sample.csv",
        content=content,
        url="https://example.org/sample.csv",
        license="test",
        provider="test",
    )
    assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    stored = tmp_path / "raw" / "TEST_SOURCE" / "sample.csv"
    assert stored.read_bytes() == content
    manifest = raw_store.read_manifest()
    assert manifest[-1]["source"] == "TEST_SOURCE"
    assert raw_store.find_entry("TEST_SOURCE")["sha256"] == entry["sha256"]


def _isolated_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_data_mode_never_mixes_real_and_synthetic():
    session = _isolated_session()
    session.add(
        GeographicUnit(
            code="PE-LL-TRU", name="Trujillo", level="district", parent_code="PE-LL",
            province="Trujillo", department="La Libertad", altitude_m=34, area_km2=45.2,
            population=314242, population_year=2024, centroid_lat=-8.1, centroid_lon=-79.0,
        )
    )
    session.add(
        DengueCase(
            district_code="PE-LL-TRU", epi_year=2024, epi_week=1, week_start=date(2024, 1, 1),
            confirmed_cases=10, is_synthetic=True,
        )
    )
    session.commit()
    assert resolve_data_mode(session) == "synthetic"
    with pytest.raises(ValueError):
        resolve_data_mode(session, mode="real")

    # Casos reales sin clima real: auto debe seguir en sintético para no mezclar.
    session.add(
        DengueCase(
            district_code="PE-LL-TRU", epi_year=2024, epi_week=2, week_start=date(2024, 1, 8),
            confirmed_cases=12, is_synthetic=False,
        )
    )
    session.commit()
    assert resolve_data_mode(session) == "synthetic"

    # Con clima real, auto cambia a real y el modo explícito deja de fallar.
    session.add(
        ClimateObservation(
            district_code="PE-LL-TRU", epi_year=2024, epi_week=2, week_start=date(2024, 1, 8),
            temperature_mean=25.0, is_synthetic=False, provider="NASA_POWER",
        )
    )
    session.commit()
    assert resolve_data_mode(session) == "real"
    assert resolve_data_mode(session, mode="real") == "real"


def test_provenance_and_panel_endpoints(client, admin_headers):
    provenance = client.get("/api/v1/data/provenance", headers=admin_headers)
    assert provenance.status_code == 200, provenance.text
    body = provenance.json()
    assert body["cases"]["total"] > 0
    assert body["effective_data_mode"] in {"real", "synthetic"}

    panel = client.get("/api/v1/data/panel", headers=admin_headers, params={"mode": "synthetic"})
    assert panel.status_code == 200, panel.text
    payload = panel.json()
    assert payload["is_synthetic"] is True
    assert payload["rows"] > 0
    assert payload["districts"] >= 10
