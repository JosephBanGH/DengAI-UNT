"""Pruebas unitarias y de integración para la Fase 1 - MVP (Validación de Grupos y Estadística)."""
from __future__ import annotations

import pytest
from app.db.session import SessionLocal
from app.services.dataset_service import build_and_register, data_dictionary, update_data_dictionary_entry
from app.services.statistics_service import compute_dataset_statistics
from app.services.validation_service import audit_group_validation


def test_group_validation_runs_and_passes_all_checks(client, admin_headers):
    with SessionLocal() as db:
        result = audit_group_validation(db)
        assert result is not None
        assert result["total_records"] > 0
        assert result["groups_summary"]["districts_count"] >= 10
        assert result["groups_summary"]["weeks_count"] >= 100

        # Verificación estricta de no fuga
        leakage = result["leakage_risk"]
        assert leakage["risk_level"] == "LOW"
        assert leakage["risk_score_pct"] == 0.0
        assert leakage["passed_checks"] == leakage["total_checks"]

        # Sin solapamiento temporal ni espacial
        temporal = result["partitions_audit"]["temporal_walk_forward"]
        assert temporal["status"] == "PASSED"
        assert temporal["overlap_weeks"] == 0

        spatial = result["partitions_audit"]["spatial_block_cv"]
        assert spatial["status"] == "PASSED"
        assert spatial["overlap_districts"] == 0


def test_dataset_statistics_computation(client, admin_headers):
    with SessionLocal() as db:
        stats = compute_dataset_statistics(db)
        assert stats is not None
        assert stats["total_records"] > 0

        # Univariadas
        univariate = {u["variable"]: u for u in stats["univariate_statistics"]}
        assert "confirmed_cases" in univariate
        cases_stat = univariate["confirmed_cases"]
        assert cases_stat["mean"] is not None
        assert cases_stat["median"] is not None
        assert cases_stat["iqr"] is not None
        assert cases_stat["valid_count"] > 0

        # Clima
        assert "temperature_mean" in univariate
        temp_stat = univariate["temperature_mean"]
        assert 15.0 <= temp_stat["mean"] <= 35.0

        # Correlaciones
        correlations = stats["correlations"]
        assert len(correlations) > 0
        for corr in correlations:
            assert -1.0 <= corr["pearson"] <= 1.0
            assert -1.0 <= corr["spearman"] <= 1.0

        # Explicabilidad automatizada
        explanations = stats["automated_explanations"]
        assert len(explanations) >= 2
        for exp in explanations:
            assert "topic" in exp
            assert "behavior" in exp
            assert "methodological_implication" in exp
            assert "epistemic_caution" in exp


def test_data_dictionary_update_and_audit(client, admin_headers):
    with SessionLocal() as db:
        dataset = build_and_register(
            db,
            name="Test Dataset Fase 1",
            description="Dataset para probar edición de diccionario",
            max_lag=4,
            actor="admin@dengai.unt.edu.pe",
        )
        dataset_id = dataset["id"]

        # Actualizar metadatos de una variable
        updated = update_data_dictionary_entry(
            db,
            dataset_id=dataset_id,
            column="confirmed_cases",
            description="Casos confirmados de dengue validados por la prueba",
            unit="casos confirmados/semana",
            source="MINSA / Laboratorio Referencial",
            actor="admin@dengai.unt.edu.pe",
        )
        assert updated["description"] == "Casos confirmados de dengue validados por la prueba"
        assert updated["unit"] == "casos confirmados/semana"

        # Verificar lectura
        dict_rows = data_dictionary(db, dataset_id)
        cases_entry = next((r for r in dict_rows if r["column"] == "confirmed_cases"), None)
        assert cases_entry is not None
        assert cases_entry["description"] == "Casos confirmados de dengue validados por la prueba"


def test_api_endpoints_group_validation_and_statistics(client, admin_headers):
    # Endpoint de validación de grupos
    resp_val = client.get("/api/v1/data/group-validation", headers=admin_headers)
    assert resp_val.status_code == 200
    val_body = resp_val.json()
    assert val_body["leakage_risk"]["risk_level"] == "LOW"

    # Endpoint de estadísticas
    resp_stat = client.get("/api/v1/data/statistics", headers=admin_headers)
    assert resp_stat.status_code == 200
    stat_body = resp_stat.json()
    assert len(stat_body["univariate_statistics"]) > 0
    assert len(stat_body["automated_explanations"]) > 0
