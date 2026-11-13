"""Pruebas antifuga temporal y de protección de artefactos (Sesión 1 - Fase 2)."""
from __future__ import annotations

import pathlib
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.db.session import SessionLocal  # noqa: E402
from app.ml.features import (  # noqa: E402
    PanelResult,
    build_dataset,
    build_features,
    feature_columns,
    load_panel,
)
from app.ml.trainer import resolve_artifact_path  # noqa: E402
from app.models.epi import ClimateObservation, DengueCase  # noqa: E402
from app.models.geo import GeographicUnit  # noqa: E402
from training.common import resolve_safe_output_dir, resolve_xgb_n_estimators  # noqa: E402


def _synthetic_toy_panel(n_weeks: int = 80, *, missing_climate_idx: int | None = None) -> PanelResult:
    """Construye un panel mínimo determinista para pruebas causales."""
    rows = []
    districts = ["PE-LL-TRU", "PE-LL-ASC"]
    start = date(2022, 1, 3)
    weeks = [start + timedelta(weeks=i) for i in range(n_weeks)]
    for d_idx, code in enumerate(districts):
        for w_idx, wk in enumerate(weeks):
            temp = 20.0 + 0.1 * w_idx + d_idx
            hum = 70.0 + (w_idx % 5)
            rain = 2.0 + (w_idx % 4)
            if missing_climate_idx is not None and w_idx == missing_climate_idx:
                temp = np.nan
                hum = np.nan
                rain = np.nan
            rows.append(
                {
                    "district_code": code,
                    "week_start": wk,
                    "epi_year": wk.isocalendar().year,
                    "epi_week": wk.isocalendar().week,
                    "confirmed_cases": float(5 + (w_idx % 7) + d_idx * 3),
                    "hospitalized": 0.0,
                    "is_synthetic": True,
                    "temperature_mean": temp,
                    "temperature_max": temp + 4.0 if np.isfinite(temp) else np.nan,
                    "temperature_min": temp - 4.0 if np.isfinite(temp) else np.nan,
                    "humidity_mean": hum,
                    "precipitation_mm": rain,
                    "wind_speed": 2.5,
                    "solar_radiation": 18.0,
                    "altitude_m": 34.0 + d_idx * 100.0,
                    "population": 100000.0,
                    "area_km2": 50.0,
                    "latitude": -8.1 + d_idx * 0.2,
                    "longitude": -79.0 - d_idx * 0.2,
                    "province": "Trujillo",
                    "population_density": 2000.0,
                }
            )
    df = pd.DataFrame(rows).sort_values(["district_code", "week_start"]).reset_index(drop=True)
    return PanelResult(
        panel=df,
        districts=districts,
        district_names={"PE-LL-TRU": "Trujillo", "PE-LL-ASC": "Ascope"},
        weeks=weeks,
        is_synthetic=True,
        data_mode="synthetic",
    )


def test_load_panel_never_uses_bfill_future_leakage():
    """Si una semana inicial carece de clima, load_panel NO debe copiar el valor futuro vía bfill."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db.base import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        db.add(
            GeographicUnit(
                code="PE-LL-TRU",
                name="Trujillo",
                level="district",
                parent_code="PE-LL",
                province="Trujillo",
                department="La Libertad",
                altitude_m=34,
                area_km2=45.0,
                population=300000,
                population_year=2024,
                centroid_lat=-8.1,
                centroid_lon=-79.0,
            )
        )
        w1 = date(2024, 1, 1)
        w2 = date(2024, 1, 8)
        w3 = date(2024, 1, 15)
        db.add_all(
            [
                DengueCase(district_code="PE-LL-TRU", epi_year=2024, epi_week=1, week_start=w1, confirmed_cases=3, is_synthetic=False),
                DengueCase(district_code="PE-LL-TRU", epi_year=2024, epi_week=2, week_start=w2, confirmed_cases=4, is_synthetic=False),
                DengueCase(district_code="PE-LL-TRU", epi_year=2024, epi_week=3, week_start=w3, confirmed_cases=5, is_synthetic=False),
                # Semana 1 no tiene clima; semana 2 tiene 99.0 °C; semana 3 no tiene clima (debe heredar 99.0 por ffill).
                ClimateObservation(district_code="PE-LL-TRU", epi_year=2024, epi_week=2, week_start=w2, temperature_mean=99.0, is_synthetic=False, provider="NASA_POWER"),
            ]
        )
        db.commit()

        panel_res = load_panel(db, mode="real")
        df = panel_res.panel.sort_values("week_start").reset_index(drop=True)
        # Semana 1 NO debe haber heredado 99.0 desde el futuro (bfill eliminado).
        assert pd.isna(df.loc[0, "temperature_mean"])
        # Semana 2 tiene 99.0 y semana 3 hereda 99.0 causalmente hacia adelante (ffill).
        assert float(df.loc[1, "temperature_mean"]) == pytest.approx(99.0)
        assert float(df.loc[2, "temperature_mean"]) == pytest.approx(99.0)
    finally:
        db.close()


def test_perturbing_future_does_not_change_past_features_or_train_tensors():
    """Modificar casos y clima del bloque futuro (val/test) no altera ni un bit de features ni train."""
    base_panel = _synthetic_toy_panel(n_weeks=80, missing_climate_idx=0)
    perturbed_panel = _synthetic_toy_panel(n_weeks=80, missing_climate_idx=0)

    # Con n_weeks=80, horizon=2, validation_weeks=12, test_weeks=12:
    # val_start cae en weeks[54]; perturbar weeks >= 54 solo toca val/test, nunca train.
    cutoff_week = base_panel.weeks[54]
    future_mask = perturbed_panel.panel["week_start"] >= cutoff_week
    perturbed_panel.panel.loc[future_mask, "confirmed_cases"] = 9999.0
    perturbed_panel.panel.loc[future_mask, "temperature_mean"] = 45.0
    perturbed_panel.panel.loc[future_mask, "precipitation_mm"] = 500.0

    f_base = build_features(base_panel, horizon=2)
    f_pert = build_features(perturbed_panel, horizon=2)
    cols = feature_columns(f_base)

    past_base = f_base[f_base["origin_week"] < cutoff_week].sort_values(["district_code", "origin_week"])
    past_pert = f_pert[f_pert["origin_week"] < cutoff_week].sort_values(["district_code", "origin_week"])
    np.testing.assert_allclose(
        past_base[cols].to_numpy(dtype=float),
        past_pert[cols].to_numpy(dtype=float),
        equal_nan=True,
    )

    # Tampoco cambia el scaler ni los tensores de train cuando val/test cambian.
    b_base = build_dataset(base_panel, horizon=2, validation_weeks=12, test_weeks=12)
    b_pert = build_dataset(perturbed_panel, horizon=2, validation_weeks=12, test_weeks=12)
    np.testing.assert_allclose(b_base.scaler_mean, b_pert.scaler_mean)
    np.testing.assert_allclose(b_base.scaler_std, b_pert.scaler_std)
    np.testing.assert_allclose(b_base.X_train, b_pert.X_train)
    np.testing.assert_allclose(b_base.seq_train, b_pert.seq_train)
    np.testing.assert_allclose(b_base.grid_train, b_pert.grid_train)


def test_rolling_windows_are_strictly_causal():
    """Verifica que las medias móviles de casos en t usen exclusivamente <= t-1 y clima <= t."""
    panel = _synthetic_toy_panel(n_weeks=40)
    frame = build_features(panel, horizon=1)
    tru = frame[frame["district_code"] == "PE-LL-TRU"].sort_values("origin_week").reset_index(drop=True)
    # En la fila idx=10, cases_mean_4 debe promediar exactamente confirmed_cases[6:10] (lags 1..4).
    expected_cases_mean_4 = float(tru.loc[6:9, "confirmed_cases"].mean())
    assert float(tru.loc[10, "cases_mean_4"]) == pytest.approx(expected_cases_mean_4)
    # temperature_mean_4 en idx=10 promedia [7:11] (<= t, estrictamente < t+h).
    expected_temp_mean_4 = float(tru.loc[7:10, "temperature_mean"].mean())
    assert float(tru.loc[10, "temperature_mean_4"]) == pytest.approx(expected_temp_mean_4)

    with pytest.raises(ValueError):
        build_features(panel, horizon=0)


def test_training_guards_and_path_resolution(tmp_path):
    """Protege data/models/, valida mapeo XGBoost y resolución de rutas legadas."""
    with pytest.raises(ValueError):
        resolve_safe_output_dir("data/models")

    safe = resolve_safe_output_dir(tmp_path / "phase2")
    assert safe.is_dir()

    assert resolve_xgb_n_estimators(epochs=1, n_estimators=None) == 1
    assert resolve_xgb_n_estimators(epochs=1, n_estimators=25) == 25
    assert resolve_xgb_n_estimators(epochs=None, n_estimators=None) is None

    legacy = r"c:\D\UNT\10 CICLO\Trabajo de Investigación\IA DENGUE\DengAI-UNT\data\models\xgboost-h1-optuna.joblib"
    resolved = resolve_artifact_path(legacy)
    assert resolved.exists(), f"No se resolvió ruta legada: {resolved}"
