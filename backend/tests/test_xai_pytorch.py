"""Pruebas XAI para XGBoost, los 4 modelos PyTorch, LIME, rezagos y atención real (Sesión 3)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.db.session import SessionLocal  # noqa: E402
from app.ml.explain import (  # noqa: E402
    TabularModelAdapter,
    _permutation_contributions,
    _shap_values,
    aggregate_lag_importance,
    explain_sample,
    lime_explanation,
    resolve_district_code,
    temporal_importance,
)
from app.ml.features import build_features, load_panel, make_sample_tensors, split_temporal, to_matrix  # noqa: E402
from app.ml.models.base import ModelData  # noqa: E402
from app.ml.trainer import load_bundle  # noqa: E402
from app.core.config import settings  # noqa: E402


@pytest.fixture(scope="module")
def xai_context():
    with SessionLocal() as db:
        panel = load_panel(db, mode="auto")
    frame = build_features(panel, horizon=1)
    train_raw, _, _, _ = split_temporal(
        frame,
        validation_weeks=settings.ML_VALIDATION_WEEKS,
        test_weeks=settings.ML_TEST_WEEKS,
    )
    sub = frame[(frame["district_code"] == "PE-LL-TRU") & (frame["epi_year"] == 2024) & (frame["epi_week"] == 15)]
    if sub.empty:
        latest = frame["origin_week"].max()
        sub = frame[(frame["district_code"] == "PE-LL-TRU") & (frame["origin_week"] == latest)]
    return panel, frame, train_raw, sub.head(1)


def test_resolve_district_code_and_ubigeo():
    assert resolve_district_code("PE-LL-TRU") == "PE-LL-TRU"
    assert resolve_district_code("pe-ll-asc") == "PE-LL-ASC"
    assert resolve_district_code("130101") == "PE-LL-TRU"
    assert resolve_district_code("130201") == "PE-LL-ASC"
    assert resolve_district_code("131001") == "PE-LL-SDC"


@pytest.mark.parametrize(
    "artifact_name",
    [
        "xgboost-h1-optuna.joblib",
        "lstm-h1-optuna.joblib",
        "convlstm-h1-optuna.joblib",
        "cnn_lstm-h1-optuna.joblib",
        "convlstm_attention-h1-optuna.joblib",
    ],
)
def test_shap_permutation_temporal_and_lime_across_all_models(xai_context, artifact_name):
    artifact = PROJECT_DIR / "data" / "models" / artifact_name
    if not artifact.exists():
        pytest.skip(f"Artefacto no disponible en disco: {artifact}")

    panel, frame, train_raw, subset = xai_context
    bundle = load_bundle(str(artifact))
    model = bundle["model"]
    columns = list(bundle["feature_names"])
    mean = np.asarray(bundle["scaler_mean"], dtype=float)
    std = np.asarray(bundle["scaler_std"], dtype=float)
    window = int(bundle.get("sequence_window", 12))

    seq, grid, cell, _ = make_sample_tensors(panel, frame, subset, columns, mean, std, window=window)
    X = to_matrix(subset, columns, mean, std)
    background = to_matrix(train_raw.sample(min(120, len(train_raw)), random_state=42), columns, mean, std)

    adapter = TabularModelAdapter(
        model,
        columns,
        X,
        base_seq=seq,
        base_grid=grid,
        base_cell=cell,
        grid_shape=tuple(bundle.get("grid_shape", (4, 5))),
    )

    shap_vals = _shap_values(model, X, background, adapter=adapter, nsamples=40, seed=42, raise_on_error=True)
    assert shap_vals is not None
    assert shap_vals.shape == X.shape
    assert np.all(np.isfinite(shap_vals))

    perm_vals = _permutation_contributions(model, X, None, columns, adapter=adapter)
    assert perm_vals.shape == (len(columns),)
    assert np.all(np.isfinite(perm_vals))

    if getattr(model, "requires", "tabular") in ("sequence", "grid"):
        t_imp = temporal_importance(model, seq=seq, grid=grid, cell=cell, window=window)
        assert len(t_imp) == window
        assert all("delta" in item and "step" in item for item in t_imp)

    lime_res = lime_explanation(adapter, X, background, columns, num_samples=150, top_k=6, random_state=42)
    assert lime_res["method"].startswith("LIME")
    assert len(lime_res["factors"]) == 6
    assert np.all(np.isfinite(lime_res["contributions"]))


def test_convlstm_attention_exposes_real_temporal_and_spatial_weights(xai_context):
    artifact = PROJECT_DIR / "data" / "models" / "convlstm_attention-h1-optuna.joblib"
    if not artifact.exists():
        pytest.skip("Modelo convlstm_attention-h1-optuna.joblib no encontrado.")

    panel, frame, train_raw, subset = xai_context
    bundle = load_bundle(str(artifact))
    model = bundle["model"]
    columns = list(bundle["feature_names"])
    mean = np.asarray(bundle["scaler_mean"], dtype=float)
    std = np.asarray(bundle["scaler_std"], dtype=float)
    window = int(bundle.get("sequence_window", 12))

    seq, grid, cell, _ = make_sample_tensors(panel, frame, subset, columns, mean, std, window=window)
    X = to_matrix(subset, columns, mean, std)
    data = ModelData(X=X, seq=seq, grid=grid, cell=cell, y=subset["target"].to_numpy(dtype=float), feature_names=columns)

    pred_plain = model.predict(data)
    pred_attn, attn = model.predict_with_attention(data)
    np.testing.assert_allclose(pred_plain, pred_attn, rtol=1e-5, atol=1e-5)

    assert attn["temporal_weights"].shape == (1, window)
    assert attn["temporal_head_weights"].ndim == 3 and attn["temporal_head_weights"].shape[:2] == (1, window)
    assert attn["spatial_weights"].shape == (1, 4, 5)
    assert float(np.sum(attn["temporal_weights"][0])) == pytest.approx(1.0, abs=1e-4)
    assert float(np.sum(attn["spatial_weights"][0])) == pytest.approx(1.0, abs=1e-4)

    result = explain_sample(
        model=model,
        panel=panel,
        frame=frame,
        subset=subset,
        background_frame=train_raw,
        columns=columns,
        mean=mean,
        std=std,
        window=window,
        district_grid=bundle.get("district_grid"),
        grid_shape=tuple(bundle.get("grid_shape", (4, 5))),
        top_k=8,
        shap_samples=40,
        lime_samples=150,
        include_lime=True,
        raise_on_error=True,
    )
    assert result["attention"] is not None
    assert result["attention"]["n_occupied_cells"] in (13, 14)
    assert len(result["lag_importance"]["lags"]) == 12
    assert result["lag_importance"]["lags"][-1]["cumulative_share_pct"] in (0.0, 100.0) or (
        result["lag_importance"]["lags"][-1]["cumulative_share_pct"] == pytest.approx(100.0, abs=0.1)
    )
    for unused_col in ("area_km2", "solar_radiation", "wind_speed"):
        if unused_col in columns:
            idx = columns.index(unused_col)
            assert float(result["shap_contributions"][idx]) == 0.0
            assert float(result["lime"]["contributions"][idx]) == 0.0


def test_aggregate_lag_importance_cumulative():
    cols = [
        "cases_lag_1",
        "cases_lag_2",
        "temperature_mean_lag_1",
        "humidity_mean_lag_2",
        "precipitation_mm_lag_12",
    ]
    contribs = np.array([2.0, -1.0, 0.5, -0.5, 1.0], dtype=float)
    agg = aggregate_lag_importance(cols, contribs, max_lag=12)
    assert len(agg["lags"]) == 12
    assert agg["summary"]["total_cases_lags_1_12"] == pytest.approx(3.0)
    assert agg["summary"]["total_climate_lags_1_12"] == pytest.approx(2.0)
    assert agg["lags"][-1]["cumulative_total_abs"] == pytest.approx(5.0)
    assert agg["lags"][-1]["cumulative_share_pct"] == pytest.approx(100.0)
