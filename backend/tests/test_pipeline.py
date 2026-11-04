"""Pruebas del pipeline de datos: particiones sin fuga y modelos (M06/M07/M08)."""
from __future__ import annotations

import numpy as np
import pytest

from app.db.session import SessionLocal
from app.ml.features import build_dataset, build_features, load_panel
from app.ml.metrics import evaluate_all, mae, mase, winkler_score
from app.ml.models.base import ModelData
from app.ml.models.registry import create_model


@pytest.fixture(scope="module")
def panel():
    session = SessionLocal()
    try:
        yield load_panel(session)
    finally:
        session.close()


def test_panel_marks_synthetic_rows(panel):
    """El panel mixto conserva la marca por fila; el filtro sintético es homogéneo."""
    assert panel.source_summary["synthetic_rows"] > 0
    session = SessionLocal()
    try:
        synthetic = load_panel(session, only_synthetic=True)
    finally:
        session.close()
    assert synthetic.is_synthetic is True
    assert len(synthetic.districts) >= 10
    assert len(synthetic.weeks) > 100


def test_temporal_split_respects_chronology(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12)
    train_end = bundle.meta_train["origin_week"].max()
    val_start = bundle.meta_val["origin_week"].min()
    val_end = bundle.meta_val["origin_week"].max()
    test_start = bundle.meta_test["origin_week"].min()
    assert train_end < val_start <= val_end < test_start


def test_features_do_not_use_future_information(panel):
    """Ningún rezago puede ser negativo y el objetivo siempre está en t+h."""
    frame = build_features(panel, horizon=2)
    lag_columns = [column for column in frame.columns if column.endswith("_lag_1")]
    assert lag_columns
    # El objetivo de la fila t debe coincidir con los casos observados en t+2.
    sample = frame.dropna(subset=["target"]).head(20)
    for _, row in sample.iterrows():
        future = panel.panel[(panel.panel["district_code"] == row["district_code"]) & (panel.panel["week_start"] == row["target_week"])]
        if not future.empty:
            assert float(future["confirmed_cases"].iloc[0]) == float(row["target"])


def test_target_week_is_after_origin_week(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12)
    assert (bundle.meta_test["target_week"] > bundle.meta_test["origin_week"]).all()


def test_geographic_split_uses_unseen_districts(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12, strategy="geographic")
    train_districts = set(bundle.meta_train["district_code"])
    test_districts = set(bundle.meta_test["district_code"])
    assert train_districts.isdisjoint(test_districts)


def test_metrics_are_finite_and_scaled(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12)
    y = bundle.y_test
    pred = np.full_like(y, float(np.mean(bundle.y_train)))
    results = evaluate_all(y, pred, lower=pred * 0.8, upper=pred * 1.2, y_insample=bundle.y_train[-60:])
    assert results["MAE"] >= 0
    assert results["RMSE"] >= results["MAE"]
    assert mae(y, pred) == pytest.approx(results["MAE"])
    assert mase(y, pred, bundle.y_train) > 0
    assert winkler_score(y, pred * 0.8, pred * 1.2) > 0


def test_xgboost_trains_and_predicts(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12)
    model = create_model("xgboost")
    train = ModelData(X=bundle.X_train, seq=None, grid=None, cell=None, y=bundle.y_train, feature_names=bundle.feature_names)
    val = ModelData(X=bundle.X_val, seq=None, grid=None, cell=None, y=bundle.y_val, feature_names=bundle.feature_names)
    test = ModelData(X=bundle.X_test, seq=None, grid=None, cell=None, y=bundle.y_test, feature_names=bundle.feature_names)
    model.fit(train, val, params={"n_estimators": 60, "max_depth": 3})
    predictions = model.predict(test)
    assert len(predictions) == len(bundle.y_test)
    assert np.all(predictions >= 0)
    assert mae(bundle.y_test, predictions) < np.mean(bundle.y_test) + 5


def test_tensor_shapes_are_consistent(panel):
    bundle = build_dataset(panel, horizon=1, validation_weeks=12, test_weeks=12)
    assert bundle.seq_train is not None and bundle.grid_train is not None
    assert bundle.seq_train.shape[0] == bundle.X_train.shape[0]
    assert bundle.grid_train.shape[1] == 12
    assert bundle.cell_train.shape[0] == bundle.X_train.shape[0]
    assert bundle.grid_shape == (4, 5)
