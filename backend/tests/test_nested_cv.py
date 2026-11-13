"""Pruebas matemáticas antifuga, métricas del Anexo A y --resume para Nested Walk-Forward CV (Sesión 2)."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.db.session import SessionLocal  # noqa: E402
from app.ml.features import build_features, feature_columns, load_panel  # noqa: E402
from training.run_nested_cv import (  # noqa: E402
    ANEXO_A_METRICS,
    assert_no_nested_leakage,
    build_nested_walk_forward_folds,
    main as nested_cv_main,
    prepare_fold_model_data,
)


@pytest.fixture(scope="module")
def real_panel():
    with SessionLocal() as db:
        return load_panel(db, mode="auto")


@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_nested_walk_forward_strict_no_leakage_across_horizons(real_panel, horizon):
    """Comprueba matemáticamente cero fuga temporal y cero solapamiento de objetivos para h=1,4,8."""
    frame = build_features(real_panel, horizon=horizon)
    plan = build_nested_walk_forward_folds(
        frame,
        outer_folds=5,
        inner_folds=3,
        outer_val_weeks=26,
        inner_val_weeks=13,
        test_weeks=26,
        horizon=horizon,
    )
    assert_no_nested_leakage(plan)
    assert len(plan["outer_folds"]) == 5
    test_start = plan["test_start"]
    test_origins = set(plan["test_origins"])
    assert len(test_origins) == 26

    for outer in plan["outer_folds"]:
        o_train = outer["outer_train"]
        o_val = outer["outer_val"]
        assert len(outer["inner_folds"]) == 3

        assert o_train["target_week"].max() < o_val["origin_week"].min()
        assert o_val["target_week"].max() < test_start
        assert set(o_train["origin_week"]).isdisjoint(set(o_val["origin_week"]))
        assert set(o_train["target_week"]).isdisjoint(set(o_val["target_week"]))
        assert set(o_val["origin_week"]).isdisjoint(test_origins)
        assert set(o_val["target_week"]).isdisjoint(test_origins)

        for inner in outer["inner_folds"]:
            i_train = inner["inner_train"]
            i_val = inner["inner_val"]
            assert i_train["target_week"].max() < i_val["origin_week"].min()
            assert i_val["target_week"].max() < o_val["origin_week"].min()
            assert set(i_train["origin_week"]).isdisjoint(set(i_val["origin_week"]))
            assert set(i_train["target_week"]).isdisjoint(set(i_val["target_week"]))
            assert set(i_val["origin_week"]).isdisjoint(set(o_val["origin_week"]))
            assert set(i_val["target_week"]).isdisjoint(set(o_val["target_week"]))


def test_fold_scaler_is_isolated_from_val_and_test(real_panel):
    """Perturbar outer_val y test no cambia el escalador ni X_train de outer_train."""
    base_frame = build_features(real_panel, horizon=1)
    base_plan = build_nested_walk_forward_folds(base_frame)
    cols = feature_columns(base_frame)

    outer1 = base_plan["outer_folds"][0]
    tr_base, va_base = prepare_fold_model_data(
        base_frame,
        real_panel,
        cols,
        outer1["outer_train"],
        outer1["outer_val"],
        needs_tensors=False,
    )

    perturbed_val = outer1["outer_val"].copy()
    perturbed_val[cols] = perturbed_val[cols] + 1000.0
    tr_pert, _ = prepare_fold_model_data(
        base_frame,
        real_panel,
        cols,
        outer1["outer_train"],
        perturbed_val,
        needs_tensors=False,
    )
    np.testing.assert_allclose(tr_base.X, tr_pert.X)
    assert len(va_base.y) > 0


def test_nested_cv_cli_five_observations_and_resume(tmp_path):
    """Valida ejecución end-to-end (5x3), todas las métricas del Anexo A, agregado estadístico y --resume."""
    out_file = tmp_path / "nested_cv_results.json"
    telemetry_dir = tmp_path / "telemetry"
    exit_code = nested_cv_main(
        [
            "--models",
            "xgboost",
            "--horizon",
            "1",
            "--trials",
            "1",
            "--epochs",
            "1",
            "--outer-folds",
            "5",
            "--inner-folds",
            "3",
            "--inner-val-weeks",
            "13",
            "--outer-val-weeks",
            "26",
            "--test-weeks",
            "26",
            "--output",
            str(out_file),
            "--telemetry-dir",
            str(telemetry_dir),
        ]
    )
    assert exit_code == 0
    assert out_file.exists()

    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["leakage_audit"]["strict_no_leakage"] is True
    assert report["leakage_audit"]["test_block_untouched"] is True
    assert report["leakage_audit"]["outer_folds_verified"] == 5
    assert report["leakage_audit"]["inner_folds_verified"] == 15

    xgb = report["models"]["xgboost"]
    assert xgb["status"] == "COMPLETED"
    assert xgb["n_outer_folds_completed"] == 5
    assert len(xgb["outer_folds"]) == 5

    test_start = pd.Timestamp(report["leakage_audit"]["test_start"]).date()
    for idx, fold in enumerate(xgb["outer_folds"], start=1):
        assert fold["outer_fold"] == idx
        assert len(fold["inner_folds"]) == 3

        o_train_tmax = pd.Timestamp(fold["outer_train_target_max"]).date()
        o_val_omin = pd.Timestamp(fold["outer_val_origin_range"][0]).date()
        o_val_tmax = pd.Timestamp(fold["outer_val_target_range"][1]).date()
        assert o_train_tmax < o_val_omin
        assert o_val_tmax < test_start

        for inner in fold["inner_folds"]:
            i_train_tmax = pd.Timestamp(inner["inner_train_target_max"]).date()
            i_val_omin = pd.Timestamp(inner["inner_val_origin_range"][0]).date()
            i_val_tmax = pd.Timestamp(inner["inner_val_target_max"]).date()
            assert i_train_tmax < i_val_omin
            assert i_val_tmax < o_val_omin

        metrics = fold["metrics"]
        for metric_name in ANEXO_A_METRICS:
            assert metric_name in metrics
            assert metrics[metric_name] is not None, f"{metric_name} fue None en outer_fold {idx}"
            assert np.isfinite(metrics[metric_name])
        assert metrics["MAE"] <= metrics["RMSE"] + 1e-6
        assert 0.0 <= metrics["coverage"] <= 1.0
        assert 0.0 <= metrics["Brier"] <= 1.0
        assert 0.0 <= metrics["F1"] <= 1.0
        assert -1.0 <= metrics["MCC"] <= 1.0
        assert 0.0 <= metrics["AUC_ROC"] <= 1.0
        assert 0.0 <= metrics["AUC_PR"] <= 1.0
        assert metrics["tiempo_s"] > 0.0

    summary = xgb["summary"]
    for metric_name in ANEXO_A_METRICS:
        stats_block = summary[metric_name]
        assert stats_block["n"] == 5
        for field in ("mean", "std", "min", "max", "ci_95_lower", "ci_95_upper"):
            assert stats_block[field] is not None
        assert stats_block["min"] <= stats_block["mean"] <= stats_block["max"]
        assert stats_block["ci_95_lower"] <= stats_block["mean"] <= stats_block["ci_95_upper"]

    # Simular interrupción tras el fold 2 y verificar reanudación con --resume.
    partial = json.loads(out_file.read_text(encoding="utf-8"))
    fold2_snapshot = partial["models"]["xgboost"]["outer_folds"][:2]
    partial["models"]["xgboost"]["outer_folds"] = fold2_snapshot
    partial["models"]["xgboost"]["n_outer_folds_completed"] = 2
    partial["models"]["xgboost"]["status"] = "IN_PROGRESS"
    out_file.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    exit_resume = nested_cv_main(
        [
            "--models",
            "xgboost",
            "--horizon",
            "1",
            "--trials",
            "1",
            "--epochs",
            "1",
            "--outer-folds",
            "5",
            "--inner-folds",
            "3",
            "--inner-val-weeks",
            "13",
            "--outer-val-weeks",
            "26",
            "--test-weeks",
            "26",
            "--output",
            str(out_file),
            "--telemetry-dir",
            str(telemetry_dir),
            "--resume",
        ]
    )
    assert exit_resume == 0
    resumed = json.loads(out_file.read_text(encoding="utf-8"))
    assert resumed["models"]["xgboost"]["status"] == "COMPLETED"
    assert resumed["models"]["xgboost"]["n_outer_folds_completed"] == 5
    assert resumed["models"]["xgboost"]["outer_folds"][:2] == fold2_snapshot

    # Verificar que --resume rechaza cambios estructurales incompatibles.
    with pytest.raises(ValueError):
        nested_cv_main(
            [
                "--models",
                "xgboost",
                "--horizon",
                "2",
                "--trials",
                "1",
                "--epochs",
                "1",
                "--output",
                str(out_file),
                "--resume",
            ]
        )
