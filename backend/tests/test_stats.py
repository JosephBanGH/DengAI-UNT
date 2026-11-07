"""Pruebas del protocolo estadístico: Friedman por bloques, McNemar y DeLong."""
from __future__ import annotations

import numpy as np

from app.ml.stats_tests import compare_models, delong_roc_test, mcnemar_test


def test_friedman_uses_independent_blocks_not_individual_observations():
    rng = np.random.default_rng(7)
    n_blocks, districts = 24, 6
    actual = rng.poisson(12, size=n_blocks * districts).astype(float)
    blocks = [f"2024-W{week:02d}" for week in range(1, n_blocks + 1) for _ in range(districts)]
    good = actual + rng.normal(0, 1.0, size=len(actual))
    medium = actual + rng.normal(0, 3.0, size=len(actual))
    bad = actual + rng.normal(0, 6.0, size=len(actual))
    result = compare_models({"good": good, "medium": medium, "bad": bad}, actual, blocks)
    assert result["block_unit"] == "target_week"
    assert result["blocks"] == n_blocks
    assert result["friedman"]["applicable"] is True
    assert result["friedman"]["n_blocks"] == n_blocks
    assert len(result["pairwise"]) == 3
    assert all("p_value_adjusted" in item for item in result["pairwise"])


def test_mcnemar_detects_discordant_classifications():
    actual = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    perfect = actual.copy()
    inverted = 1 - actual
    result = mcnemar_test(actual, perfect, inverted, "perfect", "inverted")
    assert result["applicable"] is True
    assert result["p_value"] < 0.05
    assert result["significant"] is True


def test_delong_compares_paired_auc():
    rng = np.random.default_rng(11)
    actual = np.array([1] * 40 + [0] * 60)
    strong = np.clip(actual * 0.6 + rng.normal(0, 0.25, size=len(actual)), 0, 1)
    weak = np.clip(actual * 0.2 + rng.normal(0, 0.45, size=len(actual)), 0, 1)
    result = delong_roc_test(actual, strong, weak, "strong", "weak")
    assert result["applicable"] is True
    assert result["auc_a"] > result["auc_b"]
    assert 0 <= result["p_value"] <= 1
