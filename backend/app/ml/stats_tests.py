"""Pruebas estadísticas de contraste entre modelos (M08/F9).

* Friedman: diferencia global entre más de dos modelos sobre bloques independientes.
* Wilcoxon signed-rank pareado con corrección de Holm.
* Diebold-Mariano: comparación de precisión de pronósticos por pérdida cuadrática.
* Bootstrap: intervalos de confianza de las métricas.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def friedman_test(loss_matrix: np.ndarray, model_names: Sequence[str]) -> Dict[str, object]:
    """``loss_matrix`` debe tener forma (n_bloques, n_modelos)."""
    matrix = np.asarray(loss_matrix, dtype=float)
    n_blocks, n_models = matrix.shape
    if n_blocks < 3 or n_models < 3:
        return {"test_name": "Friedman", "applicable": False, "reason": "Se requieren al menos 3 bloques y 3 modelos."}
    statistic, p_value = stats.friedmanchisquare(*[matrix[:, i] for i in range(n_models)])
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix).mean(axis=0)
    return {
        "test_name": "Friedman",
        "applicable": True,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "mean_ranks": {name: round(float(rank), 4) for name, rank in zip(model_names, ranks)},
        "n_blocks": int(n_blocks),
        "n_models": int(n_models),
        "interpretation": (
            "Existen diferencias significativas entre modelos (alpha=0.05)."
            if p_value < 0.05
            else "No se detectan diferencias significativas entre los modelos comparados."
        ),
    }


def holm_correction(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    m = len(values)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return [float(v) for v in adjusted]


def wilcoxon_pairwise(loss_a: Sequence[float], loss_b: Sequence[float], name_a: str, name_b: str) -> Dict[str, object]:
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 10 or np.allclose(a, b):
        return {
            "test_name": "Wilcoxon",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Muestras insuficientes o pérdidas idénticas.",
        }
    statistic, p_value = stats.wilcoxon(a, b)
    diff = b - a
    effect = float(np.median(diff) / (np.std(diff) + 1e-9))
    return {
        "test_name": "Wilcoxon",
        "comparison": f"{name_a} vs {name_b}",
        "applicable": True,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": effect,
        "n_pairs": int(len(a)),
        "median_loss_diff": float(np.median(diff)),
    }


def diebold_mariano(actual: Sequence[float], pred_a: Sequence[float], pred_b: Sequence[float], name_a: str, name_b: str, *, power: int = 2) -> Dict[str, object]:
    y = np.asarray(actual, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[mask], pa[mask], pb[mask]
    n = len(y)
    if n < 20:
        return {
            "test_name": "Diebold-Mariano",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Se requieren al menos 20 observaciones.",
        }
    d = np.abs(y - pa) ** power - np.abs(y - pb) ** power
    mean_d = float(np.mean(d))
    # Varianza con corrección de Newey-West simple (rezago 1) para autocorrelación.
    gamma0 = float(np.var(d, ddof=1))
    gamma1 = float(np.cov(d[:-1], d[1:])[0, 1]) if n > 2 else 0.0
    var = (gamma0 + 2 * gamma1) / n
    if var <= 0:
        return {
            "test_name": "Diebold-Mariano",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Varianza nula de la serie de diferencias.",
        }
    statistic = mean_d / np.sqrt(var)
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    better = name_a if mean_d < 0 else name_b
    return {
        "test_name": "Diebold-Mariano",
        "comparison": f"{name_a} vs {name_b}",
        "applicable": True,
        "statistic": float(statistic),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "mean_loss_diff": mean_d,
        "better_model": better,
        "n_observations": int(n),
    }


def bootstrap_metric_ci(values: Sequence[float], *, statistic=np.mean, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return {"value": float(statistic(data)) if len(data) else float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    rng = np.random.default_rng(seed)
    samples = [float(statistic(rng.choice(data, len(data), replace=True))) for _ in range(n_boot)]
    return {
        "value": float(statistic(data)),
        "ci_lower": float(np.quantile(samples, alpha / 2)),
        "ci_upper": float(np.quantile(samples, 1 - alpha / 2)),
    }


def compare_models(
    predictions: Dict[str, np.ndarray],
    actual: np.ndarray,
    blocks: Sequence[str],
) -> Dict[str, object]:
    """Construye la matriz de pérdidas por bloque y ejecuta las pruebas del protocolo."""
    names = [n for n in predictions if n in predictions]
    if len(names) < 2:
        return {"friedman": {"applicable": False, "reason": "Se requieren al menos dos modelos."}, "pairwise": []}
    block_ids = np.array(list(blocks))
    unique_blocks = list(dict.fromkeys(blocks))
    loss_rows = []
    for block in unique_blocks:
        mask = block_ids == block
        row = [float(np.mean(np.abs(actual[mask] - predictions[name][mask]))) for name in names]
        loss_rows.append(row)
    matrix = np.array(loss_rows, dtype=float)
    friedman = friedman_test(matrix, names)

    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            loss_a = np.abs(actual - predictions[names[i]])
            loss_b = np.abs(actual - predictions[names[j]])
            pairwise.append(wilcoxon_pairwise(loss_a, loss_b, names[i], names[j]))
    raw_p = [item["p_value"] for item in pairwise if item.get("applicable")]
    adjusted = holm_correction(raw_p) if raw_p else []
    cursor = 0
    for item in pairwise:
        if item.get("applicable"):
            item["p_value_adjusted"] = adjusted[cursor]
            item["significant_holm"] = bool(adjusted[cursor] < 0.05)
            cursor += 1
    return {"friedman": friedman, "pairwise": pairwise, "blocks": len(unique_blocks)}
