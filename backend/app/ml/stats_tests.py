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


def diebold_mariano(
    actual: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    name_a: str,
    name_b: str,
    *,
    power: int = 2,
    blocks: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    y = np.asarray(actual, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[mask], pa[mask], pb[mask]
    if blocks is not None:
        block_ids = np.asarray(list(blocks))[mask]
        unique = list(dict.fromkeys(block_ids.tolist()))
        y = np.asarray([np.mean(y[block_ids == block]) for block in unique])
        pa = np.asarray([np.mean(pa[block_ids == block]) for block in unique])
        pb = np.asarray([np.mean(pb[block_ids == block]) for block in unique])
    n = len(y)
    if n < 20:
        return {
            "test_name": "Diebold-Mariano",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Se requieren al menos 20 observaciones (o bloques) independientes.",
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


def mcnemar_test(
    actual_binary: Sequence[int],
    pred_a_binary: Sequence[int],
    pred_b_binary: Sequence[int],
    name_a: str,
    name_b: str,
) -> Dict[str, object]:
    """McNemar para clasificación binaria de brote (pares discordantes).

    Se usa la prueba binomial exacta de dos colas sobre ``b`` y ``c``; es válida con
    pocos pares discordantes y no depende de la aproximación chi-cuadrado.
    """
    y = np.asarray(actual_binary, dtype=int)
    a = np.asarray(pred_a_binary, dtype=int)
    b = np.asarray(pred_b_binary, dtype=int)
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {
            "test_name": "McNemar",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Se requieren al menos 10 observaciones y ambas clases.",
        }
    correct_a = a == y
    correct_b = b == y
    b_count = int(np.sum(correct_a & ~correct_b))
    c_count = int(np.sum(~correct_a & correct_b))
    discordant = b_count + c_count
    if discordant == 0:
        return {
            "test_name": "McNemar",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "No hay pares discordantes.",
            "n_pairs": int(len(y)),
        }
    p_value = float(stats.binomtest(min(b_count, c_count), discordant, 0.5).pvalue)
    return {
        "test_name": "McNemar",
        "comparison": f"{name_a} vs {name_b}",
        "applicable": True,
        "statistic": float(min(b_count, c_count)),
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "b": b_count,
        "c": c_count,
        "n_pairs": int(len(y)),
        "interpretation": (
            f"{name_a} y {name_b} difieren en la clasificación de brote."
            if p_value < 0.05
            else "No se detectan diferencias significativas en la clasificación de brote."
        ),
    }


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    result = np.empty(n, dtype=float)
    result[order] = ranks
    return result


def _fast_delong(predictions: np.ndarray, n_positive: int):
    """Implementación de DeLong (Sun & Xu, 2014) para k clasificadores."""
    k, n = predictions.shape
    m = n_positive
    n_negative = n - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.empty((k, m))
    ty = np.empty((k, n_negative))
    tz = np.empty((k, n))
    for row in range(k):
        tx[row] = _compute_midrank(positive[row])
        ty[row] = _compute_midrank(negative[row])
        tz[row] = _compute_midrank(predictions[row])
    aucs = tz[:, :m].sum(axis=1) / (m * n_negative) - (m + 1.0) / (2.0 * n_negative)
    v01 = (tz[:, :m] - tx) / n_negative
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / m + sy / n_negative
    return aucs, covariance


def delong_roc_test(
    actual_binary: Sequence[int],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    name_a: str,
    name_b: str,
) -> Dict[str, object]:
    """Contraste de DeLong para dos AUC-ROC pareadas sobre las mismas observaciones."""
    y = np.asarray(actual_binary, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[mask], pa[mask], pb[mask]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) < 20 or positives < 5 or negatives < 5:
        return {
            "test_name": "DeLong",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Se requieren al menos 20 observaciones y 5 casos por clase.",
        }
    order = np.argsort(-y, kind="mergesort")
    y_sorted = y[order]
    predictions = np.vstack([pa[order], pb[order]])
    n_positive = int(y_sorted.sum())
    aucs, covariance = _fast_delong(predictions, n_positive)
    variance = float(covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    if variance <= 1e-12:
        return {
            "test_name": "DeLong",
            "comparison": f"{name_a} vs {name_b}",
            "applicable": False,
            "reason": "Varianza nula de la diferencia de AUC.",
            "auc_a": float(aucs[0]),
            "auc_b": float(aucs[1]),
        }
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "test_name": "DeLong",
        "comparison": f"{name_a} vs {name_b}",
        "applicable": True,
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "statistic": z,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "n_observations": int(len(y)),
        "interpretation": (
            f"Las AUC de {name_a} y {name_b} difieren significativamente."
            if p_value < 0.05
            else "No se detectan diferencias significativas entre las AUC."
        ),
    }


def _block_loss_matrix(
    predictions: Dict[str, np.ndarray],
    actual: np.ndarray,
    blocks: Sequence[str],
) -> tuple[np.ndarray, List[str], List[str]]:
    names = list(predictions)
    block_ids = np.asarray(list(blocks))
    unique_blocks = list(dict.fromkeys(blocks))
    rows = []
    for block in unique_blocks:
        mask = block_ids == block
        rows.append([float(np.mean(np.abs(actual[mask] - predictions[name][mask]))) for name in names])
    return np.asarray(rows, dtype=float), unique_blocks, names


def compare_models(
    predictions: Dict[str, np.ndarray],
    actual: np.ndarray,
    blocks: Sequence[str],
    *,
    binary_actual: Optional[Sequence[int]] = None,
    binary_predictions: Optional[Dict[str, np.ndarray]] = None,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    block_unit: str = "target_week",
) -> Dict[str, object]:
    """Ejecuta el protocolo estadístico sobre **bloques independientes**.

    Las pérdidas se agregan por bloque (p. ej. semana objetivo) antes de Friedman y
    Wilcoxon, evitando tratar como independientes las observaciones distrito-semana
    correlacionadas dentro de una misma semana. McNemar y DeLong usan las
    observaciones pareadas de clasificación cuando se proporcionan.
    """
    names = [n for n in predictions if n in predictions]
    if len(names) < 2:
        return {
            "friedman": {"applicable": False, "reason": "Se requieren al menos dos modelos."},
            "pairwise": [],
            "blocks": 0,
            "block_unit": "unknown",
        }
    matrix, unique_blocks, names = _block_loss_matrix(predictions, actual, blocks)
    friedman = friedman_test(matrix, names)

    pairwise = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise.append(wilcoxon_pairwise(matrix[:, i], matrix[:, j], names[i], names[j]))
    raw_p = [item["p_value"] for item in pairwise if item.get("applicable")]
    adjusted = holm_correction(raw_p) if raw_p else []
    cursor = 0
    for item in pairwise:
        if item.get("applicable"):
            item["p_value_adjusted"] = adjusted[cursor]
            item["significant_holm"] = bool(adjusted[cursor] < 0.05)
            cursor += 1

    mcnemar: List[Dict[str, object]] = []
    if binary_actual is not None and binary_predictions:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i] in binary_predictions and names[j] in binary_predictions:
                    mcnemar.append(
                        mcnemar_test(
                            binary_actual,
                            binary_predictions[names[i]],
                            binary_predictions[names[j]],
                            names[i],
                            names[j],
                        )
                    )

    delong: List[Dict[str, object]] = []
    if binary_actual is not None and probabilities:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if names[i] in probabilities and names[j] in probabilities:
                    delong.append(
                        delong_roc_test(
                            binary_actual,
                            probabilities[names[i]],
                            probabilities[names[j]],
                            names[i],
                            names[j],
                        )
                    )

    return {
        "friedman": friedman,
        "pairwise": pairwise,
        "mcnemar": mcnemar,
        "delong": delong,
        "blocks": len(unique_blocks),
        "block_unit": block_unit,
        "assumptions": (
            "Friedman/Wilcoxon sobre pérdidas agregadas por bloque independiente. "
            "McNemar y DeLong requieren clasificaciones/probabilidades pareadas."
        ),
    }
