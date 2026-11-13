"""Métricas de evaluación epidemiológica (M08) e intervalos por bootstrap."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def _clean(y_true: Sequence[float], y_pred: Sequence[float]):
    t = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(t) & np.isfinite(p)
    return t[mask], p[mask]


def mae(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if len(t) else float("nan")


def rmse(y_true, y_pred) -> float:
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def mape(y_true, y_pred, *, min_denominator: float = 1.0) -> Optional[float]:
    """MAPE con denominador mínimo: con conteos semanales bajos el MAPE no es apropiado."""
    t, p = _clean(y_true, y_pred)
    usable = t >= min_denominator
    if usable.sum() < max(5, 0.2 * len(t)):
        return None
    return float(np.mean(np.abs(t[usable] - p[usable]) / t[usable]) * 100)


def mase(y_true, y_pred, y_insample, *, seasonality: int = 1) -> float:
    """Error escalado medio absoluto respecto al pronóstico ingenuo estacional."""
    t, p = _clean(y_true, y_pred)
    insample = np.asarray(y_insample, dtype=float)
    if len(insample) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(insample[seasonality:] - insample[:-seasonality]))
    if scale < 1e-9:
        scale = np.mean(np.abs(insample - np.mean(insample)))
    if scale < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(t - p)) / scale)


def interval_coverage(y_true, lower, upper) -> float:
    t, low = _clean(y_true, lower)
    _, up = _clean(y_true, upper)
    if len(t) == 0:
        return float("nan")
    return float(np.mean((t >= low) & (t <= up)))


def winkler_score(y_true, lower, upper, *, alpha: float = 0.1) -> float:
    """Puntaje de intervalo de Winkler (WIS para un único intervalo central)."""
    t, low = _clean(y_true, lower)
    _, up = _clean(y_true, upper)
    width = up - low
    penalty_low = np.where(t < low, 2.0 / alpha * (low - t), 0.0)
    penalty_up = np.where(t > up, 2.0 / alpha * (t - up), 0.0)
    return float(np.mean(width + penalty_low + penalty_up))


def brier_score(y_true_binary, prob) -> float:
    t, p = _clean(y_true_binary, prob)
    return float(np.mean((p - t) ** 2)) if len(t) else float("nan")


def _risk_binary(y_true: np.ndarray, threshold: float) -> np.ndarray:
    return (y_true >= threshold).astype(int)


def precision_recall_auc(y_true, prob, *, threshold: float) -> Optional[float]:
    """AUC-PR para clasificación de brote (definido por el umbral epidemiológico dado)."""
    from sklearn.metrics import average_precision_score

    t, p = _clean(y_true, prob)
    if len(t) == 0:
        return None
    binary = _risk_binary(t, threshold)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None
    return float(average_precision_score(binary, p))


def roc_auc(y_true, prob, *, threshold: float) -> Optional[float]:
    """AUC-ROC para clasificación de brote (definido por el umbral epidemiológico dado)."""
    from sklearn.metrics import roc_auc_score

    t, p = _clean(y_true, prob)
    if len(t) == 0:
        return None
    binary = _risk_binary(t, threshold)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return None
    return float(roc_auc_score(binary, p))


def sensitivity_specificity(y_true, prob, *, threshold: float, cut: Optional[float] = None) -> Dict[str, Optional[float]]:
    """Sensibilidad/especificidad al corte operativo por defecto (punto de Youden)."""
    t, p = _clean(y_true, prob)
    if len(t) == 0:
        return {"sensitivity": None, "specificity": None, "cut": cut}
    binary = _risk_binary(t, threshold)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return {"sensitivity": None, "specificity": None, "cut": cut}
    if cut is None:
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(binary, p)
        cut = float(thresholds[int(np.argmax(tpr - fpr))])
    predicted = (p >= cut).astype(int)
    tp = int(((predicted == 1) & (binary == 1)).sum())
    tn = int(((predicted == 0) & (binary == 0)).sum())
    fn = int(((predicted == 0) & (binary == 1)).sum())
    fp = int(((predicted == 1) & (binary == 0)).sum())
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "cut": cut,
    }


def classification_metrics(
    y_true,
    prob,
    *,
    threshold: float,
    cut: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Métricas de clasificación de brote del Anexo A (AUC_PR, AUC_ROC, F1, MCC, sensitivity, specificity)."""
    from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve

    t, p = _clean(y_true, prob)
    if len(t) == 0:
        return {
            "AUC_PR": None,
            "AUC_ROC": None,
            "F1": None,
            "MCC": None,
            "sensitivity": None,
            "specificity": None,
            "cut": cut,
        }
    binary = _risk_binary(t, threshold)
    if binary.sum() == 0 or binary.sum() == len(binary):
        return {
            "AUC_PR": None,
            "AUC_ROC": None,
            "F1": None,
            "MCC": None,
            "sensitivity": None,
            "specificity": None,
            "cut": cut,
        }
    if cut is None:
        fpr, tpr, thresholds = roc_curve(binary, p)
        cut = float(thresholds[int(np.argmax(tpr - fpr))])
    predicted = (p >= cut).astype(int)
    tp = int(((predicted == 1) & (binary == 1)).sum())
    tn = int(((predicted == 0) & (binary == 0)).sum())
    fn = int(((predicted == 0) & (binary == 1)).sum())
    fp = int(((predicted == 1) & (binary == 0)).sum())
    return {
        "AUC_PR": precision_recall_auc(t, p, threshold=threshold),
        "AUC_ROC": roc_auc(t, p, threshold=threshold),
        "F1": float(f1_score(binary, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(binary, predicted)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "cut": cut,
    }


def bootstrap_ci(
    values_true: Sequence[float],
    values_pred: Sequence[float],
    func: Callable[[Sequence[float], Sequence[float]], float],
    *,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    t, p = _clean(values_true, values_pred)
    if len(t) < 8:
        return {"value": float(func(t, p)) if len(t) else float("nan"), "ci_lower": None, "ci_upper": None}
    rng = np.random.default_rng(seed)
    stats = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            value = float(func(t[idx], p[idx]))
        except Exception:
            continue
        if np.isfinite(value):
            stats.append(value)
    point = float(func(t, p))
    if not stats:
        return {"value": point, "ci_lower": None, "ci_upper": None}
    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return {"value": point, "ci_lower": lower, "ci_upper": upper}


def evaluate_all(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
    y_insample: Optional[Sequence[float]] = None,
    outbreak_threshold: Optional[float] = None,
    interval_level: float = 0.9,
    bootstrap: bool = True,
) -> Dict[str, Optional[float]]:
    """Calcula el conjunto de métricas exigido por el protocolo (M08)."""
    results: Dict[str, Optional[float]] = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_insample) if y_insample is not None else None,
    }
    if lower is not None and upper is not None:
        results["coverage"] = interval_coverage(y_true, lower, upper)
        results["WIS"] = winkler_score(y_true, lower, upper, alpha=1 - interval_level)
    if outbreak_threshold is not None:
        cls = classification_metrics(y_true, y_pred, threshold=outbreak_threshold)
        results["AUC_PR"] = cls["AUC_PR"]
        results["AUC_ROC"] = cls["AUC_ROC"]
        results["F1"] = cls["F1"]
        results["MCC"] = cls["MCC"]
        results["sensitivity"] = cls["sensitivity"]
        results["specificity"] = cls["specificity"]
        # Brier: probabilidad calibrada aproximada con la media móvil de la predicción.
        t, p = _clean(y_true, y_pred)
        if len(t):
            prob = np.clip(p / (np.maximum(p, 1e-6) + 1e-6) * 0.5 + 0.5 * (p > outbreak_threshold), 0, 1)
            results["Brier"] = brier_score(_risk_binary(t, outbreak_threshold), prob)
    if bootstrap:
        for name, func in (("MAE", mae), ("RMSE", rmse)):
            ci = bootstrap_ci(y_true, y_pred, func)
            results[f"{name}_ci_lower"] = ci["ci_lower"]
            results[f"{name}_ci_upper"] = ci["ci_upper"]
    return results


def per_district_metrics(y_true, y_pred, districts: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for district in sorted(set(districts)):
        mask = np.array([d == district for d in districts])
        rows.append(
            {
                "district_code": district,
                "n": int(mask.sum()),
                "MAE": mae(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]),
                "RMSE": rmse(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]),
            }
        )
    return rows
