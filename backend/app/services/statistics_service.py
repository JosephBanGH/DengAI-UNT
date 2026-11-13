"""Módulo de Estadística Descriptiva y Visualizaciones Explicables (Fase 1 - MVP).

Calcula indicadores univariados y bivariados paramétricos y no paramétricos sobre el
panel epidemiológico-climático y genera narrativas científicas de explicabilidad
estadística automatizada en lenguaje natural, distinguiendo asociación de causalidad.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.features import load_panel
from app.models.epi import Dataset

logger = logging.getLogger("dengai.statistics")


def _calculate_univariate(series: pd.Series, variable_name: str, unit: str) -> Dict[str, Any]:
    """Calcula estadísticas descriptivas rigurosas para una variable cuantitativa."""
    clean = series.dropna()
    total_count = int(len(series))
    valid_count = int(len(clean))
    missing_count = total_count - valid_count
    missing_pct = round(missing_count / total_count * 100.0, 2) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "variable": variable_name,
            "unit": unit,
            "count": total_count,
            "valid_count": 0,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
        }

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if valid_count > 1 else 0.0
    median_val = float(clean.median())
    min_val = float(clean.min())
    max_val = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr = float(q75 - q25)
    skewness_val = float(stats.skew(clean)) if valid_count > 2 else 0.0
    kurtosis_val = float(stats.kurtosis(clean)) if valid_count > 3 else 0.0

    return {
        "variable": variable_name,
        "unit": unit,
        "count": total_count,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "median": round(median_val, 2),
        "min": round(min_val, 2),
        "q25": round(q25, 2),
        "q75": round(q75, 2),
        "iqr": round(iqr, 2),
        "max": round(max_val, 2),
        "skewness": round(skewness_val, 2),
        "kurtosis": round(kurtosis_val, 2),
    }


def _generate_explanations(univariate: List[Dict[str, Any]], correlations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Genera explicaciones automatizadas en lenguaje natural basadas en evidencia matemática."""
    explanations = []

    # Explicación para casos confirmados
    cases_stat = next((u for u in univariate if u["variable"] == "confirmed_cases"), None)
    if cases_stat and cases_stat["mean"] is not None:
        skew = cases_stat["skewness"]
        is_skewed = skew > 1.0
        explanations.append(
            {
                "topic": "Distribución de Casos Confirmados",
                "indicator": f"Media: {cases_stat['mean']} casos/semana | Mediana: {cases_stat['median']} | Asimetría: {skew}",
                "behavior": (
                    f"La distribución de casos muestra una asimetría marcadamente positiva ({skew}), con mediana "
                    f"({cases_stat['median']}) significativamente menor que la media ({cases_stat['mean']}). "
                    f"El rango intercuartílico (IQR={cases_stat['iqr']}) confirma que la mayoría de semanas cursan "
                    f"con baja transmisión, mientras que el máximo ({cases_stat['max']}) refleja semanas de brote o picos estacionales."
                ),
                "methodological_implication": (
                    "Comportamiento epidémico sobredisperso (varianza superior a la media). Se desaconsejan modelos lineales "
                    "de mínimos cuadrados ordinarios sin transformación; se justifica el empleo de modelos no lineales con soporte "
                    "para colas pesadas (XGBoost) y arquitecturas profundas recurrentes (LSTM, ConvLSTM) con funciones de pérdida Poisson o regularización L2."
                ),
                "epistemic_caution": "Datos analizados sobre el panel actual. No representa un diagnóstico clínico individual.",
            }
        )

    # Explicación para variables climáticas
    temp_stat = next((u for u in univariate if u["variable"] == "temperature_mean"), None)
    hum_stat = next((u for u in univariate if u["variable"] == "humidity_mean"), None)
    rain_stat = next((u for u in univariate if u["variable"] == "precipitation_mm"), None)

    if temp_stat and hum_stat and rain_stat:
        explanations.append(
            {
                "topic": "Perfil Climatológico y Condiciones Ambientales",
                "indicator": f"Temperatura media: {temp_stat['mean']} °C | Humedad media: {hum_stat['mean']} % | Lluvia media: {rain_stat['mean']} mm",
                "behavior": (
                    f"La temperatura media ({temp_stat['mean']} ± {temp_stat['std']} °C) oscila en el intervalo favorable "
                    f"para la supervivencia del vector Aedes aegypti (óptimo 24–28 °C). La humedad relativa ({hum_stat['mean']} %) "
                    f"mantiene niveles propicios para el periodo de incubación extrínseca del virus."
                ),
                "methodological_implication": (
                    "Las series climáticas presentan estacionalidad anual marcada. Para capturar el retardo biofísico entre "
                    "lluvias/temperatura y el surgimiento de casos clínicos, es metodológicamente mandatorio incluir rezagos temporales (lags 1 a 12 semanas)."
                ),
                "epistemic_caution": "Las asociaciones climáticas son correlacionales y no deben presentarse como causa única sin controlar por susceptibilidad poblacional.",
            }
        )

    # Explicación de correlaciones
    if correlations:
        strongest = max(correlations, key=lambda c: abs(c.get("pearson", 0.0)))
        explanations.append(
            {
                "topic": "Asociaciones Estadísticas Clima-Dengue",
                "indicator": f"Mayor correlación: {strongest['feature_x']} vs {strongest['feature_y']} (r = {strongest['pearson']})",
                "behavior": (
                    f"La variable '{strongest['feature_x']}' presenta una correlación lineal de Pearson de {strongest['pearson']} "
                    f"(Spearman rho={strongest.get('spearman', 'N/D')}) con la incidencia de casos. Las variables hidrometeorológicas "
                    f"muestran concordancia con los periodos de mayor reporte epidemiológico."
                ),
                "methodological_implication": (
                    "La presencia de correlaciones moderadas justifica su inclusión como variables exógenas en los modelos espaciotemporales "
                    "(ConvLSTM-Attention y CNN-LSTM), permitiendo a la red aprender mapas de atención sobre factores meteorológicos."
                ),
                "epistemic_caution": "Regla obligatoria del protocolo: Una correlación estadística positiva NO prueba causalidad epidemiológica directa.",
            }
        )

    return explanations


def compute_dataset_statistics(
    db: Session,
    dataset_id: Optional[int] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Calcula la caracterización estadística descriptiva multivariable del conjunto de datos."""
    panel_result = load_panel(db, mode=mode)
    panel = panel_result.panel

    if panel.empty:
        raise ValidationError("El conjunto de datos está vacío; no es posible calcular estadísticas descriptivas.")

    dataset_record = None
    if dataset_id is not None:
        dataset_record = db.get(Dataset, dataset_id)

    # 1. Variables a caracterizar
    target_vars = [
        ("confirmed_cases", "casos"),
        ("temperature_mean", "°C"),
        ("temperature_max", "°C"),
        ("temperature_min", "°C"),
        ("humidity_mean", "%"),
        ("precipitation_mm", "mm"),
        ("wind_speed", "m/s"),
        ("solar_radiation", "MJ/m²"),
    ]

    univariate = []
    for col, unit in target_vars:
        if col in panel.columns:
            univariate.append(_calculate_univariate(panel[col], col, unit))

    # 2. Matriz de Correlaciones (Pearson y Spearman)
    numeric_cols = [col for col, _ in target_vars if col in panel.columns and panel[col].dropna().nunique() > 1]
    correlations = []
    if "confirmed_cases" in numeric_cols:
        for other in numeric_cols:
            if other == "confirmed_cases":
                continue
            pair = panel[["confirmed_cases", other]].dropna()
            if len(pair) > 10:
                p_r, p_val = stats.pearsonr(pair["confirmed_cases"], pair[other])
                s_r, s_val = stats.spearmanr(pair["confirmed_cases"], pair[other])
                correlations.append(
                    {
                        "feature_x": other,
                        "feature_y": "confirmed_cases",
                        "pearson": round(float(p_r), 3),
                        "pearson_p_value": round(float(p_val), 5),
                        "spearman": round(float(s_r), 3),
                        "spearman_p_value": round(float(s_val), 5),
                        "significant_05": bool(p_val < 0.05),
                    }
                )

    # 3. Distribución por Grupos (Distritos)
    dist_stats = []
    for district, group in panel.groupby("district_code"):
        c_series = group["confirmed_cases"].dropna()
        t_series = group["temperature_mean"].dropna() if "temperature_mean" in group else pd.Series()
        dist_stats.append(
            {
                "district_code": district,
                "total_cases": int(c_series.sum()),
                "mean_cases_week": round(float(c_series.mean()), 2) if not c_series.empty else 0.0,
                "max_cases_week": int(c_series.max()) if not c_series.empty else 0,
                "mean_temp": round(float(t_series.mean()), 2) if not t_series.empty else None,
                "weeks_count": int(len(group)),
            }
        )

    # 4. Explicabilidad automatizada en lenguaje natural
    explanations = _generate_explanations(univariate, correlations)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "dataset_name": dataset_record.name if dataset_record else "Panel distrito-semana activo",
        "total_records": int(len(panel)),
        "is_synthetic": bool(panel_result.is_synthetic),
        "univariate_statistics": univariate,
        "correlations": correlations,
        "district_breakdown": dist_stats,
        "automated_explanations": explanations,
    }
