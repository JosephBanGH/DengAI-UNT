"""Comparación de modelos con pruebas estadísticas (M08)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.stats_tests import compare_models, diebold_mariano
from app.ml.trainer import load_bundle, resolve_artifact_path
from app.models.ml import Experiment, MetricResult, ModelVersion


def _load_predictions(version: ModelVersion) -> Optional[Dict[str, Any]]:
    artifact = resolve_artifact_path(version.artifact_path) if version.artifact_path else None
    bundle = load_bundle(str(artifact)) if artifact and artifact.exists() else None
    if not bundle:
        return None
    raw_path = bundle.get("predictions_path")
    path = resolve_artifact_path(raw_path) if raw_path else None
    if not path or not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {
        "y_true": data["y_true"],
        "y_pred": data["y_pred"],
        "district": data["district"].astype(str),
        "target_week": data["target_week"].astype(str),
        "horizon": int((version.feature_names or {}).get("horizon") or 0),
        "algorithm": version.algorithm,
        "threshold": bundle.get("outbreak_threshold"),
        "data_mode": bundle.get("data_mode"),
        "is_synthetic": bundle.get("is_synthetic_data"),
    }


def compare(db: Session, *, horizon: int = 1) -> Dict[str, Any]:
    """Compara las versiones más recientes de cada algoritmo para un horizonte dado."""
    versions = list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())))
    selected: Dict[str, ModelVersion] = {}
    for version in versions:
        if int((version.feature_names or {}).get("horizon") or -1) != horizon:
            continue
        selected.setdefault(version.algorithm, version)
    if len(selected) < 2:
        raise ValidationError(
            "Se requieren al menos dos modelos entrenados para el mismo horizonte para comparar.",
            details={"horizon": horizon, "algorithms": sorted(selected)},
        )

    predictions: Dict[str, np.ndarray] = {}
    actual = None
    blocks: List[str] = []
    aligned_meta = None
    threshold: Optional[float] = None
    for algorithm, version in selected.items():
        payload = _load_predictions(version)
        if payload is None:
            continue
        if actual is None:
            actual = payload["y_true"]
            # Bloques independientes: semana objetivo (agrega distritos de la misma
            # semana). Nunca una observación distrito-semana por bloque.
            blocks = [str(week) for week in payload["target_week"]]
            aligned_meta = payload
            threshold = payload.get("threshold")
        if len(payload["y_true"]) != len(actual):
            continue
        predictions[algorithm] = payload["y_pred"]

    if actual is None or len(predictions) < 2:
        raise ValidationError("No hay predicciones almacenadas comparables para ese horizonte.")

    binary_actual = None
    binary_predictions: Dict[str, np.ndarray] = {}
    if threshold is not None:
        binary_actual = (np.asarray(actual) >= float(threshold)).astype(int)
        binary_predictions = {
            algorithm: (np.asarray(pred) >= float(threshold)).astype(int)
            for algorithm, pred in predictions.items()
        }
    stats = compare_models(
        predictions,
        actual,
        blocks,
        binary_actual=binary_actual,
        binary_predictions=binary_predictions,
        # DeLong solo usa el orden de los puntajes; se documenta que no son
        # probabilidades calibradas sino el conteo pronosticado como score de riesgo.
        probabilities=predictions if binary_actual is not None else None,
        block_unit="target_week",
    )

    rows = []
    ranking = []
    for algorithm, pred in predictions.items():
        mae = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        ver = selected.get(algorithm)
        v_metrics = (ver.metrics or {}) if ver else {}
        rows.append({"algorithm": algorithm, "split": "test", "horizon_weeks": horizon, "metric": "MAE", "value": round(mae, 4), "ci_lower": v_metrics.get("MAE_ci_lower"), "ci_upper": v_metrics.get("MAE_ci_upper"), "n_observations": int(len(actual))})
        rows.append({"algorithm": algorithm, "split": "test", "horizon_weeks": horizon, "metric": "RMSE", "value": round(rmse, 4), "ci_lower": v_metrics.get("RMSE_ci_lower"), "ci_upper": v_metrics.get("RMSE_ci_upper"), "n_observations": int(len(actual))})
        ranking.append(
            {
                "algorithm": algorithm,
                "MAE": round(mae, 4),
                "MAE_ci_lower": v_metrics.get("MAE_ci_lower"),
                "MAE_ci_upper": v_metrics.get("MAE_ci_upper"),
                "RMSE": round(rmse, 4),
                "RMSE_ci_lower": v_metrics.get("RMSE_ci_lower"),
                "RMSE_ci_upper": v_metrics.get("RMSE_ci_upper"),
                "MAPE": v_metrics.get("MAPE"),
                "MASE": v_metrics.get("MASE"),
                "AUC_PR": v_metrics.get("AUC_PR"),
                "sensitivity": v_metrics.get("sensitivity"),
                "specificity": v_metrics.get("specificity"),
                "Brier": v_metrics.get("Brier"),
            }
        )
    ranking.sort(key=lambda item: item["MAE"])

    winner = ranking[0]["algorithm"] if ranking else None
    dm_tests = []
    if winner and aligned_meta is not None:
        for algorithm, pred in predictions.items():
            if algorithm == winner:
                continue
            dm_tests.append(
                diebold_mariano(actual, predictions[winner], pred, winner, algorithm, blocks=blocks)
            )

    basis = (
        f"El modelo {winner} obtuvo el MAE más bajo ({ranking[0]['MAE']}) sobre el conjunto de prueba "
        f"con {len(actual)} observaciones distrito-semana. "
        if ranking
        else ""
    )
    friedman = stats.get("friedman", {})
    if isinstance(friedman, dict) and friedman.get("applicable"):
        basis += (
            "La prueba de Friedman "
            + ("detectó diferencias significativas" if friedman.get("significant") else "no detectó diferencias significativas")
            + f" (p = {friedman.get('p_value'):.4f})."
        )
    return {
        "rows": rows,
        "ranking": ranking,
        "tests": [stats.get("friedman", {})]
        + stats.get("pairwise", [])
        + stats.get("mcnemar", [])
        + stats.get("delong", [])
        + dm_tests,
        "blocks": stats.get("blocks"),
        "block_unit": stats.get("block_unit"),
        "assumptions": stats.get("assumptions"),
        "winner": winner,
        "winner_basis": basis,
        "horizon": horizon,
        "n_observations": int(len(actual)),
        "caveat": (
            "La comparación utiliza las predicciones almacenadas del conjunto de prueba. "
            "Las pruebas de Friedman/Wilcoxon agregan pérdidas por semana objetivo como bloque "
            "independiente. Un MAE menor no implica superioridad causal; revise significancia, "
            "tamaño del efecto y estabilidad temporal."
        ),
        "artifacts": list_artifact_catalog(),
        "interpretations": build_epidemiological_interpretations(),
    }


def list_artifact_catalog() -> List[Dict[str, Any]]:
    from app.core.config import PROJECT_DIR

    exp_dir = PROJECT_DIR / "data" / "artifacts" / "experimentos"
    items = [
        {
            "filename": "tabla_1_metricas_rendimiento.csv",
            "title": "Tabla 1: Métricas de rendimiento en conjunto de prueba (n=338)",
            "kind": "table_csv",
            "format": "CSV",
            "path": exp_dir / "tabla_1_metricas_rendimiento.csv",
            "description": "MAE, RMSE (con IC 95% bootstrap), MAPE, MASE, AUC-PR, sensibilidad, especificidad y Brier Score.",
        },
        {
            "filename": "tabla_2_wilcoxon_holm.csv",
            "title": "Tabla 2: Prueba pareada de Wilcoxon con corrección de Holm-Bonferroni",
            "kind": "table_csv",
            "format": "CSV",
            "path": exp_dir / "tabla_2_wilcoxon_holm.csv",
            "description": "Contrastes post-hoc por pares sobre 26 bloques semanales independientes tras Friedman global.",
        },
        {
            "filename": "tabla_3_mcnemar_brote.csv",
            "title": "Tabla 3: Prueba exacta de McNemar en clasificación binaria de brote (>= 7 casos)",
            "kind": "table_csv",
            "format": "CSV",
            "path": exp_dir / "tabla_3_mcnemar_brote.csv",
            "description": "Evaluación de pares discordantes (b vs. c) en detección operativa de brotes epidémicos.",
        },
        {
            "filename": "tabla_4_delong_roc.csv",
            "title": "Tabla 4: Prueba pareada de DeLong sobre curvas ROC",
            "kind": "table_csv",
            "format": "CSV",
            "path": exp_dir / "tabla_4_delong_roc.csv",
            "description": "Comparación no paramétrica de áreas bajo la curva ROC (AUC-ROC) para discriminación de brotes.",
        },
        {
            "filename": "mejores_hiperparametros_optuna.json",
            "title": "Hiperparámetros óptimos seleccionados por Optuna TPE (100 trials)",
            "kind": "config_json",
            "format": "JSON",
            "path": exp_dir / "mejores_hiperparametros_optuna.json",
            "description": "Configuraciones óptimas de los 5 modelos obtenidas sin fuga de datos sobre validación.",
        },
        {
            "filename": "figura1_arquitectura.png",
            "title": "Figura 1: Arquitectura tecnológica DengAI-UNT (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura1_arquitectura.png",
            "description": "Arquitectura en capas (Cliente, Servicios FastAPI, Persistencia PostGIS/MinIO/MLflow).",
        },
        {
            "filename": "figura1_arquitectura.svg",
            "title": "Figura 1: Arquitectura tecnológica DengAI-UNT (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura1_arquitectura.svg",
            "description": "Esquema vectorial escalable de la arquitectura para publicación en revista científica.",
        },
        {
            "filename": "figura2_series.png",
            "title": "Figura 2: Serie temporal decenal 2015–2024 y meteorología (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura2_series.png",
            "description": "Dinámica epidemiológica decenal (65 305 casos) y variables bioclimáticas de NASA POWER a 600 DPI.",
        },
        {
            "filename": "figura2_series.svg",
            "title": "Figura 2: Serie temporal decenal 2015–2024 (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura2_series.svg",
            "description": "Serie temporal y régimen meteorológico en formato vectorial SVG.",
        },
        {
            "filename": "figura3_ccf.png",
            "title": "Figura 3: Función de correlación cruzada clima-dengue 0–12 semanas (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura3_ccf.png",
            "description": "Acoplamiento biofísico de temperatura y precipitación con bandas de significancia de Bartlett al 95%.",
        },
        {
            "filename": "figura3_ccf.svg",
            "title": "Figura 3: Función de correlación cruzada (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura3_ccf.svg",
            "description": "Correlación cruzada clima-dengue en formato vectorial SVG escalable.",
        },
        {
            "filename": "figura4_mae.png",
            "title": "Figura 4: MAE con IC 95% bootstrap y Curvas ROC emparejadas (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura4_mae.png",
            "description": "Panel comparativo de error de predicción y discriminación de brotes con prueba de DeLong a 600 DPI.",
        },
        {
            "filename": "figura4_mae.svg",
            "title": "Figura 4: Desempeño y curvas ROC (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura4_mae.svg",
            "description": "Panel bivariado de evaluación fuera de muestra en formato vectorial SVG.",
        },
        {
            "filename": "figura5_riesgo.png",
            "title": "Figura 5: Validación espacial distrital observada vs pronosticada (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura5_riesgo.png",
            "description": "Dispersión espacial y concordancia en los 13 distritos de La Libertad a 600 DPI.",
        },
        {
            "filename": "figura5_riesgo.svg",
            "title": "Figura 5: Validación espacial distrital (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura5_riesgo.svg",
            "description": "Mapa y gradiente territorial observado vs pronosticado en SVG.",
        },
        {
            "filename": "figura6_shap.png",
            "title": "Figura 6: Interpretabilidad global SHAP y atención temporal (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "figura6_shap.png",
            "description": "Importancia de predictores bioclimáticos y perfil de memoria temporal a 600 DPI.",
        },
        {
            "filename": "figura6_shap.svg",
            "title": "Figura 6: Interpretabilidad SHAP y atención temporal (Vectorial SVG)",
            "kind": "figure_svg",
            "format": "SVG",
            "path": exp_dir / "figura6_shap.svg",
            "description": "Explicabilidad biofísica del modelo en formato vectorial SVG.",
        },
        {
            "filename": "comparacion_modelos_optuna.png",
            "title": "Figura: Comparación de MAE y RMSE en test con intervalos de confianza al 95% (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "comparacion_modelos_optuna.png",
            "description": "Gráfico comparativo de alta resolución (600 DPI) del error puntual y cuadrático por arquitectura.",
        },
        {
            "filename": "curvas_roc_modelos.png",
            "title": "Figura: Curvas ROC para detección temprana de brote epidémico (600 DPI)",
            "kind": "figure_png",
            "format": "PNG (600 DPI)",
            "path": exp_dir / "curvas_roc_modelos.png",
            "description": "Curvas ROC y valores AUC-ROC en alta resolución (600 DPI) sobre el conjunto de prueba.",
        },
        {
            "filename": "fig2_serie_semanal_real_300dpi.png",
            "title": "Figura 2 del Manuscrito: Serie semanal real de casos y variables climáticas 2015–2024 (300 DPI)",
            "kind": "figure_png",
            "format": "PNG",
            "path": exp_dir / "fig2_serie_semanal_real_300dpi.png",
            "description": "Serie histórica de 522 semanas (65 305 casos confirmados), temperatura T2M y precipitación NASA POWER.",
        },
        {
            "filename": "fig3_correlacion_cruzada_real_300dpi.png",
            "title": "Figura 3 del Manuscrito: Correlación cruzada clima-dengue por rezago 0–12 semanas (300 DPI)",
            "kind": "figure_png",
            "format": "PNG",
            "path": exp_dir / "fig3_correlacion_cruzada_real_300dpi.png",
            "description": "Acoplamiento temporal entre temperatura, precipitación y humedad relativa frente a la incidencia.",
        },
        {
            "filename": "fig5_validacion_distrital_real_300dpi.png",
            "title": "Figura 5 del Manuscrito: Concordancia y error distrital de ConvLSTM en prueba (300 DPI)",
            "kind": "figure_png",
            "format": "PNG",
            "path": exp_dir / "fig5_validacion_distrital_real_300dpi.png",
            "description": "Desempeño territorial por distrito sobre las 338 observaciones del conjunto de prueba.",
        },
        {
            "filename": "fig6_explicabilidad_real_300dpi.png",
            "title": "Figura 6 del Manuscrito: Explicabilidad global de predictores y memoria temporal (300 DPI)",
            "kind": "figure_png",
            "format": "PNG",
            "path": exp_dir / "fig6_explicabilidad_real_300dpi.png",
            "description": "Jerarquía de importancia de las 59 variables y perfil de sensibilidad temporal en ventana de 12 semanas.",
        },
        {
            "filename": "resultados_experimentos_dengai.zip",
            "title": "Paquete completo de artefactos experimentales reproducibles (ZIP)",
            "kind": "bundle_zip",
            "format": "ZIP",
            "path": PROJECT_DIR / "resultados_experimentos_dengai.zip",
            "description": "Archivo comprimido original con tablas CSV, JSON de Optuna, figuras PNG a 300 DPI y modelos .joblib.",
        },
    ]
    out = []
    for item in items:
        p = item.pop("path")
        exists = p.exists()
        out.append(
            {
                **item,
                "available": exists,
                "size_bytes": p.stat().st_size if exists else 0,
                "download_url": f"/ml/artifacts/{item['filename']}",
            }
        )
    return out


def build_epidemiological_interpretations() -> List[Dict[str, Any]]:
    return [
        {
            "id": "rendimiento_comparativo",
            "title": "Interpretación del rendimiento comparativo y transición de régimen epidémico (Tabla 1 y Figura de Errores)",
            "summary": (
                "ConvLSTM alcanzó el menor error absoluto medio en prueba (MAE = 0.7889 casos/semana; IC 95 %: 0.6484–0.9279) "
                "y la mejor calibración probabilística (Brier = 0.1899), superando significativamente a XGBoost (MAE = 1.2317; p_Holm = 0.00044) "
                "y a CNN-LSTM (MAE = 0.9176; p_Holm = 0.0385)."
            ),
            "epidemiological_rationale": (
                "La partición walk-forward capturó un cambio de régimen extremo: la ventana de validación (2023-W52 a 2024-W25) "
                "coincidió con la mayor epidemia histórica de dengue en La Libertad (pico de 3 211 casos semanales en 2024-W15 tras el ciclón Yaku "
                "y El Niño Costero), mientras que la ventana de prueba (2024-W26 a 2024-W52, n = 338) correspondió a la fase de recesión posbrote "
                "e invierno-primavera austral (75.1 % de semanas con cero casos en el panel histórico). La arquitectura ConvLSTM de 1 capa "
                "(hidden_channels = 24, kernel 3x3) acopló la difusión espacial entre distritos contiguos de la conurbación de Trujillo "
                "(El Porvenir, La Esperanza, Florencia de Mora y Trujillo) con la memoria recurrente, amortiguando la sobreestimación residual "
                "tras el colapso invernal de la transmisión."
            ),
        },
        {
            "id": "discriminacion_brotes",
            "title": "Explicabilidad en detección temprana de brotes y curvas ROC/PR (Tablas 3 y 4 y Figura ROC)",
            "summary": (
                "En discriminación de brotes (umbral P90 de entrenamiento >= 7 casos/semana), los cinco modelos alcanzaron una sensibilidad "
                "del 90.0 % (excepto CNN-LSTM con 70.0 %). XGBoost obtuvo el mayor AUC-ROC (0.9476), seguido de LSTM (0.9229) y ConvLSTM (0.9189), "
                "sin diferencias estadísticamente significativas entre estos tres en la prueba de DeLong (p > 0.36), pero superando todos a CNN-LSTM (AUC-ROC = 0.8232, p <= 0.0106)."
            ),
            "epidemiological_rationale": (
                "En la prueba de McNemar, XGBoost superó significativamente a LSTM en pares discordantes de clasificación binaria (b = 8 vs. c = 0, p = 0.0078), "
                "mientras que LSTM lideró el AUC-PR (0.7174) en presencia de desbalance severo. Desde la perspectiva de vigilancia en salud pública "
                "(GERESA La Libertad), esto demuestra una complementariedad operativa: ConvLSTM proporciona la cuantificación puntual más precisa "
                "del volumen esperado de pacientes para dimensionar camas y kits NS1/ELISA en periodos endémicos, mientras que las particiones "
                "de umbral de XGBoost sobre rezagos cortos (cases_lag_1 a cases_lag_4) ofrecen un ordenamiento robusto de riesgo para activar cercos "
                "entomológicos tempranos."
            ),
        },
        {
            "id": "dinamica_vector_clima",
            "title": "Dinámica biofísica de Aedes aegypti y rezagos climáticos en la costa norte peruana",
            "summary": (
                "Las correlaciones cruzadas confirman que la temperatura media antecede a la incidencia de dengue con un rezago óptimo de "
                "6 a 9 semanas (r = 0.145 en lag 8, p < 0.0001) y la humedad relativa muestra acoplamiento sincrónico-rezagado (lag 0 a 4 semanas), "
                "mientras que la inercia autorregresiva domina el corto plazo (r = 0.903 en lag 1)."
            ),
            "epidemiological_rationale": (
                "En los valles costeros áridos de La Libertad (Trujillo, Ascope, Chepén y Pacasmayo; precipitación media de 4.4 mm/semana), "
                "la reproducción de Aedes aegypti no depende exclusivamente de lluvias locales intensas, sino del almacenamiento intradomiciliario "
                "de agua y del estrés térmico. El incremento térmico estival (21–26 °C) acelera el desarrollo larvario (de 14 a 7 días), "
                "reduce la duración del ciclo gonotrófico y acorta el Período de Incubación Extrínseca (PIE) del virus DENV de 12–14 días a 7–8 días. "
                "La suma encadenada de: (i) 1–2 semanas de eclosión y maduración acuática, (ii) 1–2 semanas de amplificación vectorial e infección "
                "extrínseca, (iii) 4–7 días de incubación intrínseca humana y (iv) 1–2 semanas de transmisión secundaria y notificación explica "
                "exactamente la ventana de 6 a 9 semanas capturada por los tensores secuenciales de 12 semanas."
            ),
        },
    ]


def metric_matrix(db: Session, *, horizon: Optional[int] = None) -> List[Dict[str, Any]]:
    stmt = select(MetricResult)
    if horizon:
        stmt = stmt.where(MetricResult.horizon_weeks == horizon)
    return [
        {
            "algorithm": row.algorithm,
            "split": row.split,
            "horizon_weeks": row.horizon_weeks,
            "metric": row.metric,
            "value": row.value,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "n_observations": row.n_observations,
        }
        for row in db.scalars(stmt.order_by(MetricResult.algorithm, MetricResult.metric))
    ]
