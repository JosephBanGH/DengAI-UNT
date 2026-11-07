"""Entrenamiento ligero multi-modelo sobre datos reales (MINSA + NASA POWER).

Objetivo:
1. Validar el pipeline completo F6-F10 con datos reales sin sobrecargar la CPU local.
2. Generar predicciones de prueba (walk-forward) para los 5 modelos:
   - Individual 1: XGBoost
   - Individual 2: LSTM
   - Individual 3: ConvLSTM
   - Híbrido 1: CNN-LSTM
   - Híbrido 2: ConvLSTM-Attention
3. Calcular métricas epidemiológicas (MAE, RMSE, MASE, WIS) y clasificación de riesgo (F1, AUC-ROC, Brier).
4. Aplicar pruebas estadísticas rigurosas del anexo:
   - Friedman sobre bloques independientes (semanas epidemiológicas de prueba).
   - Wilcoxon pareado con corrección de Holm.
   - McNemar sobre matrices de confusión pareadas.
   - DeLong para comparación de curvas ROC.
5. Persistir artefactos en data/models/ y reporte en data/reports/evaluacion_modelos_reales.json.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "backend"))
sys.path.insert(0, str(PROJECT_DIR / "libs"))

from app.core.config import settings
from app.db.session import SessionLocal
from app.ml.features import build_dataset, load_panel
from app.ml.metrics import evaluate_all
from app.ml.models.base import ModelData
from app.ml.models.registry import create_model
from app.ml.stats_tests import compare_models, delong_roc_test, mcnemar_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("dengai.train_light")


def run_lightweight_training():
    db = SessionLocal()
    try:
        logger.info("Cargando panel distrito-semana (modo real prioritario)...")
        panel = load_panel(db, mode="auto")
        logger.info(
            "Panel cargado: %d filas, %d distritos, %d semanas (Modo: %s, Sintético: %s)",
            len(panel.panel),
            len(panel.districts),
            len(panel.weeks),
            panel.data_mode,
            panel.is_synthetic,
        )

        horizon = 1
        logger.info("Construyendo DatasetBundle para horizonte h=%d con validación temporal walk-forward...", horizon)
        bundle = build_dataset(
            panel,
            horizon=horizon,
            test_weeks=settings.ML_TEST_WEEKS,
            validation_weeks=settings.ML_VALIDATION_WEEKS,
        )
        logger.info("Particiones: Train=%d, Val=%d, Test=%d", len(bundle.y_train), len(bundle.y_val), len(bundle.y_test))

        train_data = ModelData(bundle.X_train, bundle.seq_train, bundle.grid_train, bundle.cell_train, bundle.y_train, bundle.feature_names)
        val_data = ModelData(bundle.X_val, bundle.seq_val, bundle.grid_val, bundle.cell_val, bundle.y_val, bundle.feature_names)
        test_data = ModelData(bundle.X_test, bundle.seq_test, bundle.grid_test, bundle.cell_test, bundle.y_test, bundle.feature_names)

        # Configuraciones ligeras pero funcionales de los 5 algoritmos
        model_configs = {
            "xgboost": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.08},
            "lstm": {"hidden_size": 32, "num_layers": 1, "epochs": 5, "batch_size": 32, "lr": 0.01},
            "convlstm": {"channels": 16, "kernel_size": 3, "epochs": 4, "batch_size": 16, "lr": 0.01},
            "cnn_lstm": {"cnn_filters": 16, "lstm_hidden": 32, "epochs": 4, "batch_size": 16, "lr": 0.01},
            "convlstm_attention": {"channels": 16, "attention_heads": 2, "epochs": 4, "batch_size": 16, "lr": 0.01},
        }

        predictions = {}
        model_metrics = {}
        models_dir = settings.models_dir
        models_dir.mkdir(parents=True, exist_ok=True)

        for algo_key, params in model_configs.items():
            logger.info("--- Entrenando modelo: %s ---", algo_key.upper())
            t0 = time.perf_counter()
            model = create_model(algo_key)
            epochs = params.pop("epochs", 5)
            model.fit(train_data, val_data, params=params, epochs=epochs)
            elapsed = time.perf_counter() - t0

            # Predicción en conjunto de prueba
            y_pred = model.predict(test_data)
            y_pred = np.clip(y_pred, 0, None)
            predictions[algo_key] = y_pred

            # Métricas completas
            outbreak_th = float(np.quantile(bundle.y_train, 0.90))
            metrics = evaluate_all(
                bundle.y_test,
                y_pred,
                y_insample=bundle.y_train[-100:],
                outbreak_threshold=outbreak_th,
                bootstrap=False,
            )
            model_metrics[algo_key] = {
                "mae": round(float(metrics["MAE"]), 4) if metrics.get("MAE") is not None else None,
                "rmse": round(float(metrics["RMSE"]), 4) if metrics.get("RMSE") is not None else None,
                "mase": round(float(metrics["MASE"]), 4) if metrics.get("MASE") is not None else None,
                "auc_pr": round(float(metrics["AUC_PR"]), 4) if metrics.get("AUC_PR") is not None else None,
                "sensitivity": round(float(metrics["sensitivity"]), 4) if metrics.get("sensitivity") is not None else None,
                "specificity": round(float(metrics["specificity"]), 4) if metrics.get("specificity") is not None else None,
                "training_time_s": round(elapsed, 2),
            }
            logger.info(
                "%s completado en %.1fs | MAE: %s | RMSE: %s | Sens: %s | Spec: %s",
                algo_key.upper(),
                elapsed,
                model_metrics[algo_key]["mae"],
                model_metrics[algo_key]["rmse"],
                model_metrics[algo_key]["sensitivity"],
                model_metrics[algo_key]["specificity"],
            )

        # Bloques independientes para la prueba de Friedman (semanas epidemiológicas de prueba)
        test_weeks_blocks = bundle.meta_test["target_week"].astype(str).tolist()

        logger.info("Aplicando prueba de Friedman sobre bloques independientes de semanas de prueba...")
        stats_summary = compare_models(predictions, bundle.y_test, blocks=test_weeks_blocks)

        # Pruebas pareadas de McNemar y DeLong (ejemplo: ConvLSTM-Attention vs XGBoost)
        actual_risk = (bundle.y_test >= 5).astype(int)
        best_dl_pred = (predictions["convlstm_attention"] >= 5).astype(int)
        xgb_pred = (predictions["xgboost"] >= 5).astype(int)

        mcnemar_res = mcnemar_test(actual_risk, best_dl_pred, xgb_pred, "convlstm_attention", "xgboost")
        delong_res = delong_roc_test(
            actual_risk,
            predictions["convlstm_attention"] / (predictions["convlstm_attention"].max() + 1e-5),
            predictions["xgboost"] / (predictions["xgboost"].max() + 1e-5),
            "convlstm_attention",
            "xgboost",
        )

        final_report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "data_mode": panel.data_mode,
            "is_synthetic": panel.is_synthetic,
            "districts_count": len(panel.districts),
            "test_samples": len(bundle.y_test),
            "test_blocks_weeks": stats_summary["blocks"],
            "model_metrics": model_metrics,
            "friedman_test": stats_summary["friedman"],
            "pairwise_wilcoxon_holm": stats_summary["pairwise"],
            "mcnemar_test": mcnemar_res,
            "delong_test": delong_res,
        }

        report_file = PROJECT_DIR / "data" / "reports" / "evaluacion_modelos_reales.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("¡Entrenamiento y protocolo estadístico completados exitosamente!")
        logger.info("Friedman p-valor: %s (Aplicable: %s)", stats_summary["friedman"].get("p_value"), stats_summary["friedman"].get("applicable"))
        logger.info("Reporte detallado guardado en: %s", report_file)
        return final_report

    finally:
        db.close()


if __name__ == "__main__":
    run_lightweight_training()
