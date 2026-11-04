# Protocolo experimental F1–F12 y su implementación

Documento de trabajo que vincula cada fase del protocolo con el componente de software que la
soporta, su evidencia y su estado.

| Fase | Objetivo | Implementación en la plataforma | Evidencia / artefacto | Estado |
|---|---|---|---|---|
| **F1** Revisión científica | PRISMA 2020 en Scopus, WoS, IEEE Xplore, PubMed | Fuera del software; la plataforma documenta el vacío de conocimiento y expone el corpus verificado al asistente RAG | `docs/research/`, `assistant_service._corpus` | Documentado, pendiente de ejecución bibliográfica |
| **F2** Comprensión del problema | Población, distritos, unidad temporal, variables | Catálogo `geographic_units` (14 distritos), `epi_weeks` (ISO), diccionario de variables | `models/geo.py`, `dataset_service.FEATURE_DOCS` | Implementado |
| **F3** Datasets | Descarga, licencias, diccionario de datos | `data_sources` con licencia y estado; importación validada; `datasets` versionados con checksum | `services/data_service.py`, `services/dataset_service.py` | Implementado (carga real pendiente de credenciales) |
| **F4** EDA | Correlaciones temporales/espaciales, anomalías, visualizaciones | Dashboard, mapas, series territoriales, detección de anomalías por z-score en la importación | `dashboard_service`, `geo_service`, `data_service._record_quality` | Implementado |
| **F5** Preprocesamiento | Imputación, rezagos, normalización, particiones sin fuga | `build_features` (rezagos 1–12, vecindad, estacionalidad), escalador ajustado solo con train | `ml/features.py`, pruebas `test_pipeline.py` | Implementado y probado |
| **F6** Modelamiento | 3 individuales + 2 híbridos | XGBoost, LSTM, ConvLSTM, CNN-LSTM, ConvLSTM-Attention | `ml/models/`, `ml/trainer.py` | Implementado |
| **F7** Optimización | Optuna 50–150 trials, TPE + MedianPruner | `ExperimentRunner._optimize` con espacio de búsqueda por algoritmo | `ml/models/registry.SEARCH_SPACES` | Implementado (presupuesto de CPU reducible) |
| **F8** Validación | Walk-forward + geográfica | `split_temporal` (26/26 semanas) y `split_geographic` (distritos no vistos) | `ml/features.py`, `test_pipeline.py` | Implementado y probado |
| **F9** Evaluación estadística | Friedman, Wilcoxon-Holm, Diebold-Mariano, bootstrap | `ml/stats_tests.py`, endpoint `/ml/compare` | `services/comparison_service.py` | Implementado |
| **F10** Interpretabilidad | SHAP, importancia temporal, mapas de riesgo | `ml/explain.py` (TreeExplainer / KernelExplainer + fallback por permutación), `/ml/explain/{code}` | UI «Explicar» en Pronósticos | Implementado |
| **F11** Prototipo | Backend, frontend, orquestación | FastAPI + React/TS + n8n + reportes + alertas | Este repositorio | Implementado |
| **F12** Publicación | Artículo IMRyD, figuras, repositorio reproducible | Notebook, informe técnico, matriz de trazabilidad | `ml/notebooks/`, `docs/technical/` | Preparado; redacción pendiente |

## Decisiones metodológicas que deben defenderse en el artículo

1. **Unidad distrito-semana**: evita mezclar escalas y permite comparar incidencia con denominador
   poblacional.
2. **Sin partición aleatoria**: en series epidemiológicas una partición 80/20 aleatoria mezcla
   pasado y futuro; se usa walk-forward y validación geográfica.
3. **Intervalos empíricos de validación**: los cuantiles residuales se estiman en validación y se
   aplican a prueba; la cobertura reportada es por lo tanto honesta (out-of-sample).
4. **MAPE condicionado**: con conteos semanales bajos el MAPE es inestable; se reporta solo si al
   menos el 20 % de las observaciones supera el denominador mínimo.
5. **MASE escalado con el ingenuo estacional**: permite comparar contra una referencia trivial.
6. **Selección del ganador con contraste estadístico**: MAE como criterio primario y Friedman,
   Wilcoxon-Holm y Diebold-Mariano como evidencia de significancia.
7. **Híbridos con justificación**: la novedad debe demostrarse por superioridad empírica, no por
   combinación de arquitecturas.

## Amenazas a la validez (a declarar en el artículo)

- **Validez interna**: calidad del reporte de casos (subregistro, cambios de definición de caso),
  datos faltantes y errores de georreferenciación.
- **Validez externa**: el modelo entrenado en La Libertad no se transfiere automáticamente a otras
  regiones; requiere reentrenamiento y validación geográfica.
- **Validez de constructo**: la incidencia notificada es un proxy imperfecto de la transmisión real;
  los factores climáticos actúan con rezagos biológicos que el modelo aproxima con lags de 0–5
  semanas.
- **Conclusión estadística**: con 364 observaciones de prueba, el poder para detectar diferencias
  pequeñas es limitado; se reportan intervalos bootstrap.

## Reproducibilidad

1. Semilla fija (`set_reproducible_seed(42)`) y generación sintética con `SYNTHETIC_SEED`.
2. Checksum SHA-256 de cada artefacto y de cada conjunto de datos.
3. Predicciones de prueba persistidas para recalcular comparaciones sin reentrenar.
4. Notebook `ml/notebooks/SP01_dengue_reproducible.ipynb` con el flujo completo vía API.

## Resultados de referencia (datos sintéticos, h=1)

| Algoritmo | MAE | RMSE | MASE | Cobertura IC90 | AUC-PR | Sensibilidad | Especificidad |
|---|---|---|---|---|---|---|---|
| XGBoost | 3.497 | 11.416 | 0.858 | 0.797 | 0.419 | 0.917 | 0.803 |
| LSTM | 3.631 | 10.943 | 0.891 | 0.821 | 0.530 | 0.958 | 0.821 |
| CNN-LSTM | 4.249 | 11.969 | — | — | — | — | — |
| ConvLSTM | pendiente de corrida completa | | | | | | |
| ConvLSTM-Attention | pendiente de corrida completa | | | | | | |

Pruebas para h=1 (364 observaciones): Friedman p = 0.0050 → diferencias significativas;
Diebold-Mariano xgboost vs cnn_lstm p = 0.0001.

> Estos valores **no son evidencia científica sobre el dengue**: provienen del conjunto sintético de
> demostración. Su función es validar que el pipeline mide lo que debe medir.
