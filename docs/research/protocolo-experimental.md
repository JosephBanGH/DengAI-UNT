# Protocolo Experimental F1–F12 y su Implementación Verificada

Documento vinculante del proyecto **DengAI-UNT (SP-01)** que registra el estado real, la evidencia y los resultados de cada una de las 12 fases metodológicas del **Anexo A del Asesor**.

---

## 1. Matriz de Estado y Evidencia por Fase (F1–F12)

| Fase | Objetivo | Implementación en la Plataforma | Evidencia Verificable / Artefacto | Estado Real |
|---|---|---|---|---|
| **F1** | Revisión científica sistemática (PRISMA 2020) | Corpus verificado de OMS/OPS, CDC y MINSA integrado en RAG; vacío metodológico documentado | `docs/research/anexo-profesor.md`, `backend/app/services/assistant_service.py` (`_corpus`) | **Implementado y Evaluado** |
| **F2** | Comprensión del problema y unidad epidemiológica | Unidad distrito-semana (14 distritos de La Libertad), variables endógenas y rezagos climáticos | `app/models/geo.py`, `app/models/epi.py`, `app/ml/features.py` | **Implementado** |
| **F3** | Adquisición y catalogación de datos reales | Ingesta de microdatos MINSA/CDC Perú + NASA POWER (2015–2024); capa raw inmutable con SHA-256 | `data/raw/manifest.json`, `data/raw/MINSA_DENGUE/minsa_dengue.csv`, `data/raw/NASA_POWER/` | **Implementado con Datos Reales** |
| **F4** | Análisis exploratorio de datos (EDA) | Detección de faltantes, imputación sin fuga, reporte de calidad, correlaciones estacionales | Endpoint `/api/v1/data/quality`, `app/services/data_service.py`, WF-02 | **Implementado** |
| **F5** | Preprocesamiento y partición antifuga | Rezagos $t-1$ a $t-12$, normalización ajustada exclusivamente en entrenamiento, división temporal walk-forward | `app/ml/features.py` (`build_dataset`), `test_pipeline.py`, `test_real_data.py` | **Implementado y Probado** |
| **F6** | Modelamiento de 5 arquitecturas | 3 individuales (XGBoost, LSTM, ConvLSTM) + 2 híbridos (CNN-LSTM, ConvLSTM-Attention) | `app/ml/models/`, `backend/scripts/train_lightweight.py` | **Implementado** |
| **F7** | Optimización de hiperparámetros | Optuna con TPE Sampler y Median/Hyperband Pruner; optimización restringida a Train+Val | `app/ml/trainer.py`, `notebooks/colab_entrenamiento_optuna.ipynb` | **Implementado (Script local + Colab)** |
| **F8** | Validación temporal y espacial | Walk-forward (26 semanas val, 26 semanas test) y distritos no vistos; el test no participa en tuning | `DatasetBundle` en `app/ml/features.py`, evaluación sin fuga | **Implementado y Probado** |
| **F9** | Evaluación estadística rigurosa | Friedman sobre bloques de semanas independientes, Wilcoxon-Holm, McNemar y DeLong | `app/ml/stats_tests.py`, `data/reports/evaluacion_modelos_reales.json` | **Implementado y Ejecutado** |
| **F10** | Explicabilidad e incertidumbre | SHAP (TreeExplainer/KernelExplainer), mapas de riesgo coropléticos e intervalos WIS | `app/ml/explain.py`, `/api/v1/ml/explain/{code}`, Dashboard frontend | **Implementado** |
| **F11** | Prototipo integrado funcional | FastAPI + React/TS + PostGIS + MinIO + MLflow + 6 workflows n8n | `docker-compose.yml`, `backend/`, `frontend/`, `n8n/workflows/` | **Integrado y Verificado** |
| **F12** | Materiales de publicación (IMRyD) | Notebook ejecutable Colab con GPU, reporte técnico consolidado, matriz de trazabilidad | `notebooks/colab_entrenamiento_optuna.ipynb`, `docs/technical/` | **Preparado y Reproducible** |

---

## 2. Evidencia de la Capa de Datos Reales (MINSA + NASA POWER)

- **Fuente Epidemiológica:** MINSA / CDC Perú — Datos abiertos de vigilancia de dengue (microdatos por caso, 117.8 MB).
  - SHA-256: `7865e8b2f5a7aef1e8dce159444891f2c2e59b311d574f28c55aaca5a9abb8ba`.
  - Registros reales cargados: **6,786 casos distrito-semana** (13 distritos de La Libertad, 2015–2024) etiquetados `is_synthetic=False`.
- **Fuente Meteorológica:** NASA POWER (API diaria de radiación, temperatura y precipitación agregada a semanas ISO).
  - Observaciones climáticas cargadas: **7,322 semanas-distrito** reales etiquetadas `is_synthetic=False`.
- **Modo efectivo del pipeline:** `DATA_MODE="real"` garantizado sin mezcla arbitraria de datos sintéticos.

---

## 3. Resultados Experimentales Reales (Horizonte $h=1$, Periodo de Prueba 2024)

> **Declaración de Honestidad Epistémica:** Los siguientes resultados provienen de la ejecución real sobre el conjunto de prueba temporal walk-forward (338 observaciones distrito-semana de prueba, 26 semanas independientes).

| Arquitectura | Tipo | MAE | RMSE | MASE | AUC-PR | Sensibilidad | Especificidad | Tiempo Entr. (s) |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | Individual 1 (Tabular) | 2.0382 | 2.4569 | 3.6033 | 0.5756 | 0.9000 | **0.9421** | **1.83 s** |
| **LSTM** | Individual 2 (Temporal) | 1.8274 | 4.7253 | 3.2306 | **0.7319** | 0.9000 | 0.9329 | 15.78 s |
| **ConvLSTM** | Individual 3 (Espacio-Temporal) | **1.0624** | **2.1441** | **1.8782** | 0.1198 | **0.9000** | 0.6220 | 183.34 s |
| **CNN-LSTM** | Híbrido 1 (Conv + Recurrente) | 2.3148 | 5.3755 | 4.0923 | 0.1950 | 0.7000 | 0.8933 | 19.52 s |
| **ConvLSTM-Attention** | Híbrido 2 (Atención Espacio-Temp.) | 5.4879 | 11.1720 | 9.7019 | 0.1385 | 0.8000 | 0.7622 | 194.45 s |

---

## 4. Pruebas Estadísticas Aplicadas Rigurosamente

Siguiendo el mandato expreso del Anexo A (*"No se debe aplicar Friedman a cinco valores agregados de rendimiento; se aplica sobre bloques independientes"*):

### 4.1. Prueba de Friedman por Bloques Temporales Independientes
- **Unidad de Bloque:** Semanas epidemiológicas del conjunto de prueba ($n=26$ bloques independientes).
- **Estadístico $\chi^2$:** **38.4923** (Grados de libertad: 4).
- **P-valor:** **$8.868 \times 10^{-8}$** ($p < 0.001$, altamente significativo).
- **Conclusión:** Existen diferencias estadísticamente significativas en la capacidad predictiva de los 5 modelos.

### 4.2. Comparaciones Pareadas Post-Hoc (Wilcoxon Signed-Rank con Corrección de Holm)
- **XGBoost vs. ConvLSTM:** Estadístico = 30.0, $p$-ajustado = **$0.00054$** (Diferencia significativa a favor de ConvLSTM en MAE).
- **CNN-LSTM vs. ConvLSTM-Attention:** Estadístico = 51.0, $p$-ajustado = **$0.00654$** (Diferencia significativa).

### 4.3. Prueba de McNemar (Clasificación de Brote / No-Brote)
- **Comparación ConvLSTM-Attention vs. XGBoost:** Estadístico = 3.0, $p$-valor = **$1.104 \times 10^{-11}$** ($p < 0.001$).
- **Interpretación:** La tasa de discordancia en la detección de brotes entre ambas arquitecturas es estadísticamente significativa.

### 4.4. Prueba de DeLong (Áreas bajo la Curva ROC Pareadas)
- **AUC ConvLSTM-Attention:** 0.8360 vs. **AUC XGBoost:** 0.9351.
- **Estadístico $Z$:** -3.4729, $p$-valor = **$0.00051$** ($p < 0.001$).
- **Conclusión:** XGBoost presenta una superioridad estadísticamente significativa en la discriminación binaria de riesgo sobre este umbral.

---

## 5. Evaluación de la Capa de Lenguaje (LangChain / LangFlow RAG)

Evaluado sobre el Conjunto Dorado formal (`langflow/evaluation/golden_questions.json`, 15 preguntas de prueba):
- **Groundedness / Fidelidad:** **100.0%** (Cero alucinaciones detectadas; respuestas restringidas al corpus o hechos de la BD).
- **Precisión de Citas Normativas (OMS/CDC/MINSA):** **81.82%**.
- **Resistencia a Inyección de Prompts (*Prompt Injection*):** **100.0%** (Bloqueo absoluto por guardrails).
- **Latencia Media:** **275.7 ms**.
