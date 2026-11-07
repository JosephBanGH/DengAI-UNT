# ANEXO A: Lineamientos Metodológicos y Tecnológicos Complementarios del Asesor

> **Fuente de origen:** Ficha técnica vinculante `docs/CasoSP01.docx` (Sección final: Anexo A).  
> **Ámbito de aplicación:** Proyecto SP-01 (DengAI-UNT) — Plataforma predictiva espacio-temporal y alerta temprana de dengue.  
> **Estado de cumplimiento:** Obligatorio y vinculante.

---

## A.1. Arquitectura Tecnológica Recomendada: IA + n8n + LangChain + LangFlow

La investigación no debe limitarse a comparar algoritmos. Se desarrolla una aplicación funcional reproducible que implementa el modelo seleccionado y demuestra su utilidad en un escenario epidemiológico realista.

### A.1.1. Arquitectura General por Capas

```
┌────────────────────────────────────────────────────────────────────────┐
│                      CAPA DE ADQUISICIÓN DE DATOS                      │
│   Datasets públicos · Sensores IoT · Satélites · Registros clínicos   │
│                 (MINSA / CDC Perú · NASA POWER · INEI)                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    ORQUESTACIÓN DE WORKFLOWS (n8n)                     │
│    Ingestión · Limpieza · Validación · Programación · Alertas · Audit   │
│             (WF-01 a WF-06 con logs y control de fallos)               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                CAPA DE LENGUAJE (LangChain / LangFlow)                 │
│  LangChain: Agentes, RAG, herramientas, recuperación y explicaciones.  │
│  LangFlow: Diseño visual, experimentación y evaluación de flujos LLM.  │
│      (Restricción: Nunca sustituye al modelo numérico validado)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    MOTOR DE INTELIGENCIA ARTIFICIAL                    │
│   3 Individuales (XGBoost, LSTM, ConvLSTM) + 2 Híbridos (CNN-LSTM,    │
│  ConvLSTM-Attention) + Optuna TPE/Pruner + SHAP + Pruebas Estadísticas │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       BACKEND Y ALMACENAMIENTO                         │
│           FastAPI · PostgreSQL / PostGIS · MinIO · MLflow              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FRONTEND Y RESULTADOS                          │
│     React · Dashboard · Mapas · Alertas · Reportes · Trazabilidad      │
└────────────────────────────────────────────────────────────────────────┘
```

### A.1.2. Función del Stack Tecnológico Obligatorio

| Tecnología | Función en el Proyecto |
|---|---|
| **Python** | Desarrollo de algoritmos, transformaciones tabulares y procesamiento de datos. |
| **TensorFlow/Keras o PyTorch** | Modelos de Deep Learning (LSTM, ConvLSTM, CNN-LSTM, ConvLSTM-Attention). |
| **Scikit-learn** | Modelos tabulares base, preprocesamiento, escaladores y métricas estadísticas. |
| **XGBoost / LightGBM** | Modelo tabular de referencia de alto rendimiento. |
| **Optuna** | Optimización bayesiana de hiperparámetros (TPE Sampler + Median/Hyperband Pruner). |
| **SHAP / LIME** | Explicabilidad local y global (atribución de variables climáticas e históricas). |
| **n8n** | Automatización programada de descarga, calidad, inferencia, alertas y reportes. |
| **LangChain** | Agente epidemiológico, RAG sobre guías oficiales y herramientas de consulta. |
| **LangFlow** | Entorno visual exportable para prototipado, depuración y evaluación del agente RAG. |
| **FastAPI** | Exposición de endpoints REST asíncronos para inferencia, control y reportes. |
| **PostgreSQL / PostGIS** | Persistencia relacional y espacial (geometrías distritales y consultas geoespaciales). |
| **MinIO** | Almacén de objetos compatible con S3 para datasets raw, artefactos de modelos y reportes. |
| **MLflow** | Registro de experimentos, seguimiento de hiperparámetros, métricas y versionado de modelos. |
| **React** | Interfaz de usuario interactiva (mapas coropléticos, gráficos temporales y alertas). |
| **Docker** | Despliegue reproducible multinodo mediante contenedores orquestados. |

### A.1.3. Seis Workflows n8n Obligatorios

1. **WF-01: Descarga periódica de datos** — Descarga idempotente desde APIs y repositorios (MINSA, NASA POWER), verificación de sumas SHA-256 y registro en bitácora raw.
2. **WF-02: Limpieza y validación de calidad** — Detección de faltantes, imputación sin fuga, chequeo de rangos físicos y generación de reporte de calidad de datos.
3. **WF-03: Inferencia programada con la API** — Envío semanal de variables procesadas al endpoint `/api/v1/ml/predict`, persistencia de pronósticos e incertidumbre.
4. **WF-04: Gestión y distribución de alertas tempranas** — Identificación de distritos en riesgo (bajo, medio, alto, brote), generación de notificaciones y trazabilidad.
5. **WF-05: Elaboración automática de reportes** — Generación programada de informes en PDF, DOCX y XLSX con metadatos de procedencia y archivo en MinIO.
6. **WF-06: Monitoreo, drift y auditoría de modelos** — Detección de desviación de datos (*data drift*) y de conceptos (*concept drift*), alertas por degradación de métricas y registro de ejecuciones.

### A.1.4. Reglas Críticas para LangChain y LangFlow

- **Función legítima:** Agente de consulta de protocolos de vigilancia epidemiológica, RAG sobre guías de OMS, CDC, OPS y MINSA, explicación de alertas y recuperación de evidencia.
- **Regla de honestidad epistémica:** El LLM **nunca** calcula ni inventa predicciones numéricas, tasas de incidencia ni niveles de alerta; actúa estrictamente como interfaz interpretativa consultando las predicciones auditables generadas por los modelos numéricos de la API.
- **Evaluación formal:** Debe evaluarse con un *conjunto dorado de preguntas* verificando:
  - *Groundedness* (fidelidad a las fuentes recuperadas).
  - Precisión de citas normativas/bibliográficas.
  - Tasa de abstención ante preguntas sin evidencia en el corpus.
  - Resistencia a inyección de instrucciones (*prompt injection*).

---

## A.2. Protocolo Experimental Común (F1 a F12)

### A.2.1. Fases del Protocolo

- **F1. Estado del arte:** Revisión sistemática PRISMA 2020 en Scopus, WoS, IEEE Xplore y PubMed.
- **F2. Definición del problema:** Unidad de análisis (distrito-semana), población de estudio y variables endógenas/exógenas.
- **F3. Adquisición y catalogación:** Descarga verificable, licencias de uso, almacenamiento raw inmutable (SHA-256) y diccionario de datos.
- **F4. Análisis exploratorio (EDA):** Detección de anomalías, estacionalidad, correlaciones cruzadas lag-caso y visualización geoespacial.
- **F5. Preprocesamiento:** Imputación transparente, normalización ajustada **exclusivamente en entrenamiento**, generación de rezagos (lags 1 a 12).
- **F6. Construcción de modelos:** 3 individuales (XGBoost, LSTM, ConvLSTM) + 2 híbridos (CNN-LSTM, ConvLSTM-Attention).
- **F7. Optimización de hiperparámetros:** Optuna con TPE Sampler y Median/Hyperband Pruner bajo presupuesto computacional comparable.
- **F8. Validación rigurosa:** Validación temporal (*walk-forward*) y validación geográfica (distritos no vistos en entrenamiento). **Prohibido split aleatorio 80/20**.
- **F9. Evaluación estadística:** Prueba de Friedman por bloques independientes, Wilcoxon-Holm, McNemar, DeLong y Diebold-Mariano.
- **F10. Explicabilidad e incertidumbre:** SHAP, análisis de rezagos más informativos, mapas de riesgo e intervalos de predicción.
- **F11. Prototipo funcional:** Integración completa de Backend (FastAPI), Frontend (React), Base de Datos (PostGIS), MinIO, MLflow y n8n.
- **F12. Publicación científica:** Redacción IMRyD con tablas y figuras generadas a partir de ejecuciones reales verificables.

### A.2.2. Estrategia de Validación Antifuga

- **Epidemiología y series de tiempo:** Validación temporal por ventanas expansivas/deslizantes (*walk-forward*) y retención de distritos completos para evaluar generalización espacial.
- **Prevención de fuga (*data leakage*):** Para predecir la semana $t+h$, cualquier variable explicativa o estadística resumen debe calcularse estrictamente con observaciones $s \le t$. Los escaladores y codificadores se ajustan únicamente en el conjunto de entrenamiento de cada pliegue.

### A.2.3. Configuración de Optuna

- 50 a 150 trials por modelo en fase final; optimización ejecutada **únicamente** sobre entrenamiento y validación interna. El conjunto de prueba final (*test*) permanece intacto hasta la evaluación definitiva.

### A.2.4. Métricas de Desempeño

- **Epidemiología / Regresión:** MAE, RMSE, MASE, WIS (Weighted Interval Score) y cobertura empírica de intervalos de predicción. (*MAPE solo si no existen valores cercanos a cero*).
- **Clasificación de Riesgo:** Sensibilidad, Especificidad, F1-Score, Coeficiente de Correlación de Matthew (MCC), AUC-ROC, AUC-PR y Puntaje de Brier.

### A.2.5. Pruebas Estadísticas Rigurosas

1. **Prueba de Friedman:** Comparación de múltiples modelos evaluada **sobre bloques independientes** (semanas epidemiológicas de prueba o distritos), **nunca sobre 5 promedios agregados**.
2. **Wilcoxon signed-rank + corrección de Holm:** Comparaciones pareadas post-hoc controlando la tasa de error por familia.
3. **Prueba de McNemar:** Comparación de matrices de confusión discordantes en la clasificación de riesgo brote/no-brote sobre los mismos casos.
4. **Prueba de DeLong:** Comparación con significancia estadística de áreas bajo la curva ROC (AUC-ROC) emparejadas.
5. **Prueba de Diebold-Mariano:** Comparación formal de precisión de pronósticos temporales.
