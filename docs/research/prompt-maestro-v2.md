# PROMPT MAESTRO v2 — SP-01 DENGUE INTELLIGENCE
## Cierre de brechas, datos reales y rigor experimental (F1–F12)

> **Cómo usarlo:** abrir una sesión nueva en modo agente (no Plan), con este repositorio como
> directorio de trabajo, y pegar todo el contenido desde la sección 0 hasta la 12.

---

## 0. Contexto operativo

Trabajas dentro del repositorio **DengAI-UNT** (ruta:
`C:\D\UNT\10 CICLO\Trabajo de Investigación\IA DENGUE\DengAI-UNT`).

Es un proyecto real, con código, pruebas y documentación ya existentes. **No empieces de cero y no
reemplaces lo que funciona.** Antes de escribir una línea de código: inspecciona el repositorio,
verifica el estado real de cada componente y reporta lo que encontraste.

El proyecto **aún no opera con datos reales**: arranca con un conjunto sintético de 14 distritos de
La Libertad. Cerrar esa brecha es la prioridad número uno de esta sesión.

---

## 1. Rol

Actúa como un equipo senior multidisciplinario integrado por:

- Arquitecto de soluciones y arquitecturas empresariales.
- Ingeniero de IA/ML especializado en Deep Learning y modelos espacio-temporales.
- Científico de datos especializado en epidemiología, salud pública y análisis geoespacial.
- Desarrollador Backend (Python, FastAPI, Pydantic, SQLAlchemy, Alembic).
- Desarrollador Frontend (React, TypeScript, Vite).
- Arquitecto/administrador de PostgreSQL + PostGIS.
- Ingeniero de automatización con n8n.
- Especialista en RAG, agentes, LangChain y LangFlow.
- Especialista DevOps, ciberseguridad, CI/CD y despliegue reproducible.
- Especialista en visualización, BI y generación automatizada de informes.

Tu entregable **no es una propuesta ni un conjunto de fragmentos**: es software funcionando,
integrado, probado, reproducible y con evidencia verificable. Si algo no se pudo verificar, se
declara como limitación; no se presenta como terminado.

---

## 2. Fuentes de verdad y jerarquía

1. `docs/CasoSP01.docx` — ficha técnica SP-01 **incluyendo el ANEXO del profesor al final**
   (arquitectura IA + n8n + LangChain + LangFlow; protocolo experimental común; validación;
   Optuna; métricas; pruebas estadísticas).
2. Este prompt — resumen vinculante de ese anexo. Si hay conflicto, **gana el anexo del profesor**.
3. `README.md`, `docs/architecture/arquitectura.md`, `docs/architecture/base-de-datos.md`,
   `docs/research/protocolo-experimental.md`, `docs/technical/matriz-trazabilidad.md`,
   `docs/technical/informe-tecnico.md`.
4. El código y las pruebas existentes.

Reglas de interpretación:

- No elimines, simplifiques ni sustituyas requisitos metodológicos del documento.
- Distingue siempre **requisito original** vs **mejora propuesta**.
- Si falta información: decide técnicamente, documenta el supuesto y continúa.
- **No inventes** datos epidemiológicos, métricas, resultados, fuentes, credenciales ni
  integraciones supuestamente funcionales.

---

## 3. Estado actual verificado (NO rehacer)

Antes de tocar nada, confirma esto en el repo y corrige el inventario si algo cambió:

- **Backend**: FastAPI + Pydantic v2 + SQLAlchemy 2 + Alembic + OpenAPI. Routers en
  `backend/app/api/v1/` (auth, admin_users, module_audit, analytics, data, ml, ops, system).
- **Seguridad**: JWT access/refresh, bcrypt, bloqueo por intentos, sesiones revocables, RBAC con
  7 roles y 27 permisos, bitácora **encadenada por hash** con verificación de manipulación.
- **ML**: 5 algoritmos en `backend/app/ml/models/` (XGBoost, LSTM, ConvLSTM, CNN-LSTM,
  ConvLSTM-Attention), `trainer.py` con Optuna TPE + Median Pruner, `features.py` con lags 1–12,
  particiones temporales y geográficas, `metrics.py`, `stats_tests.py`, `explain.py` (SHAP).
- **Frontend**: React 18 + TypeScript + Vite en `frontend/src/`, con páginas Dashboard, RiskMap,
  Forecasts, MlLab, ExperimentDetail, Alerts, Data, Datasets, Reports, Assistant, Audit, Users,
  Roles, Settings, Workflows.
- **Reportes**: PDF, DOCX y XLSX en `backend/app/services/reports_service.py` (16 tipos).
- **n8n**: solo `n8n/workflows/wf-01-descarga.json` … `wf-04-alertas.json`.
- **Asistente**: `backend/app/services/assistant_service.py` con proveedor determinista
  `grounded_template`, RAG sobre corpus interno y validación de cifras; LLM OpenAI-compatible
  opcional.
- **Base de datos**: SQLite local por defecto; PostgreSQL 16 + PostGIS en `docker-compose.yml`
  (migración `0002_postgis`).
- **Datos**: conjunto **sintético** sembrado automáticamente (~5 740 registros distrito-semana).
- **Pruebas**: `backend/tests/` (auth, RBAC, auditoría, importación, pipeline, entrenamiento).

---

## 4. Brecha a cerrar (diagnóstico inicial obligatorio)

Verifica y documenta el estado real de cada punto. Como mínimo:

1. **No hay datos reales** integrados de punta a punta (solo sintéticos).
2. **Faltan WF-05 y WF-06** de n8n (reportes automáticos y auditoría/monitoreo de modelos).
3. **No hay LangFlow** ni flujos visuales exportables/evaluados.
4. **MLflow está desactivado** (`MLFLOW_ENABLED=false`) y no hay tracking real de experimentos.
5. **No hay MinIO** ni almacenamiento de objetos para datasets, artefactos y reportes.
6. **RAG sin corpus real** de OMS/CDC/documentos científicos verificables.
7. **Rigor estadístico incompleto** frente al anexo: falta validación cruzada agrupada para
   datasets pequeños, McNemar, DeLong, y verificar que Friedman se aplique **sobre bloques
   independientes**, no sobre 5 promedios agregados.
8. **Trazabilidad del anexo**: el ANEXO del profesor no está registrado como documento del repo.

---

## 5. Objetivo de esta sesión

Cerrar las brechas anteriores y dejar la plataforma:

- operando con **al menos una fuente real de datos** de punta a punta;
- con **6 workflows n8n** importables y ejecutables;
- con **LangChain + LangFlow** justificados funcionalmente y **evaluados**;
- con **MLflow + MinIO** activos en Docker;
- con el **protocolo F1–F12** implementado y trazado;
- con validación temporal/geográfica **sin fuga de información**;
- con comparación de los 5 modelos respaldada por **pruebas estadísticas correctamente aplicadas**;
- con pruebas automatizadas, documentación y reproducibilidad verificadas.

---

## 6. Requisitos vinculantes del anexo del profesor

### 6.1 Arquitectura por capas (implementar y documentar)

```
CAPA DE ADQUISICIÓN DE DATOS
Datasets públicos · APIs · registros clínicos/epidemiológicos · imágenes · sensores
        ↓
ORQUESTACIÓN (n8n)
Ingestión · limpieza · validación · programación · alertas · auditoría
        ↓
CAPA DE LENGUAJE (LangChain / LangFlow)
Agentes · RAG · herramientas · recuperación de evidencia · explicaciones
        ↓
MOTOR DE IA
3 modelos individuales + 2 híbridos + Optuna + SHAP + validación estadística
        ↓
BACKEND Y ALMACENAMIENTO
FastAPI · PostgreSQL/PostGIS · MinIO · MLflow
        ↓
FRONTEND Y RESULTADOS
React · dashboard · mapas · alertas · reportes PDF/Word/Excel · trazabilidad
```

La arquitectura debe reflejarse en `docs/architecture/arquitectura.md` y en `docker-compose.yml`.

### 6.2 Stack obligatorio y función

| Tecnología | Función |
|---|---|
| Python | Algoritmos y procesamiento de datos |
| TensorFlow/Keras o PyTorch | Deep Learning (usar el que ya esté implementado) |
| Scikit-learn | Modelos clásicos, métricas, validación |
| XGBoost / LightGBM | Modelos tabulares |
| Optuna | Optimización de hiperparámetros |
| SHAP / LIME | Explicabilidad |
| n8n | Automatización y ejecución programada |
| LangChain | Agentes, herramientas, RAG |
| LangFlow | Diseño visual, experimentación y evaluación de flujos LLM |
| FastAPI | Inferencia y API REST |
| PostgreSQL / PostGIS | Datos estructurados y geoespaciales |
| MinIO | Almacenamiento de objetos (datasets, artefactos, reportes) |
| MLflow | Tracking de experimentos y modelos |
| React | Dashboard e interfaz |
| Docker | Despliegue reproducible |

No agregues tecnologías solo para aumentar la lista. Cada una debe tener uso real y evidencia.

### 6.3 Seis workflows n8n (obligatorios)

| WF | Proceso automatizado | Criterio de aceptación |
|---|---|---|
| **WF-01** | Descarga periódica de datasets y datos meteorológicos/epidemiológicos | Ejecutable, idempotente, con checksum y registro de procedencia |
| **WF-02** | Limpieza, control de calidad y detección de faltantes | Reporte de calidad generado y persistido |
| **WF-03** | Ejecución programada de predicciones vía FastAPI | Consume endpoint real, guarda resultados y maneja errores |
| **WF-04** | Generación y distribución de alertas de riesgo | Alertas con nivel, distrito, semana, horizonte e incertidumbre |
| **WF-05** | Elaboración automática de reportes científicos y gerenciales | Genera PDF/DOCX/XLSX con metadatos y los distribuye/archiva |
| **WF-06** | Auditoría, monitoreo y evaluación del rendimiento de modelos | Registra ejecuciones, detecta degradación/drift y notifica fallos |

Cada workflow debe: ser **JSON importable**, tener control de errores, reintentos, registro de
ejecuciones y notificación de fallos. Documenta cada uno en `docs/` y verifica la importación real
en n8n (Docker).

### 6.4 LangChain y LangFlow (con justificación evaluable)

El LLM **no** se incorpora para sumar tecnologías. Debe tener función y evaluación:

- Agente de consulta de protocolos de vigilancia epidemiológica.
- RAG sobre guías de OMS, CDC y documentos científicos verificables.
- Generación de explicaciones de alertas epidemiológicas.
- Asistente para consultar resultados de los modelos.
- Recuperación de evidencia científica para interpretar resultados.

**Regla crítica:** el LLM actúa como interfaz de consulta y explicación; **nunca sustituye al modelo
predictivo validado**. No puede inventar pronósticos, cifras ni niveles de riesgo. Toda cifra debe
provenir de la API/base de datos y validarse contra los hechos recuperados.

En **LangFlow** implementa flujos visuales equivalentes y **exportables**, y evalúa:

- groundedness / fidelidad a las fuentes;
- precisión de citas;
- tasa de abstención ante preguntas sin evidencia;
- latencia y costo;
- resistencia a inyección de prompts.

Usa un **conjunto dorado de preguntas** con respuestas y fuentes esperadas. Guarda resultados.

### 6.5 Protocolo experimental F1–F12

| Fase | Actividades | Producto |
|---|---|---|
| F1 | PRISMA 2020 en Scopus, WoS, IEEE Xplore, PubMed | Estado del arte y vacío |
| F2 | Población, unidad de análisis, variables | Protocolo |
| F3 | Selección, descarga, licencias, diccionario | Dataset documentado |
| F4 | EDA, correlaciones, anomalías, visualizaciones | Informe exploratorio |
| F5 | Limpieza, imputación, normalización, selección | Dataset experimental |
| F6 | 3 individuales + 2 híbridos | 5 modelos entrenados |
| F7 | Optuna | Modelos optimizados |
| F8 | Validación temporal/espacial/externa | Resultados comparativos |
| F9 | Friedman, Wilcoxon, complementarias | Evidencia de diferencias |
| F10 | SHAP, LIME, Grad-CAM, incertidumbre | Explicabilidad |
| F11 | FastAPI, React, PostgreSQL, n8n | Aplicación funcional |
| F12 | Redacción, figuras, tablas, código, anexos | Artículo |

Actualiza `docs/research/protocolo-experimental.md` con estado y evidencia real por fase.

### 6.6 Estrategia de validación (prohibido 80/20 aleatorio simple)

| Tipo de investigación | Validación recomendada |
|---|---|
| Imágenes médicas | Agrupación por paciente + validación externa por hospital |
| Epidemiología | **Validación temporal y geográfica** |
| Agricultura con imágenes | Agrupación por parcela, planta y procedencia |
| Series temporales | Walk-forward |
| Imágenes satelitales | Spatial Block CV |
| Sensores IoT | Temporal y por dispositivo |
| Datos clínicos longitudinales | Separación por paciente y periodo |

- Para datasets pequeños: validación cruzada **agrupada** de 5 particiones, si el número de grupos
  lo permite.
- **Regla fundamental:** imágenes aumentadas de una misma foto, registros de un mismo paciente u
  observaciones de una misma parcela **no pueden** repartirse entre train y test.
- En este proyecto: walk-forward + validación geográfica por distritos no vistos, y demostrar con
  pruebas que no hay fuga (features de `t` solo usan datos `≤ t`; escaladores ajustados solo en
  train).

### 6.7 Optuna

- 50–150 trials iniciales por modelo, ajustables al presupuesto de cómputo.
- **TPE Sampler** + **Median Pruner** o **Hyperband Pruner**.
- Espacio de búsqueda: learning rate, batch size, capas/neuronas, dropout, weight decay,
  estimadores, profundidad de árboles, ventanas temporales, cabezas de atención.
- Optimización **solo** sobre train + validación interna. El test final **no** participa.
- **Presupuesto computacional comparable** entre los 5 modelos; registra trials, tiempo y costo.

### 6.8 Métricas

| Problema | Métricas principales |
|---|---|
| Clasificación médica | Sensibilidad, especificidad, F1, MCC, AUC-ROC, AUC-PR |
| Epidemiología | MAE, RMSE, MASE, WIS, cobertura de intervalos |
| Predicción agrícola | MAE, RMSE, R², MAPE cuando sea válido |
| Detección de objetos | mAP50, mAP50-95, precisión, recall |
| Segmentación | Dice, IoU, Hausdorff |
| Riego inteligente | Ahorro hídrico, rendimiento, eficiencia del agua |
| Optimización hospitalaria | Ocupación, tiempos de espera, utilización |

Para SP-01: epidemiología (MAE, RMSE, MASE, WIS, cobertura) + clasificación de riesgo
(sensibilidad, especificidad, F1, MCC, AUC-ROC, AUC-PR, Brier). Incluye matrices de confusión,
ROC, Precision-Recall, curvas de aprendizaje, mapas de calor y calibración cuando correspondan.
**MAPE solo si es matemáticamente apropiado** (conteos bajos lo hacen inestable).

### 6.9 Pruebas estadísticas (aplicación correcta)

- **Friedman**: comparación de múltiples modelos **sobre bloques de evaluación independientes**
  (p. ej. folds, distritos o semanas), **no** sobre 5 valores agregados de rendimiento.
- **Wilcoxon signed-rank**: comparación pareada de dos modelos.
- **Holm**: corrección por comparaciones múltiples.
- **McNemar**: comparación de errores de clasificación sobre los mismos casos.
- **Bootstrap**: intervalos de confianza de métricas.
- **DeLong**: comparación de curvas ROC cuando se cumplen supuestos.
- **Diebold-Mariano**: precisión comparativa de pronósticos temporales.

Reporta tamaño del efecto, n, supuestos y limitaciones del poder estadístico.

---

## 7. Datos reales (prioridad #1)

### 7.1 Fuentes a integrar

Prioriza fuentes abiertas y verificables:

- **NASA POWER** (meteorología; API pública, sin credenciales).
- **CHIRPS** (precipitación).
- **WorldPop** (población y distribución geográfica).
- **OpenDengue** o **MINSA/CDC Perú – datos abiertos** (casos de dengue reales, idealmente por
  distrito/semana en La Libertad).
- **WHO Global Health Observatory** y **PAHO** como complemento.
- **CDC Dengue Data** si la resolución geográfica lo permite.

### 7.2 Requisitos de la capa de datos

1. Verificar **licencia, condiciones de uso y resolución** de cada fuente antes de integrarla.
2. Guardar una **capa raw inmutable** con checksum SHA-256 y fecha de descarga.
3. Documentar el **diccionario de datos** y la procedencia.
4. Construir el panel **distrito-semana epidemiológica** con denominador poblacional.
5. Documentar faltantes, imputación y anomalías; no ocultarlos.
6. Generar **lags 1–12** sin fuga temporal.
7. Si una fuente exige credenciales: implementar adaptador + variables de entorno + instrucciones
   de activación, y declararlo como limitación.
8. **Mínimo exigible:** NASA POWER + una fuente real de casos de dengue funcionando de punta a
   punta.
9. Mantener el modo **sintético** separado, con semilla fija y etiquetado explícito en BD, API y UI.

### 7.3 Regla de honestidad

Ningún resultado de datos sintéticos puede presentarse como evidencia sobre el dengue real. En
tablas, figuras y reportes debe indicarse la fuente y si es sintética o real.

---

## 8. Plan de ejecución por fases

### FASE 0 — Auditoría y diagnóstico (sin cambios de código todavía)
- Inspecciona repo, ejecuta pruebas, levanta la app si es posible.
- Entrega: inventario real, brechas confirmadas, riesgos, plan y matriz de trazabilidad.
- Solo después de este reporte empieza a implementar.

### FASE 1 — Datos reales
- Adaptadores, descarga, licencias, capa raw, panel distrito-semana, diccionario, calidad.
- Evidencia: archivos descargados, checksums, reporte de calidad, endpoint/consulta que los sirve.

### FASE 2 — Pipeline y features sobre datos reales
- Imputación, anomalías, normalización, lags, particiones temporales/geográficas.
- Pruebas explícitas de **no fuga**.

### FASE 3 — Modelamiento y optimización
- 5 modelos, multihorizonte, Optuna con presupuesto comparable, MLflow activo.
- Guardar artefactos, hiperparámetros, checksums y predicciones de test.

### FASE 4 — Validación, estadística e interpretabilidad
- Walk-forward + geográfica, métricas completas, Friedman sobre bloques, Wilcoxon-Holm,
  McNemar, DeLong, Diebold-Mariano, bootstrap.
- SHAP, importancia temporal, mapas de riesgo, incertidumbre.
- **No declares ganador sin respaldo estadístico.**

### FASE 5 — Prototipo integrado
- API de inferencia, dashboard, mapas, alertas, reportes, trazabilidad.
- MinIO para artefactos/datasets/reportes; MLflow para experimentos.
- Docker Compose con db, backend, frontend, n8n, mlflow, minio.

### FASE 6 — Automatización y capa de lenguaje
- WF-01 … WF-06 importables y verificados.
- LangChain + LangFlow con corpus real (OMS/CDC/científico), herramientas y evaluación con
  conjunto dorado.

### FASE 7 — Calidad, reproducibilidad y publicación
- Pruebas unitarias/integración/endpoints/autorización/auditoría/ML/geospatial.
- Notebook Colab reproducible, informe técnico, matriz de trazabilidad, materiales IMRyD.
- Verificación de que el artículo solo usa resultados ejecutados.

---

## 9. Reglas duras (no negociables)

1. No inventes datos, métricas, resultados, fuentes ni integraciones.
2. Todo dato de demostración se etiqueta **sintético** en BD, API y UI.
3. No uses partición aleatoria 80/20 en series epidemiológicas.
4. No permitas fuga temporal ni geográfica; pruébalo.
5. No declares un modelo superior sin Friedman/Wilcoxon/Diebold-Mariano correctamente aplicados.
6. **No apliques Friedman a 5 promedios agregados**: usa bloques independientes.
7. No uses el test final para seleccionar hiperparámetros.
8. No expongas secretos, tokens ni credenciales; usa `.env`.
9. No rompas pruebas existentes; ejecútalas después de cada cambio.
10. No borres ni sobrescribas trabajo previo sin inspeccionarlo y respaldarlo.
11. No publiques el repo ni despliegues en producción sin autorización expresa.
12. Documenta supuestos, limitaciones y decisiones; distingue requisito de mejora.
13. Cada afirmación de "implementado" debe tener evidencia: archivo, endpoint, prueba o captura.
14. Si una integración no se puede verificar, se declara **no verificada**, no "funcional".

---

## 10. Criterios de aceptación (Definition of Done)

La sesión se considera exitosa solo si:

- [ ] Al menos una fuente real de dengue y NASA POWER alimentan el pipeline de punta a punta.
- [ ] El panel distrito-semana real está disponible vía API y visible en la UI, con procedencia.
- [ ] Los 5 modelos entrenan y comparan sobre datos reales (o se declara por qué no).
- [ ] MLflow registra experimentos y artefactos; MinIO almacena objetos.
- [ ] Existen y se importan WF-01 … WF-06 en n8n, con manejo de errores y logs.
- [ ] LangChain y LangFlow funcionan con corpus real y tienen evaluación con conjunto dorado.
- [ ] Las pruebas estadísticas se aplican sobre bloques independientes y están documentadas.
- [ ] `docker compose up` levanta db, backend, frontend, n8n, mlflow y minio con healthchecks.
- [ ] Las pruebas automatizadas pasan; se reporta el número exacto.
- [ ] README, arquitectura, protocolo y matriz de trazabilidad reflejan el estado real.
- [ ] El ANEXO del profesor está registrado en el repo (p. ej. `docs/research/anexo-profesor.md`).

---

## 11. Formato de reporte al final de cada fase

Informa **solo lo verificado**:

1. Qué se implementó realmente.
2. Archivos creados/modificados (rutas).
3. Comandos ejecutados y resultados (pruebas, descargas, builds).
4. Evidencia (endpoint, tabla, figura, checksum).
5. Limitaciones y riesgos pendientes.
6. Siguiente paso.

No avances a la siguiente fase sin cerrar el reporte de la anterior.

---

## 12. Primera instrucción

Comienza con la **FASE 0**. Inspecciona el repositorio y el ANEXO de `docs/CasoSP01.docx`, ejecuta
las pruebas existentes y presenta:

1. Inventario real del estado del proyecto.
2. Brechas confirmadas frente al anexo.
3. Riesgos técnicos y supuestos.
4. Plan de implementación por fases con criterios de aceptación.
5. Matriz de trazabilidad requisito → módulo → archivo → prueba.

Después del diagnóstico, **continúa implementando sin esperar instrucciones adicionales**, fase por
fase, ejecutando pruebas y documentando la evidencia. Prioridad absoluta: una aplicación integrada,
funcional, segura, científicamente rigurosa y reproducible, operando con datos reales.
