# Arquitectura de SP-01 Dengue Intelligence

## 1. Principios de diseño

| Principio | Aplicación concreta |
|---|---|
| Separación de responsabilidades | `api/` solo traduce HTTP; `services/` contiene reglas de negocio; `ml/` contiene ciencia; `models/` contiene persistencia |
| Módulos por dominio | Un router por módulo funcional (M01–M15), sin archivos monolíticos |
| Dependencias explícitas | Inyección de `Session`, `AuthContext` y `RequestContext` mediante `Depends` |
| Errores tipados | Jerarquía `AppError` → `code`, `status_code`, `details`; un único manejador responde `{"error": {...}}` |
| Reproducibilidad | Semillas fijas, checksums de artefactos, particiones deterministas |
| Honestidad metodológica | El sistema se niega a responder sin modelo en producción o sin datos; marca los datos sintéticos en toda la interfaz |

## 2. Capas

```
Frontend (React 18 + TS)
   │  fetch + JWT (access/refresh), cliente único en src/api/client.ts
   ▼
FastAPI (backend/app/main.py)
   ├─ Middleware: request-id, tiempos, cabeceras de seguridad, CORS
   ├─ Manejadores: AppError, RequestValidationError, HTTPException
   ├─ api/v1/*: routers por módulo
   ▼
Servicios (backend/app/services)
   ├─ auth_service · user_service · audit (cadena de hash)
   ├─ data_service · dataset_service (importación, calidad, versionado)
   ├─ dashboard_service · geo_service · alerts_service
   ├─ reports_service (PDF/DOCX/XLSX) · assistant_service (RAG) · workflows_service
   ▼
Dominio científico (backend/app/ml)
   ├─ features: panel distrito-semana, rezagos 1–12, tensores (T, H, W, C)
   ├─ models: XGBoost, LSTM, ConvLSTM, CNN-LSTM, ConvLSTM-Attention
   ├─ trainer: Optuna (TPE + MedianPruner), artefactos, registro
   ├─ metrics / stats_tests / explain (SHAP + narrativa)
   ▼
Persistencia (SQLAlchemy 2)
   ├─ SQLite (desarrollo) · PostgreSQL 16 + PostGIS (producción)
   └─ data/models (joblib + npz) · data/reports (PDF/DOCX/XLSX)
```

## 3. Flujo de un pronóstico

1. `POST /api/v1/forecasts/generate` con horizontes y filtros opcionales.
2. `ml.inference.pick_version` selecciona la versión **en producción** para cada horizonte; si no
   existe, responde `422` con instrucciones (no inventa resultados).
3. Se reconstruye el panel con `load_panel` y las variables con `build_features` (rezagos ≤ t).
4. Se construyen las ventanas y el tensor espacio-temporal con `make_sample_tensors`, aplicando el
   escalador guardado en el artefacto.
5. El modelo predice; los intervalos se obtienen sumando los cuantiles residuales de validación
   almacenados en el artefacto.
6. Se calcula la incidencia por 100 000 habitantes y se clasifica el riesgo con los umbrales
   configurados en `system_config.risk_thresholds`.
7. Se persisten `predictions` y `risk_classifications`; `POST /alerts/generate` crea las alertas
   (idempotente por `alert_key`).
8. Todo queda en la bitácora (`MODEL_EXECUTION`, `CREATE`, `EXPORT`…).

## 4. Flujo de entrenamiento

1. `POST /api/v1/ml/experiments` valida algoritmo disponible y presupuesto de trials.
2. Se crea el registro `QUEUED` y se lanza un hilo (`submit_experiment`) con progreso persistido.
3. Por horizonte: `build_dataset` → Optuna (solo train/val) → reentreno con train+val → evaluación
   única sobre test → métricas + IC bootstrap → artefacto + `model_versions` (CANDIDATE).
4. El usuario promueve una versión; promover demota las demás del **mismo horizonte** para que
   `pick_version` sea determinista.
5. `GET /ml/compare` recalcula ranking y pruebas estadísticas desde las predicciones persistidas.

## 5. Orquestación

Los cuatro workflows operacionales existen en dos formas equivalentes:

- **n8n** (`n8n/workflows/*.json`): importables, con Schedule Trigger, autenticación, reintentos y
  notificación de fallos.
- **Ejecución interna** (`services/workflows_service.py`): los mismos pasos con registro en
  `workflow_executions`, reintento automático y estado `SKIPPED` cuando falta una precondición
  (p. ej. sin modelo en producción o integraciones externas desactivadas).

Esto permite operar la plataforma sin el orquestador externo sin duplicar lógica de negocio.

## 6. Extensibilidad

- **Nuevo algoritmo**: implementar `BaseForecaster` en `ml/models/`, registrarlo en
  `registry.ALGORITHMS` con su `SEARCH_SPACES` y queda disponible en API y en el Laboratorio.
- **Nueva fuente**: añadir un adaptador en `integrations/`, sembrar su registro en `data_sources`
  (con licencia) y exponer la importación por archivo o API.
- **Nuevo reporte**: declararlo en `reports_service.REPORT_TYPES` y añadir su bloque en `_collect`.

## 7. Riesgos técnicos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Fuga de información en series temporales | Particiones temporales/geográficas y pruebas automáticas que verifican el orden cronológico |
| Sobreajuste del modelo ganador | Prueba independiente + pruebas de Friedman/Wilcoxon/Diebold-Mariano |
| Datos sintéticos confundidos con reales | Marca `is_synthetic` en BD, banner permanente, advertencia en reportes y en el disclaimer de la API |
| Ausencia de PyTorch | Los algoritmos neuronales se marcan `UNAVAILABLE` y la plataforma sigue operando con XGBoost |
| Coste computacional de ConvLSTM en CPU | Trials reducidos, `max_epochs` con parada temprana y límite de hilos |
| Pérdida de trazabilidad | Bitácora encadenada + checksums + versionado de conjuntos y modelos |
