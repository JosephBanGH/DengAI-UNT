# Matriz de trazabilidad · requisito → módulo → implementación → prueba

Cada fila del protocolo se vincula con el componente que la implementa y la evidencia que permite
verificarla. Las rutas son relativas a la raíz del repositorio.

## 1. Requisitos funcionales (M01–M16)

| ID | Requisito | Módulo | Implementación | Prueba / evidencia |
|---|---|---|---|---|
| RF-01 | Inicio y cierre de sesión | M01 | `services/auth_service.login/logout`, `api/v1/auth.py` | `tests/test_auth.py::test_login_accepts_valid_credentials…`, `test_logout_revokes_session` |
| RF-02 | Gestión segura de contraseñas | M01 | `core/security.hash_password/validate_password_policy` | `test_auth.py` (login inválido), política en `change_password` |
| RF-03 | Recuperación de contraseña | M01 | `request_password_reset` / `reset_password` (token de un uso) | Flujo cubierto por API; entrega manual documentada |
| RF-04 | Control de sesiones y expiración | M01 | `UserSession`, refresh con hash, revocación | `test_refresh_token_rotates_access_token`, `test_logout_revokes_session` |
| RF-05 | RBAC de mínimo privilegio | M02 | 7 roles, 27 permisos, `require_permissions` | `test_consultation_role_cannot_manage_users` |
| RF-06 | Administración de usuarios | M02 | `services/user_service.py` | `tests/test_auth.py`, UI Usuarios |
| RF-07 | Bitácora obligatoria | M03 | `services/audit.py` (cadena de hash), `api/v1/module_audit.py` | `tests/test_audit.py` (5 pruebas) |
| RF-08 | Panel de auditoría consultable y exportable | M03 | `/audit`, `/audit/stats`, `/audit/export` | `test_audit_event_includes_hash_and_previous_hash` |
| RF-09 | Dashboard con indicadores | M04 | `services/dashboard_service.overview` | `test_consultation_role_can_read_dashboard`, captura `01-dashboard.png` |
| RF-10 | Gestión e integración de fuentes | M05 | `data_service` + `integrations/nasa_power.py` | `tests/test_data_import.py` (5 pruebas) |
| RF-11 | Pipeline reproducible sin fuga | M06 | `ml/features.build_features/split_*` | `test_features_do_not_use_future_information`, `test_temporal_split_respects_chronology` |
| RF-12 | Cinco algoritmos | M07 | `ml/models/{xgboost,torch}_forecasters.py`, `registry.py` | `test_xgboost_trains_and_predicts`, experimentos reales en la plataforma |
| RF-13 | Optuna 50–150 trials | M07 | `trainer._optimize` (TPE + MedianPruner) | Campo `n_trials`, `trials_completed`, `search_space` en `experiments` |
| RF-14 | Pronóstico multihorizonte | M07 | `ml/inference.generate_forecast` | Endpoint `/forecasts/generate`; `predictions` persistidas |
| RF-15 | Clasificación de riesgo | M07/M11 | `inference.classify_risk` + `system_config.risk_thresholds` | `risk_classifications`, `/risk/current` |
| RF-16 | Validación temporal y geográfica | M08 | `split_temporal`, `split_geographic` | `test_geographic_split_uses_unseen_districts` |
| RF-17 | Métricas del protocolo | M08 | `ml/metrics.py` | `test_metrics_are_finite_and_scaled` |
| RF-18 | Pruebas estadísticas | M08 | `ml/stats_tests.py`, `comparison_service` | `/ml/compare` con Friedman/Wilcoxon/DM |
| RF-19 | Interpretabilidad | M09 | `ml/explain.py` (SHAP + permutación) | `/ml/explain/{code}`, UI «Explicar» |
| RF-20 | Mapas interactivos | M10 | `geo_service`, `components/Choropleth.tsx` | Captura `mapa.png`, `/geo/risk-map` |
| RF-21 | Alertas tempranas | M11 | `services/alerts_service.py` | `/alerts/generate`, `alerts` con `alert_key` único |
| RF-22 | Workflows WF-01..WF-04 | M12 | `n8n/workflows/*.json`, `services/workflows_service.py` | `/workflows`, `/workflows/{code}/run`, `workflow_executions` |
| RF-23 | Asistente RAG | M13 | `services/assistant_service.py` | `/assistant/ask` con fuentes citadas |
| RF-24 | Reportes PDF/DOCX/XLSX | M14 | `services/reports_service.py` | 16 tipos; descarga verificada |
| RF-25 | Configuración del sistema | M15 | `system_config`, `/admin/config` | Validación `low < moderate < high` |
| RF-26 | Reproducibilidad científica | M16 | `datasets`, diccionario, notebook | `dataset_service`, `ml/notebooks/` |

## 2. Requisitos no funcionales

| ID | Requisito | Evidencia |
|---|---|---|
| RNF-01 | Separación de responsabilidades | `api/` ↔ `services/` ↔ `ml/` ↔ `models/`; ningún archivo monolítico |
| RNF-02 | Seguridad de endpoints | Todos los routers dependen de `require_permissions`; `/health` es el único público junto con `auth/login` |
| RNF-03 | Validación de entradas | Esquemas Pydantic en cada endpoint; `RequestValidationError` homogéneo |
| RNF-04 | Protección contra inyección | SQLAlchemy parametrizado; sin SQL concatenado |
| RNF-05 | Gestión de secretos | `.env` + `.env.example`; `SECRET_KEY` autogenerada y nunca impresa |
| RNF-06 | Errores consistentes | `AppError` → `{"error": {code, message, details}}` |
| RNF-07 | Rendimiento percibido | Entrenamiento asíncrono con progreso persistido; UI con sondeo |
| RNF-08 | Diseño responsive | Rejillas CSS adaptativas; verificado a 1480×1000 y 390×844 |
| RNF-09 | Accesibilidad básica | Roles ARIA en tabs, `aria-label` en mapas y modales, contraste AA en textos principales |
| RNF-10 | Documentación | README, arquitectura, base de datos, API, manual, protocolo, informe técnico |

## 3. Requisitos científicos del documento fuente

| Requisito del caso SP-01 | Implementación | Estado |
|---|---|---|
| Unidad distrito-semana | `dengue_cases` con restricción única | Cumplido |
| Variables: temperatura, humedad, precipitación, altitud, densidad, incidencia histórica, índices ambientales, estacionalidad | `climate_observations`, `environmental_variables`, `build_features` | Cumplido (datos sintéticos) |
| Rezagos 1–12 semanas | `cases_lag_1..12`, `*_lag_0..4` | Cumplido |
| Pronóstico multihorizonte + clasificación | `predictions`, `risk_classifications` | Cumplido |
| Validación por distritos | `split_geographic` | Cumplido |
| SHAP, importancia temporal, mapas de riesgo | `explain.py`, `Choropleth` | Cumplido |
| MAE, RMSE, MAPE, MASE, WIS, cobertura, AUC-PR, sensibilidad, especificidad, Brier | `metrics.py` | Cumplido |
| Friedman, Wilcoxon-Holm, Diebold-Mariano, bootstrap | `stats_tests.py` | Cumplido |
| Optuna 50–150 trials con TPE y pruner | `trainer._optimize` | Cumplido (reducible por CPU) |
| n8n WF-01..WF-04 | `n8n/workflows/` + ejecución interna | Cumplido |
| LangChain/LangFlow y RAG | `assistant_service` (RAG propio, proveedor LLM opcional) | Parcial: el grafo LangFlow no se incluye; la arquitectura RAG sí |
| PostgreSQL + PostGIS | SQLAlchemy + migración `0002_postgis` | Parcial: SQLite en local, PostgreSQL listo para Docker |
| MLflow | `trainer._track_mlflow` | Parcial: registro interno; MLflow se activa al instalarlo |

## 4. Requisitos no especificados que se resolvieron con supuestos

| Vacío | Decisión | Justificación |
|---|---|---|
| Distritos concretos | 14 distritos de La Libertad con centroides y población de referencia | El caso menciona Perú/La Libertad sin listar distritos |
| Umbrales de riesgo | Incidencia semanal por 100 000 hab. (5/20/50) configurables | Criterio epidemiológico estándar y parametrizable; justificado en `system_config` |
| Ventanas de validación | 26 semanas de validación y 26 de prueba | Suficiente para capturar estacionalidad anual parcial con datos de 8 años |
| Entrega de correo | Token de restablecimiento devuelto para entrega manual | No hay SMTP configurado; se documenta la limitación |
| Cartografía | Teselación de Voronoi | No se dispone del shapefile oficial; se advierte en la interfaz |
