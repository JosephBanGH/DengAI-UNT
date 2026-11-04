# Modelo de datos y PostGIS

## 1. Visión general

El esquema está normalizado, con claves primarias sustitutas (`id`), claves naturales únicas
(`code`, `email`, `slug`, `run_id`, `report_key`, `alert_key`) y claves foráneas con política de
borrado explícita (`CASCADE`, `SET NULL`).

Motor por defecto: **SQLite** (desarrollo, sin infraestructura). Motor de producción:
**PostgreSQL 16 + PostGIS 3.4**.

## 2. Entidades por dominio

### Seguridad y auditoría (`models/security.py`)
| Tabla | Propósito | Restricciones |
|---|---|---|
| `users` | Cuentas | `email` único, `password_hash`, bloqueo por intentos, `must_change_password` |
| `roles` | 7 roles del protocolo | `slug` único, `is_system` |
| `permissions` | 27 permisos por módulo | `code` único |
| `role_permissions` / `user_roles` | Asociaciones N:M | PK compuesta, `CASCADE` |
| `user_sessions` | Sesiones activas | `public_id` opaco, hash del refresh, `revoked_at` |
| `login_attempts` | Intentos de acceso | Índice por correo y fecha |
| `audit_events` | Bitácora encadenada | `hash` y `prev_hash` (SHA-256) |
| `system_config` | Configuración en BD | `key` única, valor JSON |

### Geografía y calendario (`models/geo.py`)
| Tabla | Propósito |
|---|---|
| `geographic_units` | Distritos con centroide, altitud, área, población y `geometry_geojson` |
| `epi_weeks` | Semanas epidemiológicas ISO con fecha de inicio y fin |

### Epidemiología y datos (`models/epi.py`)
| Tabla | Propósito |
|---|---|
| `dengue_cases` | Casos por distrito-semana (**único** `district_code + epi_year + epi_week`) |
| `climate_observations` | Temperatura, humedad, precipitación, viento, radiación |
| `environmental_variables` | Índices adicionales (densidad poblacional u otros) |
| `data_sources` | Catálogo con licencia, método de acceso y estado de integración |
| `import_jobs` / `import_row_errors` | Trazabilidad de importaciones y errores fila a fila |
| `data_quality_records` | Chequeos de faltantes, duplicados y anomalías |
| `datasets` | Versiones del panel con `checksum`, esquema y diccionario de datos |

### Machine learning (`models/ml.py`)
| Tabla | Propósito |
|---|---|
| `experiments` | Run, algoritmo, estado, progreso, trials, métricas, hiperparámetros, `mlflow_run_id` |
| `model_versions` | Versión, artefacto, checksum, estado, `is_production` |
| `predictions` | Pronóstico por distrito, origen y objetivo con intervalos |
| `risk_classifications` | Nivel de riesgo por distrito-semana con justificación |
| `metric_results` / `statistical_tests` | Métricas e inferencia estadística |

### Operaciones (`models/ops.py`)
`alerts`, `reports`, `workflow_executions`, `assistant_queries`.

## 3. Relaciones principales

```
geographic_units 1 ─── N dengue_cases
geographic_units 1 ─── N climate_observations
geographic_units 1 ─── N predictions
experiments      1 ─── N model_versions
model_versions   1 ─── N predictions
data_sources     1 ─── N dengue_cases / climate_observations
import_jobs      1 ─── N import_row_errors
users            1 ─── N user_sessions / audit_events
roles            N ─── N permissions (role_permissions)
users            N ─── N roles (user_roles)
```

## 4. Índices y restricciones relevantes

- `uq_dengue_cases_district_week` y `uq_climate_district_week`: impiden duplicar la unidad de análisis.
- `ix_dengue_cases_week`, `ix_predictions_district_target`, `ix_alerts_district_week`: consultas
  territoriales y temporales.
- `ix_audit_events_type_time` / `ix_audit_events_user_time`: filtros de bitácora.
- `uq_model_versions_algorithm_version`: evita colisiones de versión.

## 5. PostGIS

En SQLite la geometría se guarda como GeoJSON (`geometry_geojson`) y el frontend la proyecta a SVG.
Con `POSTGIS_ENABLED=true` y PostgreSQL, la migración `0002_postgis`:

1. crea la extensión `postgis`;
2. añade `geom geometry(MultiPolygon, 4326)`;
3. rellena la geometría desde el GeoJSON existente;
4. crea un índice GiST;
5. instala el trigger `trg_sync_geographic_units_geom` que mantiene `geom` sincronizado en cada
   `INSERT/UPDATE` de `geometry_geojson`.

Consultas espaciales habilitadas: intersección con áreas de vigilancia, distancias entre distritos,
vecindad real (`ST_Touches`) para reemplazar el proxy de "distritos vecinos" usado en las variables.

## 6. Migraciones

```bash
cd backend
alembic upgrade head          # 0001_initial + 0002_postgis
alembic revision --autogenerate -m "descripcion"
```

`env.py` toma `DATABASE_URL` de la configuración y usa `render_as_batch` en SQLite para permitir
`ALTER TABLE`.

## 7. Respaldo y recuperación

- **PostgreSQL**: `pg_dump -Fc dengai > backup.dump` programado; `pg_restore` para recuperar.
- **SQLite**: copia del archivo `data/dengai.db` con la aplicación detenida (o `.backup`).
- **Artefactos**: `data/models` y `data/reports` deben incluirse en el respaldo; su pérdida invalida
  las comparaciones históricas aunque las métricas permanezcan en la base.
- **Rotación**: conservar al menos los últimos 30 días y una copia mensual; cifrar en reposo.
