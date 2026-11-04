# API REST · SP-01 Dengue Intelligence

- **Base**: `http://127.0.0.1:8000/api/v1`
- **Documentación interactiva**: `/api/docs` (Swagger UI) · `/api/redoc` · `/api/openapi.json`
- **Autenticación**: `Authorization: Bearer <access_token>`
- **Formato de error**:
  ```json
  { "error": { "code": "ACCESS_DENIED", "message": "No cuenta con permisos para esta operación.",
               "details": { "missing": ["users:write"] } } }
  ```
- **Cabeceras de respuesta**: `X-Request-ID`, `X-Response-Time`.

## 1. Autenticación (M01)

| Método | Ruta | Permiso | Descripción |
|---|---|---|---|
| POST | `/auth/login` | público | Devuelve `access_token`, `refresh_token`, `expires_in`, `must_change_password` |
| POST | `/auth/refresh` | refresh token | Renueva el token de acceso |
| POST | `/auth/logout` | autenticado | Revoca la sesión actual |
| GET | `/auth/me` | autenticado | Perfil, roles y permisos efectivos |
| POST | `/auth/change-password` | autenticado | Cambio con verificación de política |
| POST | `/auth/forgot-password` | público | Genera token de un uso (entrega manual sin SMTP) |
| POST | `/auth/reset-password` | público | Consume el token |
| GET | `/auth/sessions` | autenticado | Sesiones del usuario con actividad |
| DELETE | `/auth/sessions/{public_id}` | autenticado | Revoca una sesión |

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@dengai.unt.edu.pe","password":"Admin#Dengue2025"}'
```

## 2. Usuarios, roles y auditoría (M02, M03)

| Método | Ruta | Permiso |
|---|---|---|
| GET/POST | `/users` | `users:read` / `users:write` |
| GET/PATCH | `/users/{id}` | `users:read` / `users:write` |
| POST | `/users/{id}/status` | `users:write` |
| POST | `/users/{id}/force-logout` | `users:write` |
| GET/POST | `/roles` | `roles:read` / `roles:write` |
| PATCH | `/roles/{id}` | `roles:write` |
| GET | `/roles/permissions/catalog` | `roles:read` |
| GET | `/audit` | `audit:read` (`event_type`, `module`, `result`, `user_email`, `search`, `start`, `end`, `page`, `size`) |
| GET | `/audit/stats` | `audit:read` |
| GET | `/audit/verify` | `audit:read` — recalcula la cadena de hashes |
| GET | `/audit/export` | `audit:export` — CSV con BOM |
| POST | `/audit/module-event` | autenticado — `MODULE_ENTER` / `MODULE_EXIT` con duración calculada |

## 3. Dashboard, mapas y pronósticos (M04, M07, M10)

| Método | Ruta | Permiso | Notas |
|---|---|---|---|
| GET | `/dashboard/overview` | `dashboard:read` | KPIs, serie, top incidencia/riesgo, clima, integraciones, calidad |
| GET | `/geo/units` | `geo:read` | Catálogo geográfico |
| GET | `/geo/geojson` | `geo:read` | `FeatureCollection` |
| GET | `/geo/risk-map` | `geo:read` | Riesgo vigente + leyenda + umbrales |
| GET | `/geo/incidence-map?weeks=4` | `geo:read` | Incidencia acumulada |
| GET | `/geo/territory/{code}` | `geo:read` | Serie histórica del distrito |
| POST | `/forecasts/generate` | `predictions:run` | `{horizons: [1,2], district_codes: [...]}` |
| GET | `/forecasts/latest` | `predictions:read` | Última ejecución persistida |
| GET | `/risk/current` | `predictions:read` | Matriz de riesgo con resumen |

## 4. Datos y conjuntos (M05, M06, M16)

| Método | Ruta | Permiso |
|---|---|---|
| GET | `/data/sources` | `data:read` |
| POST | `/data/sources/{code}/probe` | `data:import` |
| POST | `/data/import/preview` | `data:import` (multipart: `file`, `kind`) |
| POST | `/data/import/cases` | `data:import` (multipart: `file`, `source_code`, `is_synthetic`) |
| POST | `/data/import/climate` | `data:import` |
| GET | `/data/imports` · `/data/imports/{id}/errors` | `data:read` |
| GET | `/data/quality` | `data:read` |
| GET | `/data/cases` · `/data/climate` | `data:read` |
| POST | `/datasets/build` | `datasets:write` |
| GET | `/datasets` · `/datasets/{id}` · `/datasets/{id}/dictionary` | `datasets:read` |

Ejemplo de importación:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/data/import/cases \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@casos.csv" -F "source_code=MINSA_VIGILANCIA" -F "is_synthetic=false"
```

## 5. Laboratorio de IA (M07, M08, M09)

| Método | Ruta | Permiso |
|---|---|---|
| GET | `/ml/algorithms` · `/ml/backend` | `ml:read` |
| POST | `/ml/experiments` | `ml:run` |
| GET | `/ml/experiments` · `/ml/experiments/{id}` | `ml:read` |
| GET | `/ml/models` | `ml:read` |
| POST | `/ml/models/{id}/promote` | `ml:promote` |
| GET | `/ml/metrics?horizon=1` | `ml:read` |
| GET | `/ml/compare?horizon=1` | `ml:read` |
| GET | `/ml/explain/{district_code}?horizon=1` | `explain:read` |

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/ml/experiments \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"algorithm":"xgboost","n_trials":50,"horizons":[1,2],"validation_strategy":"walk_forward"}'
```

Estados del experimento: `QUEUED → RUNNING (stage: horizonte_h_datos → optuna_k/n → horizonte_h_entrenamiento_final) → COMPLETED | FAILED`.

## 6. Alertas, reportes, asistente y workflows (M11–M14)

| Método | Ruta | Permiso |
|---|---|---|
| GET | `/alerts` · `/alerts/summary` | `alerts:read` |
| POST | `/alerts/generate` | `alerts:write` |
| POST | `/alerts/{id}` | `alerts:write` (`action`: `acknowledge` \| `close` \| `reopen`) |
| GET | `/reports/types` · `/reports` | `reports:read` |
| POST | `/reports` | `reports:generate` |
| GET | `/reports/{id}/download` | `reports:read` |
| DELETE | `/reports/{id}` | `reports:generate` |
| POST | `/assistant/ask` | `assistant:ask` |
| GET | `/assistant/documents` · `/assistant/history` | `assistant:ask` |
| GET | `/workflows` · `/workflows/files` · `/workflows/executions` | `workflows:read` |
| POST | `/workflows/{code}/run` | `workflows:run` |

## 7. Administración (M15)

| Método | Ruta | Permiso |
|---|---|---|
| GET | `/health` | público |
| GET | `/admin/config` | `admin:config` |
| PUT | `/admin/config/{key}` | `admin:config` |
| GET | `/admin/stats` | `admin:config` |

## 8. Códigos de estado

| Código | `error.code` | Significado |
|---|---|---|
| 400 | `APP_ERROR` | Regla de negocio incumplida |
| 401 | `AUTH_ERROR` | Token ausente/expirado o credenciales inválidas |
| 403 | `ACCESS_DENIED` | Permiso insuficiente |
| 404 | `NOT_FOUND` | Recurso inexistente |
| 409 | `CONFLICT` | Duplicado o ejecución concurrente |
| 422 | `VALIDATION_ERROR` | Datos o invariantes inválidas (incluye "no hay modelo en producción") |
| 429 | `TOO_MANY_REQUESTS` | Cuenta bloqueada por intentos fallidos |
| 503 | `SERVICE_UNAVAILABLE` | Dependencia externa no disponible |
