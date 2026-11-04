# Historial de cambios

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto se adhiere
a [Versionado Semántico](https://semver.org/lang/es/).

## [1.0.0] — 2026-11-04

Primera versión funcional del prototipo: plataforma completa de predicción espacio-temporal y alerta
temprana de brotes de dengue.

### Añadido

- **Backend FastAPI** modular por dominio (72 rutas documentadas en OpenAPI), Pydantic v2,
  SQLAlchemy 2 y Alembic con migración PostGIS opcional.
- **Seguridad**: autenticación JWT con refresh y sesiones revocables, bcrypt, bloqueo por intentos,
  RBAC con 7 roles y 27 permisos, política de contraseñas y recuperación con token de un uso.
- **Auditoría**: bitácora encadenada por hash con verificación de manipulación, 13 tipos de evento,
  duración de permanencia por módulo y exportación CSV restringida.
- **Datos**: catálogo de fuentes con licencias, importación CSV/JSON validada fila a fila,
  deduplicación, procedencia, chequeos de calidad y conjuntos de datos versionados con checksum.
- **Pipeline científico**: panel distrito-semana, imputación temporal, rezagos 1–12, variables de
  vecindad y estacionalidad, particiones walk-forward y geográficas sin fuga de información.
- **Modelos**: XGBoost, LSTM, ConvLSTM, CNN-LSTM y ConvLSTM-Attention en PyTorch, con Optuna
  (TPE + Median Pruner), artefactos versionados y predicciones de prueba persistidas.
- **Evaluación**: MAE, RMSE, MAPE, MASE, WIS, cobertura de intervalos, AUC-PR, sensibilidad,
  especificidad y Brier; Friedman, Wilcoxon con corrección de Holm, Diebold-Mariano y bootstrap.
- **Explicabilidad**: SHAP (TreeExplainer / KernelExplainer) con respaldo por permutación,
  importancia temporal y narrativa verificable.
- **Vigilancia**: dashboard ejecutivo, mapas coropléticos de riesgo e incidencia, pronósticos
  multihorizonte, clasificación de riesgo con umbrales configurables y alertas tempranas con
  seguimiento y cierre.
- **Automatización**: workflows WF-01 a WF-04 importables en n8n y ejecución interna equivalente con
  reintentos y registro.
- **Asistente RAG** sobre documentación verificada y hechos de la plataforma, con proveedor
  determinista por defecto y validación de cifras si se conecta un LLM.
- **Reportes**: 16 tipos operacionales y gerenciales en PDF, DOCX y XLSX con metadatos y notas
  metodológicas.
- **Frontend React + TypeScript** (14 módulos) con gráficos y mapa SVG propios, sin dependencias de
  CDN en producción.
- **Despliegue**: Docker Compose con PostgreSQL 16 + PostGIS, nginx y n8n; `.dockerignore` y
  verificación de dependencias en el build.
- **Pruebas**: 27 pruebas automatizadas (autenticación y RBAC, auditoría e integridad, importación y
  calidad, particiones sin fuga, tensores y entrenamiento).
- **Documentación**: arquitectura, modelo de datos, API, manual de usuario, protocolo F1–F12, matriz
  de trazabilidad, informe técnico, notebook reproducible y capturas de pantalla.

### Corregido

- `run_api.py` eliminaba las rutas `site-packages` del `sys.path` también en Docker, dejando la
  imagen sin `uvicorn` ni el resto de dependencias. El aislamiento ahora se aplica solo cuando existe
  `libs/` (modo local con `pip --target`).
- El servidor escuchaba en `127.0.0.1` dentro del contenedor: ahora detecta Docker y usa `0.0.0.0`.
- El wheel de `xgboost` arrastraba `nvidia-nccl-cu12` (351 MB de CUDA inútil en CPU); se instala con
  `--no-deps`, reduciendo la imagen de 3.87 GB a 2.99 GB.

### Limitaciones conocidas

- El conjunto de datos incluido es **sintético**; no sustituye la vigilancia epidemiológica real.
- La cartografía de los distritos es una teselación de Voronoi, no el límite administrativo oficial.
- ConvLSTM y ConvLSTM-Attention están implementados y verificados, pero su corrida completa en CPU
  excede el presupuesto de la sesión de referencia.
- El registro de experimentos es interno; MLflow se activa al instalar la librería.
- La entrega de correo para restablecer contraseñas es manual (no hay SMTP configurado).
