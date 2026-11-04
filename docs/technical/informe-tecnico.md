# Informe técnico del prototipo

**Proyecto**: SP-01 Dengue Intelligence · Sistema Inteligente de Predicción Espacio-Temporal y
Alerta Temprana de Brotes de Dengue
**Versión**: 1.0.0 · **Entorno de referencia**: Windows 11, Python 3.12.14, Node 22, CPU de 8 hilos

---

## 1. Resumen ejecutivo

Se construyó una plataforma web funcional que cubre el ciclo completo de vigilancia predictiva:
ingesta y validación de datos distrito-semana, ingeniería de variables sin fuga de información,
entrenamiento y optimización de cinco algoritmos (tres individuales y dos híbridos), evaluación con
métricas epidemiológicas y pruebas estadísticas, clasificación de riesgo con umbrales configurables,
mapas territoriales, alertas tempranas, reportes PDF/DOCX/XLSX, asistente RAG y bitácora de auditoría
encadenada.

Todo el software se ejecutó y verificó en este entorno: **27 pruebas automatizadas en verde**, un
ciclo completo de entrenamiento → promoción → pronóstico → alerta → reporte, y verificación visual
de las pantallas con un navegador real.

## 2. Alcance implementado

| Componente | Estado | Detalle |
|---|---|---|
| API REST (FastAPI) | Operativo | 81 rutas, OpenAPI en `/api/docs` |
| Frontend SPA (React + TS) | Operativo | 14 módulos, compilado y servido por FastAPI |
| Base de datos | Operativa | SQLite local; PostgreSQL/PostGIS por Docker + migración |
| Seguridad | Operativa | JWT + refresh, bcrypt, RBAC (7 roles / 27 permisos), bloqueo por intentos |
| Auditoría | Operativa | Cadena SHA-256 verificable, 13 tipos de evento, exportación CSV |
| Datos | Operativo | Catálogo con licencias, importación CSV/JSON validada, calidad |
| IA | Operativo | 5 algoritmos, Optuna, artefactos con checksum, comparación estadística |
| Explicabilidad | Operativa | SHAP con respaldo por permutación, narrativa verificable |
| Alertas | Operativas | Generación idempotente, seguimiento y cierre |
| Reportes | Operativos | 16 tipos × 3 formatos |
| Orquestación | Operativa | 4 workflows n8n importables + ejecución interna con reintentos |
| Asistente | Operativo | RAG sobre corpus interno + hechos; proveedor determinista |

## 3. Verificación ejecutada

### 3.1 Pruebas automatizadas

```
$ python -m pytest backend/tests -q
...........................                                              [100%]
```

Cubren: login válido/inválido, refresh y revocación, RBAC (consulta no administra), integridad y
detección de manipulación de la bitácora, eventos de módulo con duración, previsualización e
importación de datos (filas válidas y rechazadas, permisos), orden cronológico de particiones,
ausencia de fuga (objetivo en t+h), validación geográfica con distritos disjuntos, métricas finitas,
entrenamiento y predicción de XGBoost, y consistencia de formas de los tensores.

### 3.2 Ciclo funcional completo (API real)

| Paso | Resultado obtenido |
|---|---|
| Login administrador | 200 · `must_change_password=true` |
| Dashboard | 200 · KPIs, 53 puntos de serie, 10 distritos por incidencia |
| Mapa de riesgo | 200 · 14 features con nivel y score |
| Algoritmos | 5 disponibles (`READY`) con PyTorch 2.9.1+cpu y XGBoost 3.0.5 |
| Entrenamiento XGBoost h=1 (4 trials) | `COMPLETED` · MAE 3.497 · RMSE 11.416 · MASE 0.858 · cobertura 0.797 · AUC-PR 0.419 |
| Entrenamiento LSTM h=1 (2 trials) | `COMPLETED` · MAE 3.631 · RMSE 10.943 · cobertura 0.821 |
| Promoción | 200 · versión en producción por horizonte |
| Pronóstico h=1 | 200 · 14 distritos con intervalo e incidencia |
| Alertas | 3 alertas creadas a partir de pronósticos |
| Explicación SHAP | 200 · factores ordenados por contribución + narrativa |
| Reportes | PDF 4 011 B, DOCX y XLSX generados y descargados |
| Asistente | 200 · respuesta con 5 fuentes citadas |
| Auditoría | Cadena íntegra verificada |
| Workflow WF-02 | `COMPLETED` · 5 740 ítems procesados |

### 3.3 Verificación visual

Capturas automáticas con Chrome headless sobre las 11 pantallas principales: sin errores de consola,
sin excepciones de página, 14 elementos de navegación renderizados y banner de datos sintéticos
presente.

## 4. Decisiones técnicas y su justificación

1. **FastAPI + Pydantic v2 + SQLAlchemy 2**: tipado fuerte, OpenAPI automático y sesiones limpias.
2. **Servicios por dominio**: la lógica de auditoría, alertas, reportes y RAG no vive en los routers.
3. **Optuna con TPE y MedianPruner**: exploración eficiente y poda temprana, aislada de prueba.
4. **Intervalos por cuantiles de validación**: evita la sobreconfianza de los intervalos in-sample.
5. **Riesgo por incidencia poblacional**: comparable entre distritos de tamaños muy distintos.
6. **Bitácora encadenada**: evidencia criptográfica de no manipulación sin infraestructura externa.
7. **RAG determinista por defecto**: elimina el riesgo de cifras inventadas y funciona sin API key.
8. **Frontend sin dependencias de UI pesadas**: gráficos y mapa en SVG propio, arranque rápido y sin
   llamadas a CDN en producción.

## 5. Rendimiento observado

| Operación | Tiempo |
|---|---|
| Arranque (creación de tablas + poblado) | ≈ 0.5 s |
| Dashboard (5 740 registros) | < 300 ms |
| Entrenamiento XGBoost, 4 trials, h=1 | ≈ 45 s |
| Entrenamiento LSTM, 2 trials, h=1 | ≈ 108 s |
| Pronóstico h=1 (14 distritos) | ≈ 2 s |
| Reporte PDF con 6 secciones | ≈ 0.4 s |
| Build del frontend | 1.65 s (59 módulos, 281 kB / 86 kB gzip) |

## 6. Limitaciones y trabajo pendiente

1. **Datos**: el conjunto es sintético; falta cargar MINSA/CDC y verificar términos de uso.
2. **Cartografía**: reemplazar la teselación por el shapefile oficial del INEI y activar PostGIS.
3. **Integraciones en vivo**: `LIVE_INTEGRATIONS_ENABLED=1` y credenciales para CHIRPS/WorldPop;
   NASA POWER ya tiene adaptador funcional sin clave.
4. **LangFlow**: se implementó la arquitectura RAG y el agente, no el grafo visual de LangFlow.
5. **MLflow**: el registro es interno; al instalar `mlflow-skinny` se activa el tracking automático.
6. **n8n**: los JSON están listos para importar; la instancia de n8n corre por Docker.
7. **Escalabilidad**: el entrenamiento ocurre en hilos del proceso API; en producción conviene
   separarlo en una cola de tareas (Celery/RQ) con workers dedicados.
8. **Pruebas de frontend**: la verificación fue visual y funcional, sin suite unitaria de
   componentes (Jest/Vitest) todavía.

## 7. Conclusiones

El prototipo cumple el objetivo del proyecto: no es una maqueta ni una colección de fragmentos, sino
una aplicación integrada, reproducible y verificada, con trazabilidad requisito → implementación →
prueba. Permite entrenar modelos reales, explicar sus predicciones, emitir alertas con fundamento y
generar reportes institucionales, dejando declaradas de forma explícita las limitaciones de datos y
de infraestructura que debe resolverse antes de un despliegue en producción.
