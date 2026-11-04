# Manual de usuario · SP-01 Dengue Intelligence

Este manual está dirigido al personal de vigilancia epidemiológica. Cada pantalla indica el módulo
del protocolo al que contribuye y qué permisos se requieren.

---

## 1. Acceso

1. Abra `http://127.0.0.1:8000` (o la URL institucional).
2. Ingrese correo y contraseña asignados. El primer acceso marca la cuenta con
   `must_change_password`.
3. En el menú superior (esquina derecha) → **Mi perfil y sesiones** → cambie la contraseña.
4. El banner naranja superior indica que los datos son sintéticos de demostración. Desaparece
   cuando la base contiene únicamente datos de fuentes externas.

**Roles y alcance**

| Rol | Puede hacer |
|---|---|
| Administrador del sistema | Todo, incluida la gestión de usuarios, roles y configuración |
| Administrador epidemiológico | Datos, modelos, alertas, reportes y configuración |
| Epidemiólogo | Analizar situación, gestionar alertas, generar reportes |
| Analista de datos | Importar datos, construir conjuntos, ejecutar experimentos |
| Investigador | Reproducir experimentos y documentar resultados |
| Auditor | Consultar y exportar la bitácora; sin capacidad de modificar datos |
| Usuario de consulta | Solo lectura de dashboard, mapas, pronósticos, alertas y reportes |

---

## 2. Dashboard ejecutivo (M04)

- **KPIs**: casos de 12 semanas con variación respecto a las 12 previas, casos de la última semana,
  alertas activas, distritos en riesgo alto/crítico, modelos en producción y proporción de datos
  sintéticos.
- **Evolución temporal**: serie semanal (2 años) y media móvil de 4 semanas; pase el cursor para ver
  el valor exacto.
- **Distribución de riesgo**: dona con el conteo por nivel vigente.
- **Top incidencia / Top riesgo**: tablas ordenadas; el botón *Ver mapa* lleva al módulo territorial.
- **Paneles inferiores**: clima de las últimas 8 semanas frente al baseline, estado de integraciones,
  rendimiento del modelo en producción (MAE, RMSE, AUC-PR, cobertura) y operación de la plataforma.

> Si el dashboard muestra "Sin modelos en producción", entrene y promueva un modelo (sección 5).

---

## 3. Mapas y análisis territorial (M10)

1. Elija **Riesgo pronosticado** o **Incidencia histórica**.
2. Ajuste la **ventana histórica** (4, 8, 12 o 26 semanas).
3. Pase el cursor sobre un distrito: verá nivel de riesgo, casos pronosticados, población e
   incidencia por 100 000 habitantes.
4. Haga clic para fijar el distrito en **Detalle del distrito** (provincia, población, altitud,
   incidencia pronosticada y justificación del nivel).
5. **Ranking territorial** ordena los 10 distritos con mayor incidencia.

> La geometría es una teselación de Voronoi alrededor de los centroides; no es cartografía oficial.

---

## 4. Pronósticos (M07)

1. En **Pronósticos**, seleccione uno o varios horizontes (Ctrl + clic) y pulse **Generar pronóstico**.
2. La tarjeta superior resume la ejecución: `run_key`, modelo utilizado y número de registros.
3. La tabla muestra distrito, semana objetivo, horizonte, casos pronosticados, intervalo del 90 %,
   incidencia y nivel de riesgo.
4. **Explicar** abre la explicación del modelo para ese distrito (SHAP si está disponible, si no
   permutación): factores que aumentan o reducen la estimación, su contribución y una narrativa
   en español.

> El intervalo se calibra con los residuales de validación. Si no hay modelo en producción, la API
> responde con un mensaje que indica cómo crearlo; nunca inventa pronósticos.

---

## 5. Laboratorio de IA (M07, M08, M09)

### 5.1 Entrenar

1. Pestaña **Algoritmos**: tarjetas de XGBoost, LSTM, ConvLSTM, CNN-LSTM y ConvLSTM-Attention con su
   rol y disponibilidad (un algoritmo sin PyTorch aparece como no disponible).
2. **Entrenar** → defina trials de Optuna, estrategia de validación (temporal o geográfica) y
   horizontes → **Iniciar experimento**.
3. La pestaña **Experimentos** muestra progreso en vivo (`optuna_k/n`), MAE del horizonte 1 y
   duración. Haga clic en el nombre para ver el detalle.

### 5.2 Revisar resultados

En el detalle del experimento encontrará, por horizonte: métricas completas (MAE, RMSE, MAPE, MASE,
cobertura, WIS, AUC-PR, sensibilidad, especificidad, Brier), intervalos bootstrap del MAE, mejores
hiperparámetros y las particiones utilizadas (con fechas de corte).

### 5.3 Promover a producción

Pestaña **Versiones de modelo** → **Promover**. Solo hay una versión en producción por horizonte; al
promover se archiva la anterior.

### 5.4 Comparar modelos

Pestaña **Comparación estadística** → elija el horizonte. Verá ranking por MAE/RMSE, el modelo
ganador con su fundamento y las pruebas Friedman, Wilcoxon-Holm y Diebold-Mariano con su p-valor y
conclusión.

---

## 6. Alertas tempranas (M11)

1. **Generar alertas** evalúa los pronósticos vigentes y crea alertas para los niveles iguales o
   superiores a `MODERATE` (configurable en el módulo de administración).
2. Filtre por estado (abiertas, reconocidas, cerradas) y por nivel.
3. **Gestionar** abre el detalle: justificación, intervalo, modelo asociado y notas de seguimiento.
   Acciones: **Reconocer** (asume responsabilidad), **Cerrar** (con nota) o **Reabrir**.

> Las alertas son idempotentes: regenerar no duplica las existentes.

---

## 7. Datos y calidad (M05, M06)

- **Fuentes de datos**: catálogo con proveedor, licencia, método de acceso y estado. **Verificar**
  prueba el adaptador (NASA POWER tiene API pública; el resto se carga por archivo).
- **Importación**: elija tipo (casos o clima), fuente, archivo CSV/JSON y si los registros son
  sintéticos. La previsualización valida columnas obligatorias, duplicados y nulos antes de cargar.
  El resultado informa insertados, actualizados, rechazados y duplicados.
- **Calidad**: totales, cobertura de distritos y semanas, proporción sintética y tabla de chequeos.
- **Casos / Clima**: consulta paginada de los registros almacenados.

Formato de casos: `district_code, epi_year, epi_week, confirmed_cases` (+ opcionales
`probable_cases, hospitalized, deaths, severity`).

---

## 8. Conjuntos de datos (M16)

**Construir conjunto** genera una versión del panel con rezagos 1–12, la registra con `checksum` y
guarda su diccionario de datos. **Diccionario** muestra cada variable, su unidad y su origen
(observada, derivada o de fuente externa).

---

## 9. Reportes (M14)

1. Elija categoría (operacional / gerencial) y un tipo.
2. Seleccione formato: **PDF**, **DOCX** o **XLSX**; opcionalmente un periodo.
3. **Generar reporte** y luego **Descargar** desde el historial.

Cada documento incluye fecha de generación, periodo, filtros, fuentes de datos, versión del modelo,
notas metodológicas y la advertencia de datos sintéticos cuando corresponde.

---

## 10. Asistente inteligente (M13)

1. Seleccione un distrito si la consulta es territorial.
2. Escriba la pregunta o use una sugerencia.
3. La respuesta cita los documentos consultados (protocolo, metodología, umbrales, licencias,
   modelos, limitaciones) y los hechos de la plataforma.

> El asistente **no** produce pronósticos ni cifras propias: explica resultados existentes. Las
> solicitudes de credenciales, datos personales o acciones destructivas se bloquean.

---

## 11. Automatización (M12)

Cada workflow muestra su propósito, programación, pasos y última ejecución. **Ejecutar ahora**
dispara el proceso interno con registro en el historial. Los archivos JSON de `n8n/workflows/` son
importables en n8n para la operación programada real.

---

## 12. Auditoría (M03)

- El estado de la cadena se muestra al entrar: **verde** si es íntegra, **rojo** si se alteró algún
  evento.
- Filtre por tipo de evento y busque por usuario, descripción o recurso.
- Cada fila muestra el hash; **Exportar CSV** descarga la bitácora filtrada (permiso `audit:export`).

---

## 13. Administración (M15) y perfil

- **Configuración**: umbrales de riesgo (con validación `low < moderate < high`), parámetros de
  pronóstico, política de validación, notificaciones y banderas del sistema. Cada cambio queda
  auditado con el valor anterior y el nuevo.
- **Estado de la plataforma**: versión, base de datos, PostGIS, PyTorch, XGBoost, integraciones y
  limitaciones declaradas.
- **Mi perfil**: datos de la cuenta, cambio de contraseña y revocación de sesiones activas.

---

## 14. Preguntas frecuentes

**¿Por qué el pronóstico es 0 en varios distritos?** En semanas de bajo reporte y distritos pequeños
la incidencia esperada es inferior a un caso; el modelo devuelve valores continuos (p. ej. 0,26) y el
intervalo refleja la incertidumbre.

**¿Puedo usar los resultados en un informe oficial?** No con los datos sintéticos. Cargue datos
autorizados de MINSA/CDC, reemplace la cartografía por la oficial y documente las fuentes.

**¿Qué hago si un experimento falla?** El detalle muestra el error y la etapa. Las causas típicas son
ventanas insuficientes de datos o falta de PyTorch para modelos neuronales.

**¿Cómo sé que los datos no se mezclaron?** El detalle del experimento lista `train_end`,
`val_range` y `test_start`; las pruebas automáticas verifican el orden cronológico y la disyunción de
distritos en validación geográfica.
