# Guía Completa: Cómo Levantar DengAI-UNT (SP-01) con Todos los Datos

Esta guía explica paso a paso cómo desplegar y ejecutar **SP-01 Dengue Intelligence (DengAI-UNT)** desde cero con **todos los datos reales** (vigilancia epidemiológica MINSA/CDC Perú + meteorología NASA POWER), los modelos entrenados y el entorno completo (local o Docker).

---

## 1. Qué incluye el repositorio y qué se obtiene/genera aparte

Por límites de tamaño de GitHub (>100 MB por archivo) y seguridad de credenciales, el archivo `.gitignore` excluye:

| Carpeta / Archivo | Tamaño aprox. | Por qué no está en Git | Cómo se obtiene |
| :--- | :--- | :--- | :--- |
| `.env` | < 4 KB | Secretos locales (`SECRET_KEY`, claves API) | Copiar `.env.example` a `.env` |
| `data/raw/MINSA_DENGUE/minsa_dengue.csv` | ~112 MB | Supera el límite de 100 MB de GitHub | Se descarga automáticamente con `ingest_real_data.py` o `POST /api/v1/data/ingest/real` |
| `data/raw/NASA_POWER/*.json` | ~5.5 MB (14 archivos) | Datos crudos descargables | Se descargan automáticamente por distrito desde la API pública de NASA POWER |
| `data/dengai.db` | ~5 MB | Base de datos SQLite operativa | Se crea al arrancar la API y se puebla con la ingesta real |
| `data/models/*.joblib` y `*.npz` | ~15 MB | Artefactos binarios de modelos | Se entrenan con `training/` o el notebook de Colab |
| `libs/` / `.venv/` / `node_modules/` | Variable | Dependencias instalables | `pip install` y `npm install` |
| `frontend/dist/` | ~350 KB | Compilación del frontend | `npm run build` en `frontend/` |

---

## 2. Requisitos Previos

- **Python 3.12** (compatible con 3.10+)
- **Node.js 22** (opcional si deseas recompilar o desarrollar el frontend; necesario para generar `frontend/dist/` tras clonar el repo)
- **Git**
- **Docker Desktop / Docker Engine + Compose v2** (solo si eliges el despliegue con contenedores PostgreSQL + PostGIS + MinIO + MLflow + n8n)
- **Conexión a Internet** (solo la primera vez para descargar paquetes y los datasets públicos de MINSA/CDC Perú y NASA POWER)

---

## 3. Opción A: Puesta en Marcha Local (Python + SQLite) con Datos Reales

### Paso 1: Clonar el repositorio y preparar variables de entorno

```powershell
git clone https://github.com/JosephBanGH/DengAI-UNT.git
cd DengAI-UNT
Copy-Item .env.example .env
```

> En Linux/macOS usa `cp .env.example .env`.
> Si deseas que el sistema use **datos reales** prioritariamente, verifica en `.env` que `DATA_MODE=auto` (o `DATA_MODE=real`) y, si quieres permitir consultas en vivo desde la interfaz, `LIVE_INTEGRATIONS_ENABLED=true`. Si dejas `SECRET_KEY=` vacío, `backend/app/core/config.py` generará una clave automáticamente en el primer arranque.

### Paso 2: Instalar dependencias del backend

Puedes usar entorno virtual (`.venv`) o carpeta local aislada (`./libs`, soportada nativamente por `run_api.py`):

**Opción recomendada (`.venv`):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate        # En Linux/macOS: source .venv/bin/activate
pip install -r backend/requirements.txt
```

**Opción alternativa (`./libs` sin tocar el Python global):**
```powershell
python -m pip install --target .\libs -r backend\requirements.txt
```

*(Nota: si tienes GPU NVIDIA con CUDA, puedes instalar PyTorch con soporte CUDA según `backend/requirements.txt` en lugar de `torch==2.9.1+cpu`).*

### Paso 3: Compilar el frontend (React + Vite)

Como `frontend/dist/` no está versionado en GitHub, compílalo antes de levantar el servidor si accedes desde `http://127.0.0.1:8000`:

```powershell
cd frontend
npm install
npm run build
cd ..
```

### Paso 4: Descargar e ingerir TODOS los datos reales (MINSA + NASA POWER)

Este comando descarga el archivo oficial de vigilancia de dengue del **MINSA / CDC Perú** (`datos_abiertos_vigilancia_dengue_2000_2025_0.csv`, ~112 MB) en `data/raw/MINSA_DENGUE/minsa_dengue.csv`, descarga las series diarias climáticas de **NASA POWER** para los **14 distritos de La Libertad** en `data/raw/NASA_POWER/`, registra checksums SHA-256 en `data/raw/manifest.json` y los agrega por **distrito-semana epidemiológica** dentro de `data/dengai.db` con `is_synthetic=False`:

```powershell
python backend/scripts/ingest_real_data.py --start-year 2015 --end-year 2024
```

Opciones útiles del script:
- `--skip-cases` / `--skip-climate`: omite una de las dos fuentes.
- `--force`: fuerza la re-descarga del CSV de MINSA aunque ya exista en `data/raw/`.
- `--json`: imprime el resumen completo de ingesta y procedencia en JSON.

> **Nota sobre modo offline / backup:** Si ya cuentas con `data/raw/` (el CSV de MINSA y los JSON de NASA POWER) copiados manualmente, el script detectará el archivo existente en disco por su manifiesto/ruta y realizará la ingesta sin volver a descargar.

### Paso 5: Entrenar los modelos (o restaurar modelos preentrenados)

Si deseas entrenar los 5 modelos (`xgboost`, `lstm`, `convlstm`, `cnn_lstm`, `convlstm_attention`) sobre los datos reales recién ingeridos:

**Prueba rápida / validación ligera multi-modelo:**
```powershell
python backend/scripts/train_lightweight.py
```

**Entrenamiento modular por algoritmo (Fase 2, guarda en `data/models/phase2/`):**
```powershell
python training/train_xgboost.py --trials 50 --horizon 1
python training/train_lstm.py --trials 30 --epochs 40 --horizon 1
python training/train_cnn_lstm.py --trials 30 --epochs 40 --horizon 1
python training/train_convlstm.py --trials 30 --epochs 40 --horizon 1
python training/train_convlstm_attention.py --trials 30 --epochs 40 --horizon 1
```

**Optimización conjunta o Validación Cruzada Anidada (Nested Walk-Forward CV 5×3):**
```powershell
python training/optimize_optuna.py --trials 30 --epochs 40 --horizon 1
python training/run_nested_cv.py --models xgboost lstm --trials 20 --epochs 30 --horizon 1 --resume
```

**Explicabilidad (SHAP + LIME + mapas de atención):**
```powershell
python training/explain_models.py --district "PE-LL-TRU" --week 202415
```

> **Tip GPU (Google Colab):** Para entrenar `ConvLSTM`, `CNN-LSTM` y `ConvLSTM-Attention` con 50+ trials de Optuna rápidamente, utiliza los notebooks `notebooks/colab_entrenamiento_optuna.ipynb` o `notebooks/colab_nested_cv_fase2.ipynb` y luego copia los archivos `.joblib` y `.npz` generados dentro de `data/models/` (y `data/dengai.db` si deseas conservar el registro de experimentos).

### Paso 6: Levantar la aplicación

```powershell
python run_api.py
```

- **Aplicación Web:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Swagger / OpenAPI:** [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
- **Puerto alternativo:** `python run_api.py --port 8001`

*(En modo desarrollo de frontend con hot-reload, levanta `python run_api.py` en una terminal y en otra ejecuta `cd frontend; npm run dev` para abrir `http://127.0.0.1:5174`).*

---

## 4. Opción B: Despliegue Completo con Docker (PostgreSQL/PostGIS + MinIO + MLflow + n8n)

Si quieres levantar toda la arquitectura de servicios definida en `docker-compose.yml`:

### Paso 1: Configurar `.env`

```bash
cp .env.example .env
```

Edita `.env` y asigna obligatoriamente un `SECRET_KEY` (si está vacío, `docker compose` fallará por seguridad):

```powershell
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))"
```

### Paso 2: Construir y levantar los contenedores

```bash
docker compose up -d --build
```

Servicios expuestos:
- **Aplicación Web (Nginx + Frontend):** [http://localhost:8080](http://localhost:8080)
- **API Backend (FastAPI):** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **MLflow UI:** [http://localhost:5050](http://localhost:5050)
- **MinIO Console:** [http://localhost:9001](http://localhost:9001)
- **n8n Automatización:** [http://localhost:5680](http://localhost:5680)
- **PostgreSQL + PostGIS:** puerto `5432`

### Paso 3: Ingerir los datos reales dentro del contenedor backend

Una vez que los contenedores estén saludables (`docker compose ps`):

```bash
docker compose exec backend python backend/scripts/ingest_real_data.py --start-year 2015 --end-year 2024
```

También puedes disparar la ingesta desde la API (`POST /api/v1/data/ingest/real` con token de administrador) o desde la vista **Datos y Calidad** en la interfaz web.

---

## 5. Credenciales Iniciales de Acceso

El primer arranque siembra los roles y usuarios iniciales (solo si la tabla de usuarios está vacía):

| Rol | Correo | Contraseña inicial |
| :--- | :--- | :--- |
| Administrador del sistema | `admin@dengai.unt.edu.pe` | `Admin#Dengue2025` |
| Administrador epidemiológico | `admin.epi@dengai.unt.edu.pe` | `Demo#Dengue2025` |
| Epidemiólogo | `epidemiologo@dengai.unt.edu.pe` | `Demo#Dengue2025` |
| Analista de datos | `analista@dengai.unt.edu.pe` | `Demo#Dengue2025` |
| Investigador | `investigador@dengai.unt.edu.pe` | `Demo#Dengue2025` |
| Auditor | `auditor@dengai.unt.edu.pe` | `Demo#Dengue2025` |
| Consulta | `consulta@dengai.unt.edu.pe` | `Demo#Dengue2025` |

> Cambia estas contraseñas y define un `SECRET_KEY` propio antes de exponer el sistema en red o producción.

---

## 6. Verificación del Sistema y Pruebas

1. **Salud de la API:** abrir `http://127.0.0.1:8000/api/v1/health` (debe responder HTTP `200`).
2. **Procedencia real vs. sintética:** consultar `GET /api/v1/data/provenance` o la pestaña **Datos** para confirmar `effective_data_mode: "real"`, casos MINSA (`2015-2024`) y clima NASA POWER (`14` distritos).
3. **Suite de pruebas automatizadas:**
   ```powershell
   $env:PYTHONPATH="$PWD\libs;$PWD\backend;$PWD"
   python -m pytest backend/tests -q
   ```
4. **Verificación de tipos y build del frontend:**
   ```powershell
   cd frontend
   npx tsc --noEmit
   npm run build
   ```
