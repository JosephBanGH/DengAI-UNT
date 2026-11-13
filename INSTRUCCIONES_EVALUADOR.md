# 🎓 Guía de Puesta en Marcha Rápida (Para el Docente / Evaluador)

Bienvenido a la entrega del software **SP-01 Dengue Intelligence** (Sistema Inteligente de Predicción Espacio-Temporal y Alerta Temprana de Brotes de Dengue - DengAI UNT).

Este paquete ha sido optimizado bajo estándares de ingeniería y reproducibilidad científica:
* **Frontend precompilado:** Se incluye la compilación lista en `frontend/dist/`. **No requiere instalar Node.js ni compilar assets** para evaluar la plataforma completa.
* **Modelos y Base de Datos preconfigurados:** El archivo `data/dengai.db` y los modelos predictivos en `data/models/` ya están listos, por lo que el sistema funciona inmediatamente sin necesidad de entrenamientos prolongados.

---

## 🚀 Opción 1: Ejecución Directa con Python (Recomendada, rápida)

**Requisito:** Python 3.10 o superior (recomendado Python 3.12).

### Paso 1: Crear y activar un entorno virtual aislado
En la raíz de la carpeta descomprimida:

* **En Windows (PowerShell / CMD):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **En Linux / macOS (Terminal):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Paso 2: Instalar dependencias del backend
```bash
pip install -r backend/requirements.txt
```

### Paso 3: Iniciar la aplicación
```bash
python run_api.py
```

### Paso 4: Abrir en el navegador
* **Aplicación Web Completa:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Documentación interactiva de la API (Swagger / OpenAPI):** [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)

---

## 🐳 Opción 2: Ejecución Hermética con Docker

Si prefiere evaluar el sistema mediante contenedores (incluye microservicios adicionales como PostGIS, n8n y MinIO):

**Requisito:** Docker Desktop o Docker Engine con Docker Compose v2+.

```bash
docker compose up -d --build
```

* **Aplicación:** [http://localhost:8080](http://localhost:8080)
* **API y Docs:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

Para detener los contenedores:
```bash
docker compose down
```

---

## 🔑 Credenciales de Acceso para Demostración

| Rol | Correo Electrónico | Contraseña | Perfil y Acceso |
| :--- | :--- | :--- | :--- |
| **Administrador del Sistema** | `admin@dengai.unt.edu.pe` | `Admin#Dengue2025` | Acceso total, configuración, gestión de usuarios y bitácora |
| **Epidemiólogo** | `epidemiologo@dengai.unt.edu.pe` | `Demo#Dengue2025` | Mapas de riesgo, curvas de alerta y simulación de escenarios |
| **Analista de Datos** | `analista@dengai.unt.edu.pe` | `Demo#Dengue2025` | Calidad de datos, métricas y validación cruzada |
| **Investigador** | `investigador@dengai.unt.edu.pe` | `Demo#Dengue2025` | Laboratorio de IA, SHAP, curvas ROC y análisis espaciotemporal |

*(Nota: En el primer ingreso el sistema permite el uso inmediato de los tableros con datos cargados).*

---

## 🧪 Ejecución de Pruebas Automatizadas

Para verificar la integridad matemática y funcional del sistema (27 pruebas unitarias y de integración):

```bash
python -m pytest backend/tests -q
```
Todas las pruebas deben concluir en verde.
