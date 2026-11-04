# Imagen del backend: API FastAPI + motor analítico.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias base. Se excluyen dos paquetes del archivo y se instalan aparte:
#   * torch   -> rueda CPU del índice oficial (evita las librerías CUDA de PyPI)
#   * xgboost -> sin dependencias, porque su wheel declara nvidia-nccl-cu12 (351 MB
#                de CUDA inútil en un despliegue de CPU); sus librerías reales
#                (numpy, scipy) ya vienen del archivo de requisitos.
# Ambas se instalan en el site-packages del intérprete, y run_api.py lo detecta para
# no filtrar esas rutas (a diferencia del modo local con libs/).
COPY backend/requirements.txt /app/backend/requirements.txt
RUN grep -Ev "^(torch|xgboost)" /app/backend/requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt \
    && pip install --no-deps "xgboost==3.0.5" \
    && pip install --index-url https://download.pytorch.org/whl/cpu --no-deps "torch==2.9.1+cpu" \
    && pip install "filelock==3.20.0" "sympy==1.14.0" "networkx==3.5" "fsspec==2025.9.0" \
    && pip install "psycopg2-binary==2.9.10" "geoalchemy2==0.17.1"

# Verificación temprana: si falta cualquier dependencia crítica el build falla aquí
# y no en el arranque del contenedor.
RUN python -c "import uvicorn, fastapi, sqlalchemy, pydantic, jwt, bcrypt, reportlab, openpyxl, docx, numpy, scipy, xgboost, torch, shap, optuna; print('dependencias OK')"

COPY backend /app/backend
COPY run_api.py /app/run_api.py

RUN mkdir -p /app/data/models /app/data/reports /app/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

CMD ["python", "run_api.py"]
