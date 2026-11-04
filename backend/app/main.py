"""Aplicación FastAPI: ciclo de vida, middleware, manejo de errores y entrega del frontend."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1 import api_router
from app.core.config import PROJECT_DIR, settings
from app.core.errors import AppError, NotFoundError
from app.db.session import SessionLocal, engine
from app.models import Base
from app.services import audit as audit_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("dengai.main")

FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

TAGS_METADATA = [
    {"name": "M01 Autenticación", "description": "Login, sesiones, contraseñas y control de acceso."},
    {"name": "M02 Usuarios y roles", "description": "RBAC, perfiles y administración de identidades."},
    {"name": "M03 Auditoría", "description": "Bitácora encadenada, verificación y exportación."},
    {"name": "M04 Dashboard · M10 Mapas · M07 Pronósticos", "description": "Indicadores, cartografía y pronósticos."},
    {"name": "M05 Datos · M06 Pipeline · M16 Reproducibilidad", "description": "Importación, calidad y conjuntos de datos."},
    {"name": "M07 Laboratorio de IA · M08 · M09", "description": "Entrenamiento, evaluación y explicabilidad."},
    {"name": "M11 Alertas tempranas", "description": "Detección, seguimiento y cierre de alertas."},
    {"name": "M12 Orquestación", "description": "Workflows WF-01 a WF-04."},
    {"name": "M13 Asistente inteligente", "description": "RAG sobre documentación verificada."},
    {"name": "M14 Reportes", "description": "Generación PDF, DOCX y XLSX."},
    {"name": "M15 Administración · Estado del sistema", "description": "Configuración y salud de la plataforma."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando %s v%s (%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT)
    Base.metadata.create_all(bind=engine)
    from app.db.seed import run_seed

    session = SessionLocal()
    try:
        summary = run_seed(session)
        logger.info("Poblado inicial: %s", summary)
    except Exception:
        logger.exception("Fallo el poblado inicial; la aplicación continúa en modo degradado")
    finally:
        session.close()
    for directory in (settings.models_dir, settings.reports_dir, settings.data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        audit_service.record(
            session,
            event_type=audit_service.AuditEventType.MODEL_EXECUTION,
            module="SYS",
            action="startup",
            result="SUCCESS",
            description=f"Plataforma iniciada (v{settings.APP_VERSION}, entorno {settings.ENVIRONMENT}).",
        )
        session.commit()
    yield
    logger.info("Deteniendo la plataforma")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Plataforma inteligente de predicción espacio-temporal y alerta temprana de brotes de dengue. "
        "Sistema de apoyo a la decisión epidemiológica; no realiza diagnóstico clínico individual."
    ),
    version=settings.APP_VERSION,
    openapi_tags=TAGS_METADATA,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "Content-Disposition"],
)


@app.middleware("http")
async def platform_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except AppError:
        raise
    duration_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    if request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Los datos enviados no son válidos.",
                "details": {"errors": exc.errors()[:10]},
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": f"HTTP_{exc.status_code}", **detail}},
    )


app.include_router(api_router, prefix=settings.API_V1_PREFIX)


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api"):
            raise NotFoundError("Recurso no encontrado.")
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/api/docs")
