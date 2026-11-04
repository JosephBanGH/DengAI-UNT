"""Agregador de routers de la API v1."""
from fastapi import APIRouter

from app.api.v1 import admin_users, analytics, data, ml, module_audit, ops, system

api_router = APIRouter()
api_router.include_router(admin_users.users_router)
api_router.include_router(admin_users.roles_router)
api_router.include_router(admin_users.audit_router)
api_router.include_router(module_audit.router)
api_router.include_router(analytics.router)
api_router.include_router(data.router)
api_router.include_router(ml.router)
api_router.include_router(ops.alerts_router)
api_router.include_router(ops.reports_router)
api_router.include_router(ops.assistant_router)
api_router.include_router(ops.workflows_router)
api_router.include_router(system.router)

# El router de autenticación se incluye al final para mantener el orden del OpenAPI.
from app.api.v1 import auth as auth_router  # noqa: E402

api_router.include_router(auth_router.router)

__all__ = ["api_router"]
