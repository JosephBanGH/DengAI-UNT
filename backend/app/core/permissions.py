"""Catálogo de permisos y plantillas de roles (RBAC)."""
from __future__ import annotations

from typing import Dict, List, Tuple

# (código, módulo, descripción)
PERMISSIONS: List[Tuple[str, str, str]] = [
    ("dashboard:read", "M04", "Consultar el dashboard ejecutivo"),
    ("data:read", "M05", "Consultar datos epidemiológicos y climáticos"),
    ("data:write", "M05", "Crear y modificar registros de datos"),
    ("data:import", "M05", "Importar archivos de fuentes de datos"),
    ("datasets:read", "M05", "Consultar conjuntos de datos"),
    ("datasets:write", "M05", "Construir y versionar conjuntos de datos"),
    ("ml:read", "M07", "Consultar experimentos y modelos"),
    ("ml:run", "M07", "Ejecutar entrenamientos y optimización"),
    ("ml:promote", "M07", "Promover versiones de modelo a producción"),
    ("predictions:read", "M07", "Consultar pronósticos y riesgo"),
    ("predictions:run", "M07", "Generar inferencias"),
    ("explain:read", "M09", "Consultar explicaciones de predicciones"),
    ("geo:read", "M10", "Consultar mapas y unidades geográficas"),
    ("alerts:read", "M11", "Consultar alertas tempranas"),
    ("alerts:write", "M11", "Reconocer, cerrar y generar alertas"),
    ("workflows:read", "M12", "Consultar workflows y ejecuciones"),
    ("workflows:run", "M12", "Ejecutar workflows operacionales"),
    ("assistant:ask", "M13", "Consultar el asistente inteligente"),
    ("reports:read", "M14", "Consultar y descargar reportes"),
    ("reports:generate", "M14", "Generar reportes"),
    ("users:read", "M02", "Consultar usuarios"),
    ("users:write", "M02", "Crear y modificar usuarios"),
    ("roles:read", "M02", "Consultar roles y permisos"),
    ("roles:write", "M02", "Administrar roles y permisos"),
    ("audit:read", "M03", "Consultar la bitácora de auditoría"),
    ("audit:export", "M03", "Exportar la bitácora de auditoría"),
    ("admin:config", "M15", "Administrar configuración del sistema"),
]

ALL_PERMISSION_CODES = [code for code, _, _ in PERMISSIONS]

_ROLE_TEMPLATES: Dict[str, Dict[str, object]] = {
    "admin_sistema": {
        "name": "Administrador del sistema",
        "description": "Control total de la plataforma, usuarios, roles y configuración.",
        "permissions": ALL_PERMISSION_CODES,
    },
    "admin_epidemiologico": {
        "name": "Administrador epidemiológico",
        "description": "Administra datos, modelos, alertas y reportes epidemiológicos.",
        "permissions": [
            "dashboard:read", "data:read", "data:write", "data:import", "datasets:read",
            "datasets:write", "ml:read", "ml:run", "ml:promote", "predictions:read",
            "predictions:run", "explain:read", "geo:read", "alerts:read", "alerts:write",
            "workflows:read", "workflows:run", "assistant:ask", "reports:read",
            "reports:generate", "users:read", "roles:read", "audit:read", "admin:config",
        ],
    },
    "epidemiologo": {
        "name": "Epidemiólogo",
        "description": "Analiza la situación epidemiológica y gestiona alertas.",
        "permissions": [
            "dashboard:read", "data:read", "datasets:read", "ml:read", "predictions:read",
            "predictions:run", "explain:read", "geo:read", "alerts:read", "alerts:write",
            "assistant:ask", "reports:read", "reports:generate", "workflows:read",
        ],
    },
    "analista_datos": {
        "name": "Analista de datos",
        "description": "Prepara datos, ejecuta experimentos y valida resultados.",
        "permissions": [
            "dashboard:read", "data:read", "data:write", "data:import", "datasets:read",
            "datasets:write", "ml:read", "ml:run", "predictions:read", "predictions:run",
            "explain:read", "geo:read", "assistant:ask", "reports:read", "workflows:read",
            "workflows:run",
        ],
    },
    "investigador": {
        "name": "Investigador",
        "description": "Reproduce experimentos y documenta resultados científicos.",
        "permissions": [
            "dashboard:read", "data:read", "datasets:read", "ml:read", "ml:run",
            "predictions:read", "explain:read", "geo:read", "assistant:ask",
            "reports:read", "reports:generate",
        ],
    },
    "auditor": {
        "name": "Auditor",
        "description": "Verifica trazabilidad, bitácora y control de acceso.",
        "permissions": [
            "dashboard:read", "audit:read", "audit:export", "users:read", "roles:read",
            "reports:read", "workflows:read", "data:read",
        ],
    },
    "consulta": {
        "name": "Usuario de consulta",
        "description": "Consulta información publicada sin capacidad de modificación.",
        "permissions": [
            "dashboard:read", "data:read", "predictions:read", "geo:read",
            "alerts:read", "reports:read", "assistant:ask",
        ],
    },
}

ROLE_TEMPLATES = _ROLE_TEMPLATES
