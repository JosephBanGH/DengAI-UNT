"""Punto de entrada del backend.

Resuelve el orden de búsqueda de módulos en los dos escenarios de despliegue:

* **Local (Windows/Linux sin Docker)**: las dependencias se instalaron con
  ``pip install --target libs``; esa carpeta debe tener prioridad y se descartan las
  rutas de ``site-packages`` ajenas para no mezclar versiones.
* **Contenedor (Docker)**: las dependencias están en el ``site-packages`` del
  intérprete y **no debe filtrarse nada**; hacerlo dejaría la imagen sin ``uvicorn``,
  ``fastapi`` ni el resto de paquetes.

La decisión se toma observando si existe ``libs/`` junto a este archivo.
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
LIBS = ROOT / "libs"
BACKEND = ROOT / "backend"

logger = logging.getLogger("dengai.bootstrap")


def _is_site_packages(entry: str) -> bool:
    return entry.lower().replace("\\", "/").rstrip("/").endswith("site-packages")


def prepare_sys_path() -> str:
    """Deja ``backend/`` importable y devuelve el modo de resolución usado."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))

    if LIBS.is_dir():
        if str(LIBS) not in sys.path:
            sys.path.insert(0, str(LIBS))
        # Solo con dependencias locales tiene sentido aislarse de site-packages.
        sys.path[:] = [entry for entry in sys.path if not _is_site_packages(entry)]
        return "libs-local"

    # Contenedor o instalación convencional: se respeta el sys.path del intérprete.
    return "site-packages"


def _default_host() -> str:
    """Escucha en todas las interfaces dentro de un contenedor."""
    explicit = os.environ.get("HOST")
    if explicit:
        return explicit
    if pathlib.Path("/.dockerenv").exists():
        return "0.0.0.0"
    return "127.0.0.1"


def main() -> None:
    mode = prepare_sys_path()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "info").upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    logger.info("Resolución de módulos: %s (%s)", mode, LIBS if mode == "libs-local" else "site-packages")

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = _default_host()
    reload_enabled = os.environ.get("RELOAD", "0") == "1"
    logger.info("Servidor en http://%s:%s (reload=%s)", host, port, reload_enabled)
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=os.environ.get("LOG_LEVEL", "info"),
        access_log=True,
    )


if __name__ == "__main__":
    main()
