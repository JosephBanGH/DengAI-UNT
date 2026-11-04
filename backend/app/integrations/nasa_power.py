"""Adaptador NASA POWER (API pública, sin clave).

Documentación: https://power.larc.nasa.gov/docs/services/api/
Se usa para temperatura (T2M), humedad relativa (RH2M) y precipitación (PRECTOTCORR)
por punto geográfico. La descarga real solo se ejecuta si LIVE_INTEGRATIONS_ENABLED=1.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("dengai.integrations.nasa")

VARIABLES = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M")


def _get(url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    import httpx

    response = httpx.get(url, params=params, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def probe_nasa_power(latitude: float = -8.1061, longitude: float = -79.0287) -> Dict[str, Any]:
    """Verificación de disponibilidad sin descargar la serie completa."""
    started = time.perf_counter()
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=30)
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        payload = _get(settings.NASA_POWER_BASE_URL, params, min(settings.NASA_POWER_TIMEOUT, 25))
        latency = int((time.perf_counter() - started) * 1000)
        properties = payload.get("properties", {}).get("parameter", {})
        sample = {key: list(values.values())[:3] for key, values in properties.items()}
        return {
            "reachable": True,
            "latency_ms": latency,
            "message": (
                "NASA POWER respondió correctamente. Para ingerir la serie completa active "
                "LIVE_INTEGRATIONS_ENABLED=1 y ejecute WF-01."
            ),
            "sample": sample,
        }
    except Exception as exc:
        logger.warning("NASA POWER no disponible: %s", exc)
        return {
            "reachable": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "message": f"No fue posible contactar NASA POWER: {type(exc).__name__}: {exc}",
            "sample": None,
        }


def fetch_weekly(latitude: float, longitude: float, start: date, end: date) -> List[Dict[str, Any]]:
    """Descarga la serie semanal y la devuelve en el formato del pipeline."""
    if not settings.LIVE_INTEGRATIONS_ENABLED:
        raise RuntimeError(
            "Las integraciones en vivo están desactivadas (LIVE_INTEGRATIONS_ENABLED=0)."
        )
    params = {
        "parameters": ",".join(VARIABLES),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    payload = _get(settings.NASA_POWER_BASE_URL, params, settings.NASA_POWER_TIMEOUT)
    parameter = payload.get("properties", {}).get("parameter", {})
    dates = sorted({key for series in parameter.values() for key in series})
    rows = []
    for stamp in dates:
        year, month, day = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8])
        if month == 12 and day >= 26:
            continue  # NASA POWER usa semanas de 7 días; se descarta el corte parcial de fin de año
        value = lambda name: parameter.get(name, {}).get(stamp)  # noqa: E731
        rows.append(
            {
                "date": date(year, month, day),
                "temperature_mean": value("T2M"),
                "temperature_max": value("T2M_MAX"),
                "temperature_min": value("T2M_MIN"),
                "humidity_mean": value("RH2M"),
                "precipitation_mm": value("PRECTOTCORR"),
                "wind_speed": value("WS2M"),
                "provider": "NASA_POWER",
            }
        )
    return [row for row in rows if row["temperature_mean"] is not None]
