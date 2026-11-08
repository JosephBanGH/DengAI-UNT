"""Adaptador NASA POWER (API pública, sin credenciales).

La API no ofrece un endpoint semanal: se descarga la serie **diaria** y se agrega a
semanas ISO (lunes-domingo) de forma determinista. Variables: T2M, T2M_MAX, T2M_MIN,
RH2M, PRECTOTCORR y WS2M.

Documentación: https://power.larc.nasa.gov/docs/services/api/
Licencia: datos de acceso abierto NASA, uso libre con atribución.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app.core.config import settings
from app.integrations import raw_store

logger = logging.getLogger("dengai.integrations.nasa")

VARIABLES = ("T2M", "T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M")
PROVIDER = "NASA POWER"
LICENSE = "Datos de acceso abierto NASA (uso libre con atribución)"


def _get(url: str, params: Dict[str, Any], timeout: float, *, retries: int = 2) -> Dict[str, Any]:
    import httpx

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = httpx.get(url, params=params, timeout=timeout, follow_redirects=True)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - depende de la red
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER no respondió: {type(last_error).__name__}: {last_error}")


def _params(latitude: float, longitude: float, start: date, end: date, parameters: Sequence[str]) -> Dict[str, Any]:
    return {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }


def probe_nasa_power(latitude: float = -8.1061, longitude: float = -79.0287) -> Dict[str, Any]:
    """Verificación de disponibilidad sin descargar la serie completa."""
    started = time.perf_counter()
    end = date.today() - timedelta(days=7)
    start = end - timedelta(days=30)
    base_url = settings.NASA_POWER_BASE_URL.replace("/temporal/weekly/", "/temporal/daily/")
    try:
        payload = _get(
            base_url,
            _params(latitude, longitude, start, end, ("T2M", "RH2M", "PRECTOTCORR")),
            min(settings.NASA_POWER_TIMEOUT, 30),
        )
        latency = int((time.perf_counter() - started) * 1000)
        properties = payload.get("properties", {}).get("parameter", {})
        sample = {key: list(values.items())[:3] for key, values in properties.items()}
        return {
            "reachable": True,
            "latency_ms": latency,
            "message": (
                "NASA POWER respondió correctamente (endpoint diario). Para ingerir la serie "
                "completa ejecute WF-01 o POST /data/ingest/real."
            ),
            "sample": sample,
            "endpoint": base_url,
        }
    except Exception as exc:
        logger.warning("NASA POWER no disponible: %s", exc)
        return {
            "reachable": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "message": f"No fue posible contactar NASA POWER: {type(exc).__name__}: {exc}",
            "sample": None,
            "endpoint": base_url,
        }


def fetch_daily(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    *,
    save_raw: bool = False,
    label: str = "",
    force_download: bool = False,
) -> List[Dict[str, Any]]:
    """Descarga la serie diaria y la devuelve como filas normalizadas.

    Si el archivo ya existe en data/raw/NASA_POWER y no se fuerza la descarga,
    se carga directamente de disco con integridad verificable.
    """
    import json

    target_filename = f"nasa_power_{label or 'point'}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    raw_file = settings.raw_dir / "NASA_POWER" / target_filename
    payload = None

    if raw_file.exists() and not force_download:
        try:
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
            logger.info("NASA POWER raw cargado desde disco: %s", raw_file.name)
        except Exception as exc:
            logger.warning("No se pudo leer raw existente %s: %s; se reintentará descarga", raw_file, exc)
            payload = None

    if payload is None:
        base_url = settings.NASA_POWER_BASE_URL.replace("/temporal/weekly/", "/temporal/daily/")
        payload = _get(
            base_url,
            _params(latitude, longitude, start, end, VARIABLES),
            settings.NASA_POWER_TIMEOUT,
        )
        if save_raw:
            raw_store.save_raw_json(
                source="NASA_POWER",
                filename=target_filename,
                payload=payload,
                url=base_url,
                license=LICENSE,
                provider=PROVIDER,
                metadata={"latitude": latitude, "longitude": longitude, "start": start.isoformat(), "end": end.isoformat()},
            )
    parameter = payload.get("properties", {}).get("parameter", {})
    stamps = sorted({key for series in parameter.values() for key in series})
    rows: List[Dict[str, Any]] = []
    for stamp in stamps:
        try:
            day = date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
        except ValueError:
            continue
        value = lambda name: parameter.get(name, {}).get(stamp)  # noqa: E731
        rows.append(
            {
                "date": day,
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


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / len(clean) if clean else None


def _sum(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) if clean else None


def aggregate_weekly(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrega filas diarias a semanas ISO. ``week_start`` es el lunes de la semana."""
    buckets: Dict[date, List[Dict[str, Any]]] = {}
    for row in rows:
        day: date = row["date"]
        monday = day - timedelta(days=day.weekday())
        buckets.setdefault(monday, []).append(row)
    weekly: List[Dict[str, Any]] = []
    for monday in sorted(buckets):
        chunk = buckets[monday]
        iso = monday.isocalendar()
        weekly.append(
            {
                "week_start": monday,
                "epi_year": int(iso.year),
                "epi_week": int(iso.week),
                "temperature_mean": _mean(r["temperature_mean"] for r in chunk),
                "temperature_max": _mean(r["temperature_max"] for r in chunk),
                "temperature_min": _mean(r["temperature_min"] for r in chunk),
                "humidity_mean": _mean(r["humidity_mean"] for r in chunk),
                "precipitation_mm": _sum(r["precipitation_mm"] for r in chunk),
                "wind_speed": _mean(r["wind_speed"] for r in chunk),
                "days_observed": len(chunk),
                "provider": "NASA_POWER",
            }
        )
    return [row for row in weekly if row["temperature_mean"] is not None]


def fetch_weekly(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    *,
    save_raw: bool = False,
    label: str = "",
) -> List[Dict[str, Any]]:
    """Descarga la serie diaria y la devuelve agregada por semana ISO."""
    daily = fetch_daily(latitude, longitude, start, end, save_raw=save_raw, label=label)
    return aggregate_weekly(daily)


def ingest_nasa_power(
    db,
    *,
    units: Sequence[Any],
    start: date,
    end: date,
    actor: str = "system",
) -> Dict[str, Any]:
    """Descarga y persiste observaciones climáticas reales por distrito.

    ``units`` debe contener objetos con ``code``, ``centroid_lat`` y ``centroid_lon``.
    Los datos se marcan con ``is_synthetic=False`` y ``provider='NASA_POWER'``.
    """
    from sqlalchemy import delete, select

    from app.db.base import utcnow
    from app.models.epi import ClimateObservation, DataSource, ImportJob

    source = db.scalar(select(DataSource).where(DataSource.code == "NASA_POWER"))
    job = ImportJob(
        job_key=f"nasa-{utcnow().strftime('%Y%m%d%H%M%S')}",
        dataset_name="NASA POWER · serie diaria agregada a semana ISO",
        kind="climate",
        status="RUNNING",
        file_name="nasa_power_daily.json",
        initiated_by=actor,
        is_synthetic=False,
        source_id=source.id if source else None,
    )
    db.add(job)
    db.flush()

    inserted = updated = skipped = 0
    per_district: Dict[str, int] = {}
    errors: List[str] = []
    try:
        for unit in units:
            if unit.centroid_lat is None or unit.centroid_lon is None:
                skipped += 1
                continue
            try:
                rows = fetch_weekly(
                    float(unit.centroid_lat),
                    float(unit.centroid_lon),
                    start,
                    end,
                    save_raw=True,
                    label=str(unit.code),
                )
            except Exception as exc:  # pragma: no cover - red
                errors.append(f"{unit.code}: {type(exc).__name__}: {exc}")
                skipped += 1
                continue
            per_district[str(unit.code)] = len(rows)
            for row in rows:
                existing = db.scalar(
                    select(ClimateObservation).where(
                        ClimateObservation.district_code == unit.code,
                        ClimateObservation.epi_year == row["epi_year"],
                        ClimateObservation.epi_week == row["epi_week"],
                    )
                )
                payload = {
                    "temperature_mean": _round(row["temperature_mean"]),
                    "temperature_max": _round(row["temperature_max"]),
                    "temperature_min": _round(row["temperature_min"]),
                    "humidity_mean": _round(row["humidity_mean"]),
                    "precipitation_mm": _round(row["precipitation_mm"]),
                    "wind_speed": _round(row["wind_speed"]),
                    "altitude_m": unit.altitude_m,
                    "provider": "NASA_POWER",
                    "source_id": source.id if source else None,
                    "import_job_id": job.id,
                    "is_synthetic": False,
                    "data_quality_flag": "OK",
                }
                if existing is None:
                    db.add(
                        ClimateObservation(
                            district_code=unit.code,
                            epi_year=row["epi_year"],
                            epi_week=row["epi_week"],
                            week_start=row["week_start"],
                            **payload,
                        )
                    )
                    inserted += 1
                else:
                    for key, value in payload.items():
                        if value is not None:
                            setattr(existing, key, value)
                    updated += 1
        job.rows_inserted = inserted
        job.rows_updated = updated
        job.rows_rejected = skipped
        job.rows_total = inserted + updated + skipped
        job.status = "COMPLETED" if inserted + updated > 0 else "FAILED"
        job.error_message = "; ".join(errors[:5])
        job.finished_at = utcnow()
        job.duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
        job.log = {"per_district": per_district, "errors": errors[:10]}
        if source is not None:
            source.last_success_at = utcnow() if job.status == "COMPLETED" else source.last_success_at
            source.last_error = "" if job.status == "COMPLETED" else job.error_message
        db.commit()
        return {
            "source": "NASA_POWER",
            "status": job.status,
            "inserted": inserted,
            "updated": updated,
            "skipped_districts": skipped,
            "per_district": per_district,
            "errors": errors,
            "job_id": job.id,
            "range": [start.isoformat(), end.isoformat()],
        }
    except Exception as exc:
        db.rollback()
        raise RuntimeError(f"Ingesta NASA POWER falló: {type(exc).__name__}: {exc}") from exc


def _round(value: Optional[float]) -> Optional[float]:
    return round(float(value), 3) if value is not None else None
