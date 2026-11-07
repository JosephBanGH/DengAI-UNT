"""Adaptador MINSA/CDC Perú · vigilancia epidemiológica de dengue.

Fuente oficial de datos abiertos del Estado peruano:
``datos_abiertos_vigilancia_dengue_2000_2025.csv``. Contiene microdatos por caso
(departamento, provincia, distrito, año, semana epidemiológica, tipo de diagnóstico
y enfermedad). La plataforma agrega a **distrito-semana**, que es la unidad de análisis
del protocolo SP-01.

Licencia: datos abiertos del Estado peruano, uso con atribución. La fuente original es
el Ministerio de Salud (MINSA) / CDC Perú. No se redistribuye el archivo: se descarga
desde el portal oficial y se conserva en la capa raw inmutable con checksum.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from app.core.config import settings
from app.integrations import raw_store

logger = logging.getLogger("dengai.integrations.minsa")

PROVIDER = "MINSA / CDC Perú"
LICENSE = "Datos abiertos del Estado peruano (uso con atribución)"
SOURCE_CODE = "MINSA_VIGILANCIA"
DATASET_LABEL = "Vigilancia epidemiológica de dengue (MINSA/CDC Perú)"

# Distritos del catálogo de la plataforma que aparecen en la fuente oficial.
DISTRICT_NAME_TO_CODE: Dict[str, str] = {
    "TRUJILLO": "PE-LL-TRU",
    "VICTOR LARCO HERRERA": "PE-LL-VLH",
    "EL PORVENIR": "PE-LL-EPO",
    "LA ESPERANZA": "PE-LL-LES",
    "FLORENCIA DE MORA": "PE-LL-FDM",
    "HUANCHACO": "PE-LL-HUA",
    "LAREDO": "PE-LL-LAR",
    "MOCHE": "PE-LL-MOC",
    "SALAVERRY": "PE-LL-SAL",
    "ASCOPE": "PE-LL-ASC",
    "PAIJAN": "PE-LL-PAI",
    "CHEPEN": "PE-LL-CHE",
    "PACASMAYO": "PE-LL-PAC",
    "SANTIAGO DE CHUCO": "PE-LL-SDC",
}

REQUIRED_COLUMNS = ("departamento", "provincia", "distrito", "enfermedad", "ano", "semana", "tipo_dx")
USECOLS = ("departamento", "provincia", "distrito", "enfermedad", "ano", "semana", "tipo_dx", "ubigeo")


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip().upper()
    return re.sub(r"\s+", " ", text)


def _severity(value: str) -> str:
    text = normalize_name(value)
    if "GRAVE" in text:
        return "GRAVE"
    if "SIGNOS" in text:
        return "CON_SIGNOS_ALARMA"
    return "SIN_SIGNOS_ALARMA"


def _severity_rank(value: str) -> int:
    return {"SIN_SIGNOS_ALARMA": 0, "CON_SIGNOS_ALARMA": 1, "GRAVE": 2}.get(value, 0)


def _week_start(year: int, week: int) -> Optional[date]:
    try:
        return date.fromisocalendar(int(year), int(week), 1)
    except ValueError:
        return None


def probe_minsa_dengue() -> Dict[str, Any]:
    """Verifica disponibilidad de la fuente sin descargar los ~118 MB completos."""
    import time

    import httpx

    started = time.perf_counter()
    try:
        response = httpx.head(settings.MINSA_DENGUE_URL, timeout=30, follow_redirects=True)
        if response.status_code == 405:
            with httpx.stream("GET", settings.MINSA_DENGUE_URL, timeout=30, follow_redirects=True) as stream:
                status = stream.status_code
                headers = stream.headers
        else:
            status = response.status_code
            headers = response.headers
        reachable = status < 400
        return {
            "reachable": reachable,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "status": status,
            "content_length": headers.get("content-length"),
            "content_type": headers.get("content-type", ""),
            "message": (
                "Fuente MINSA/CDC Perú disponible. La ingesta descarga el CSV oficial a la "
                "capa raw con checksum SHA-256 antes de agregarlo a distrito-semana."
                if reachable
                else f"La fuente MINSA respondió con estado HTTP {status}."
            ),
            "endpoint": settings.MINSA_DENGUE_URL,
        }
    except Exception as exc:  # pragma: no cover - red
        logger.warning("MINSA no disponible: %s", exc)
        return {
            "reachable": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "status": None,
            "message": f"No fue posible contactar MINSA/CDC Perú: {type(exc).__name__}: {exc}",
            "endpoint": settings.MINSA_DENGUE_URL,
        }


def download_raw(*, force: bool = False) -> Dict[str, Any]:
    """Descarga el CSV oficial a la capa raw y devuelve su entrada de manifiesto."""
    existing = raw_store.find_entry("MINSA_DENGUE")
    if existing and not force:
        path = settings.raw_dir.parent.parent / existing["path"]
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == existing["sha256"]:
            logger.info("MINSA raw ya disponible: %s", path)
            return existing

    import httpx

    logger.info("Descargando MINSA/CDC Perú desde %s", settings.MINSA_DENGUE_URL)
    response = httpx.get(settings.MINSA_DENGUE_URL, timeout=600, follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if len(content) < 1_000_000:
        raise RuntimeError(
            f"La descarga MINSA es sospechosamente pequeña ({len(content)} bytes); "
            "verifique la URL o el portal de datos abiertos."
        )
    entry = raw_store.save_raw_bytes(
        source="MINSA_DENGUE",
        filename="minsa_dengue.csv",
        content=content,
        url=settings.MINSA_DENGUE_URL,
        license=LICENSE,
        provider=PROVIDER,
        metadata={"dataset": DATASET_LABEL, "content_type": response.headers.get("content-type", "")},
    )
    return entry


def _read_csv(path: Path) -> pd.DataFrame:
    # Lectura por columnas para no cargar variables clínicas innecesarias.
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=list(USECOLS), low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"El CSV MINSA no contiene las columnas requeridas: {missing}")
    return frame


def build_district_week(
    path: Path,
    *,
    start_year: int,
    end_year: int,
    district_codes: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Agrega los microdatos a distrito-semana y devuelve el panel y metadatos de calidad."""
    frame = _read_csv(path)
    frame["_dep"] = frame["departamento"].map(normalize_name)
    frame["_dist"] = frame["distrito"].map(normalize_name)
    frame = frame[(frame["_dep"] == "LA LIBERTAD")].copy()
    frame["_code"] = frame["_dist"].map(DISTRICT_NAME_TO_CODE)
    frame = frame[frame["_code"].notna()].copy()
    frame["ano"] = pd.to_numeric(frame["ano"], errors="coerce")
    frame["semana"] = pd.to_numeric(frame["semana"], errors="coerce")
    frame = frame.dropna(subset=["ano", "semana"])
    frame["ano"] = frame["ano"].astype(int)
    frame["semana"] = frame["semana"].astype(int)
    frame = frame[(frame["ano"] >= start_year) & (frame["ano"] <= end_year)]
    if district_codes:
        frame = frame[frame["_code"].isin(set(district_codes))]
    if frame.empty:
        raise ValueError("La fuente MINSA no contiene registros para los distritos y años solicitados.")

    frame["severity"] = frame["enfermedad"].map(_severity)
    frame["is_confirmed"] = frame["tipo_dx"].astype(str).str.upper().eq("C").astype(int)
    frame["is_probable"] = frame["tipo_dx"].astype(str).str.upper().eq("P").astype(int)
    frame["severity_rank"] = frame["severity"].map(_severity_rank)

    grouped = (
        frame.groupby(["_code", "ano", "semana"], as_index=False)
        .agg(
            confirmed_cases=("is_confirmed", "sum"),
            probable_cases=("is_probable", "sum"),
            severity_rank=("severity_rank", "max"),
            records=("tipo_dx", "size"),
        )
    )
    grouped["confirmed_cases"] = grouped["confirmed_cases"].astype(int)
    grouped["probable_cases"] = grouped["probable_cases"].astype(int)
    grouped["severity"] = grouped["severity_rank"].map({0: "SIN_SIGNOS_ALARMA", 1: "CON_SIGNOS_ALARMA", 2: "GRAVE"})

    # Semanas ISO completas del rango solicitado (lunes a domingo).
    first = date.fromisocalendar(start_year, 1, 1)
    last = date.fromisocalendar(end_year, 52, 1) + timedelta(days=6)
    weeks: List[Tuple[int, int, date]] = []
    cursor = first
    while cursor <= last:
        iso = cursor.isocalendar()
        weeks.append((iso.year, iso.week, cursor))
        cursor += timedelta(days=7)
    weeks = [(y, w, d) for (y, w, d) in weeks if start_year <= y <= end_year]

    observed_codes = sorted(grouped["_code"].unique().tolist())
    # Rejilla completa distrito × semana ISO del rango. La combinación correcta es
    # (code, year, week) de la lista de semanas, no un producto cartesiano de años y
    # semanas independientes.
    keys = pd.MultiIndex.from_tuples([(code, y, w) for code in observed_codes for (y, w, _d) in weeks])
    complete = pd.DataFrame(index=keys, columns=["confirmed_cases", "probable_cases", "severity", "records"]).reset_index()
    complete.columns = ["_code", "ano", "semana", "confirmed_cases", "probable_cases", "severity", "records"]
    merged = complete.merge(grouped, on=["_code", "ano", "semana"], how="left", suffixes=("", "_obs"))
    merged["confirmed_cases"] = merged["confirmed_cases_obs"].fillna(0).astype(int)
    merged["probable_cases"] = merged["probable_cases_obs"].fillna(0).astype(int)
    merged["severity"] = merged["severity_obs"].fillna("SIN_SIGNOS_ALARMA")
    merged["records"] = merged["records_obs"].fillna(0).astype(int)
    merged["data_quality_flag"] = merged["records"].map(lambda n: "REAL" if n > 0 else "REAL_ZERO_FILLED")
    merged["week_start"] = [
        _week_start(int(y), int(w)) for y, w in zip(merged["ano"], merged["semana"])
    ]
    merged = merged[merged["week_start"].notna()].copy()
    merged["district_code"] = merged["_code"]
    merged["source"] = SOURCE_CODE
    merged = merged.sort_values(["district_code", "week_start"]).reset_index(drop=True)

    metadata = {
        "source": SOURCE_CODE,
        "provider": PROVIDER,
        "license": LICENSE,
        "url": settings.MINSA_DENGUE_URL,
        "start_year": start_year,
        "end_year": end_year,
        "districts": observed_codes,
        "rows_panel": int(len(merged)),
        "rows_observed": int((merged["data_quality_flag"] == "REAL").sum()),
        "rows_zero_filled": int((merged["data_quality_flag"] == "REAL_ZERO_FILLED").sum()),
        "confirmed_total": int(merged["confirmed_cases"].sum()),
        "probable_total": int(merged["probable_cases"].sum()),
        "weeks": int(merged["week_start"].nunique()),
        "missing_pct": round(float((merged["data_quality_flag"] == "REAL_ZERO_FILLED").mean() * 100), 3),
        "note": (
            "La ausencia de un registro individual para un distrito-semana se interpreta como "
            "cero casos notificados en el registro de vigilancia; se marca REAL_ZERO_FILLED."
        ),
    }
    return merged, metadata


def ingest(
    db,
    *,
    path: Path,
    start_year: int,
    end_year: int,
    actor: str = "system",
    district_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Reemplaza el panel distrito-semana del rango con datos reales MINSA."""
    from sqlalchemy import delete, select

    from app.db.base import utcnow
    from app.models.epi import DataQualityRecord, DataSource, DengueCase, ImportJob
    from app.models.geo import GeographicUnit

    panel, metadata = build_district_week(
        path, start_year=start_year, end_year=end_year, district_codes=district_codes
    )
    source = db.scalar(select(DataSource).where(DataSource.code == SOURCE_CODE))
    units = {unit.code for unit in db.scalars(select(GeographicUnit))}
    unknown = sorted(set(panel["district_code"]) - units)
    if unknown:
        raise ValueError(f"Distritos del panel sin catálogo geográfico: {unknown}")

    job = ImportJob(
        job_key=f"minsa-{utcnow().strftime('%Y%m%d%H%M%S')}",
        dataset_name=DATASET_LABEL,
        kind="epidemiological",
        status="RUNNING",
        file_name=path.name,
        file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows_total=int(len(panel)),
        initiated_by=actor,
        is_synthetic=False,
        source_id=source.id if source else None,
    )
    db.add(job)
    db.flush()

    codes = sorted(panel["district_code"].unique().tolist())
    # Reemplazo explícito del rango: evita colisiones con el conjunto sintético y
    # garantiza que una misma clave distrito-semana tenga una sola procedencia.
    db.execute(
        delete(DengueCase).where(
            DengueCase.district_code.in_(codes),
            DengueCase.epi_year >= start_year,
            DengueCase.epi_year <= end_year,
        )
    )
    records = []
    for row in panel.itertuples(index=False):
        records.append(
            DengueCase(
                district_code=row.district_code,
                epi_year=int(row.ano),
                epi_week=int(row.semana),
                week_start=row.week_start,
                confirmed_cases=int(row.confirmed_cases),
                probable_cases=int(row.probable_cases),
                hospitalized=0,
                deaths=0,
                severity=str(row.severity),
                source_id=source.id if source else None,
                import_job_id=job.id,
                is_synthetic=False,
                data_quality_flag=str(row.data_quality_flag),
            )
        )
    db.add_all(records)
    job.rows_inserted = len(records)
    job.status = "COMPLETED" if records else "FAILED"
    job.finished_at = utcnow()
    job.duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
    job.log = metadata
    db.add(
        DataQualityRecord(
            import_job_id=job.id,
            dataset_name=DATASET_LABEL,
            check_name="panel_distrito_semana_real",
            severity="WARN" if metadata["missing_pct"] > 50 else "INFO",
            rows_checked=metadata["rows_panel"],
            rows_flagged=metadata["rows_zero_filled"],
            details=metadata,
            created_by=actor,
        )
    )
    if source is not None:
        source.last_success_at = utcnow()
        source.last_error = ""
    db.commit()
    return {
        "source": SOURCE_CODE,
        "status": job.status,
        "job_id": job.id,
        "inserted": len(records),
        "districts": metadata["districts"],
        "weeks": metadata["weeks"],
        "rows_observed": metadata["rows_observed"],
        "rows_zero_filled": metadata["rows_zero_filled"],
        "confirmed_total": metadata["confirmed_total"],
        "probable_total": metadata["probable_total"],
        "missing_pct": metadata["missing_pct"],
        "range": [start_year, end_year],
    }
