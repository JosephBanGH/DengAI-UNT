"""Importación de datos, validación y calidad (M05/M06)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ValidationError
from app.db.base import utcnow
from app.ml.features import resolve_data_mode
from app.models.epi import (
    ClimateObservation,
    DataQualityRecord,
    DataSource,
    DengueCase,
    ImportJob,
    ImportRowError,
)
from app.models.geo import GeographicUnit
from app.services import audit as audit_service

logger = logging.getLogger("dengai.data")

CASE_COLUMNS = ["district_code", "epi_year", "epi_week", "confirmed_cases"]
CASE_OPTIONAL = ["probable_cases", "hospitalized", "deaths", "severity"]
CLIMATE_COLUMNS = ["district_code", "epi_year", "epi_week"]
CLIMATE_OPTIONAL = [
    "temperature_mean", "temperature_max", "temperature_min", "humidity_mean",
    "precipitation_mm", "wind_speed", "solar_radiation",
]


def _read_dataframe(raw: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValidationError("No fue posible decodificar el archivo CSV (use UTF-8 o Latin-1).")
    if name.endswith(".json"):
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValidationError(f"JSON inválido: {exc}") from exc
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        return pd.DataFrame(payload)
    raise ValidationError("Formato no soportado. Use CSV o JSON con registros distrito-semana.")


def preview(raw: bytes, filename: str, kind: str = "epidemiological") -> Dict[str, Any]:
    frame = _read_dataframe(raw, filename)
    required = CASE_COLUMNS if kind == "epidemiological" else CLIMATE_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    validation: List[Dict[str, Any]] = []
    if missing:
        validation.append(
            {"level": "ERROR", "column": ", ".join(missing), "message": "Faltan columnas obligatorias.", "count": len(missing)}
        )
    if "district_code" in frame.columns:
        known = set()
        validation.append(
            {
                "level": "INFO",
                "column": "district_code",
                "message": f"{frame['district_code'].nunique()} distritos distintos en el archivo.",
                "count": int(frame["district_code"].nunique()),
            }
        )
    duplicates = int(frame.duplicated(subset=[c for c in required if c in frame.columns]).sum()) if not missing else 0
    if duplicates:
        validation.append(
            {"level": "WARN", "column": "-", "message": "Filas duplicadas que se consolidarán.", "count": duplicates}
        )
    nulls = int(frame.isna().sum().sum())
    if nulls:
        validation.append({"level": "WARN", "column": "-", "message": "Valores faltantes detectados.", "count": nulls})
    return {
        "kind": kind,
        "columns": [str(c) for c in frame.columns],
        "sample_rows": frame.head(8).fillna("").astype(str).to_dict(orient="records"),
        "detected_rows": int(len(frame)),
        "validation": validation,
        "ready": not missing,
    }


def _job_key(filename: str, digest: str) -> str:
    return f"imp-{digest[:10]}-{int(datetime.now(timezone.utc).timestamp())}"


def import_cases(
    db: Session,
    *,
    raw: bytes,
    filename: str,
    source_code: Optional[str],
    actor: str,
    is_synthetic: bool = False,
    ip_address: str = "",
    request_id: str = "",
) -> Dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    frame = _read_dataframe(raw, filename)
    missing = [column for column in CASE_COLUMNS if column not in frame.columns]
    job = ImportJob(
        job_key=_job_key(filename, digest),
        dataset_name=filename,
        kind="epidemiological",
        status="RUNNING",
        file_name=filename,
        file_hash=digest,
        rows_total=int(len(frame)),
        initiated_by=actor,
        is_synthetic=is_synthetic,
        source_id=(db.scalar(select(DataSource.id).where(DataSource.code == source_code)) if source_code else None),
    )
    db.add(job)
    db.flush()

    if missing:
        job.status = "FAILED"
        job.error_message = f"Columnas obligatorias ausentes: {', '.join(missing)}"
        job.finished_at = utcnow()
        job.rows_rejected = int(len(frame))
        db.commit()
        return _job_payload(db, job)

    units = {unit.code: unit for unit in db.scalars(select(GeographicUnit))}
    inserted = updated = rejected = duplicated = 0
    errors: List[ImportRowError] = []
    seen: set = set()

    for index, row in frame.iterrows():
        row_number = int(index) + 2  # +1 por encabezado, +1 por índice base 0
        code = str(row["district_code"]).strip()
        if code not in units:
            rejected += 1
            errors.append(ImportRowError(job_id=job.id, row_number=row_number, column="district_code", reason="Distrito no registrado en el catálogo geográfico.", raw_value=code[:255]))
            continue
        try:
            year = int(row["epi_year"])
            week = int(row["epi_week"])
            confirmed = int(float(row["confirmed_cases"]))
        except (TypeError, ValueError):
            rejected += 1
            errors.append(ImportRowError(job_id=job.id, row_number=row_number, column="epi_year/epi_week/confirmed_cases", reason="Valor no numérico en campos clave.", raw_value=str(row.to_dict())[:255]))
            continue
        if not (1 <= week <= 53) or not (2000 <= year <= 2100):
            rejected += 1
            errors.append(ImportRowError(job_id=job.id, row_number=row_number, column="epi_week", reason="Semana o año epidemiológico fuera de rango.", raw_value=f"{year}-W{week}"))
            continue
        if confirmed < 0 or confirmed > 100_000:
            rejected += 1
            errors.append(ImportRowError(job_id=job.id, row_number=row_number, column="confirmed_cases", reason="Casos fuera de rango plausible.", raw_value=str(confirmed)))
            continue
        key = (code, year, week)
        if key in seen:
            duplicated += 1
            continue
        seen.add(key)
        week_start = _week_start(year, week)
        existing = db.scalar(
            select(DengueCase).where(
                DengueCase.district_code == code, DengueCase.epi_year == year, DengueCase.epi_week == week
            )
        )
        payload = {
            "confirmed_cases": confirmed,
            "probable_cases": _optional_int(row, "probable_cases"),
            "hospitalized": _optional_int(row, "hospitalized"),
            "deaths": _optional_int(row, "deaths"),
            "severity": str(row.get("severity") or "")[:20],
            "source_id": job.source_id,
            "import_job_id": job.id,
            "is_synthetic": is_synthetic,
        }
        if existing is None:
            db.add(
                DengueCase(
                    district_code=code, epi_year=year, epi_week=week, week_start=week_start, **payload
                )
            )
            inserted += 1
        else:
            for field, value in payload.items():
                if value is not None:
                    setattr(existing, field, value)
            updated += 1
        if len(errors) > 500:
            break

    job.rows_inserted = inserted
    job.rows_updated = updated
    job.rows_rejected = rejected
    job.rows_duplicated = duplicated
    job.status = "COMPLETED" if inserted + updated > 0 else "FAILED"
    job.finished_at = utcnow()
    job.duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
    job.log = {"errors": len(errors), "source": source_code}
    for error in errors[:500]:
        db.add(error)

    if job.status == "COMPLETED":
        _record_quality(db, job, frame)
        _mark_source_success(db, source_code)
    db.commit()

    audit_service.record(
        db, event_type=audit_service.AuditEventType.DATA_IMPORT, module="M05", action="import_cases",
        result="SUCCESS" if job.status == "COMPLETED" else "ERROR", user_email=actor,
        resource_type="import_job", resource_id=job.id, ip_address=ip_address, request_id=request_id,
        description=(
            f"Importación de casos: {inserted} insertados, {updated} actualizados, "
            f"{rejected} rechazados, {duplicated} duplicados."
        ),
        context={"file": filename, "source": source_code, "synthetic": is_synthetic},
    )
    return _job_payload(db, job)


def import_climate(
    db: Session,
    *,
    raw: bytes,
    filename: str,
    source_code: Optional[str],
    actor: str,
    is_synthetic: bool = False,
    ip_address: str = "",
    request_id: str = "",
) -> Dict[str, Any]:
    digest = hashlib.sha256(raw).hexdigest()
    frame = _read_dataframe(raw, filename)
    missing = [column for column in CLIMATE_COLUMNS if column not in frame.columns]
    job = ImportJob(
        job_key=_job_key(filename, digest), dataset_name=filename, kind="climate", status="RUNNING",
        file_name=filename, file_hash=digest, rows_total=int(len(frame)), initiated_by=actor,
        is_synthetic=is_synthetic,
        source_id=(db.scalar(select(DataSource.id).where(DataSource.code == source_code)) if source_code else None),
    )
    db.add(job)
    db.flush()
    if missing:
        job.status = "FAILED"
        job.error_message = f"Columnas obligatorias ausentes: {', '.join(missing)}"
        job.finished_at = utcnow()
        db.commit()
        return _job_payload(db, job)

    units = {unit.code: unit for unit in db.scalars(select(GeographicUnit))}
    inserted = updated = rejected = duplicated = 0
    seen: set = set()
    for index, row in frame.iterrows():
        code = str(row["district_code"]).strip()
        if code not in units:
            rejected += 1
            continue
        try:
            year, week = int(row["epi_year"]), int(row["epi_week"])
        except (TypeError, ValueError):
            rejected += 1
            continue
        key = (code, year, week)
        if key in seen:
            duplicated += 1
            continue
        seen.add(key)
        values = {column: _optional_float(row, column) for column in CLIMATE_OPTIONAL}
        if all(value is None for value in values.values()):
            rejected += 1
            continue
        existing = db.scalar(
            select(ClimateObservation).where(
                ClimateObservation.district_code == code,
                ClimateObservation.epi_year == year,
                ClimateObservation.epi_week == week,
            )
        )
        payload = {
            **values,
            "altitude_m": units[code].altitude_m,
            "provider": source_code or "file",
            "source_id": job.source_id,
            "import_job_id": job.id,
            "is_synthetic": is_synthetic,
        }
        if existing is None:
            db.add(
                ClimateObservation(district_code=code, epi_year=year, epi_week=week, week_start=_week_start(year, week), **payload)
            )
            inserted += 1
        else:
            for field, value in payload.items():
                if value is not None:
                    setattr(existing, field, value)
            updated += 1

    job.rows_inserted, job.rows_updated = inserted, updated
    job.rows_rejected, job.rows_duplicated = rejected, duplicated
    job.status = "COMPLETED" if inserted + updated > 0 else "FAILED"
    job.finished_at = utcnow()
    job.duration_ms = int((job.finished_at - job.started_at).total_seconds() * 1000)
    if job.status == "COMPLETED":
        _record_quality(db, job, frame)
        _mark_source_success(db, source_code)
    db.commit()
    audit_service.record(
        db, event_type=audit_service.AuditEventType.DATA_IMPORT, module="M05", action="import_climate",
        result="SUCCESS" if job.status == "COMPLETED" else "ERROR", user_email=actor,
        resource_type="import_job", resource_id=job.id, ip_address=ip_address, request_id=request_id,
        description=f"Importación climática: {inserted} insertados, {updated} actualizados, {rejected} rechazados.",
        context={"file": filename, "source": source_code},
    )
    return _job_payload(db, job)


def _mark_source_success(db: Session, source_code: Optional[str]) -> None:
    if not source_code:
        return
    source = db.scalar(select(DataSource).where(DataSource.code == source_code))
    if source is not None:
        source.last_success_at = utcnow()
        source.last_error = ""


def _record_quality(db: Session, job: ImportJob, frame: pd.DataFrame) -> None:
    total = len(frame)
    nulls = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    db.add(
        DataQualityRecord(
            import_job_id=job.id, dataset_name=job.dataset_name, check_name="valores_faltantes",
            severity="WARN" if nulls else "INFO", rows_checked=total, rows_flagged=nulls,
            details={"total_nulls": nulls}, created_by=job.initiated_by,
        )
    )
    db.add(
        DataQualityRecord(
            import_job_id=job.id, dataset_name=job.dataset_name, check_name="duplicados_exactos",
            severity="WARN" if duplicates else "INFO", rows_checked=total, rows_flagged=duplicates,
            details={"duplicates": duplicates}, created_by=job.initiated_by,
        )
    )
    if "confirmed_cases" in frame.columns:
        series = pd.to_numeric(frame["confirmed_cases"], errors="coerce").dropna()
        if len(series) > 10:
            mean, std = float(series.mean()), float(series.std() or 0)
            anomalies = int((np.abs((series - mean) / (std or 1)) > 4).sum())
            db.add(
                DataQualityRecord(
                    import_job_id=job.id, dataset_name=job.dataset_name, check_name="anomalias_zscore",
                    severity="WARN" if anomalies else "INFO", rows_checked=int(len(series)),
                    rows_flagged=anomalies,
                    details={"threshold": 4, "mean": round(mean, 3), "std": round(std, 3)},
                    created_by=job.initiated_by,
                )
            )


def _week_start(year: int, week: int):
    from datetime import date, timedelta

    try:
        return date.fromisocalendar(year, week, 1)
    except ValueError:
        return date(year, 1, 1) + timedelta(weeks=week - 1)


def _optional_int(row: pd.Series, column: str) -> Optional[int]:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_float(row: pd.Series, column: str) -> Optional[float]:
    value = row.get(column)
    if value is None or pd.isna(value):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _job_payload(db: Session, job: ImportJob) -> Dict[str, Any]:
    source = db.get(DataSource, job.source_id) if job.source_id else None
    return {
        "id": job.id,
        "job_key": job.job_key,
        "source_code": source.code if source else None,
        "dataset_name": job.dataset_name,
        "kind": job.kind,
        "status": job.status,
        "file_name": job.file_name,
        "rows_total": job.rows_total,
        "rows_inserted": job.rows_inserted,
        "rows_updated": job.rows_updated,
        "rows_rejected": job.rows_rejected,
        "rows_duplicated": job.rows_duplicated,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "duration_ms": job.duration_ms,
        "initiated_by": job.initiated_by,
        "is_synthetic": job.is_synthetic,
        "error_message": job.error_message,
    }


def list_imports(db: Session, *, page: int = 1, size: int = 20, kind: Optional[str] = None) -> Dict[str, Any]:
    stmt = select(ImportJob)
    if kind:
        stmt = stmt.where(ImportJob.kind == kind)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = [_job_payload(db, job) for job in db.scalars(stmt.order_by(ImportJob.started_at.desc()).offset((page - 1) * size).limit(size))]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def import_errors(db: Session, job_id: int, *, limit: int = 200) -> List[Dict[str, Any]]:
    rows = db.scalars(select(ImportRowError).where(ImportRowError.job_id == job_id).limit(limit))
    return [
        {"row_number": row.row_number, "column": row.column, "reason": row.reason, "raw_value": row.raw_value}
        for row in rows
    ]


def quality_summary(db: Session) -> Dict[str, Any]:
    use_synthetic = resolve_data_mode(db) == "synthetic"
    total_cases = int(
        db.scalar(select(func.count()).select_from(DengueCase).where(DengueCase.is_synthetic.is_(use_synthetic))) or 0
    )
    total_climate = int(
        db.scalar(
            select(func.count())
            .select_from(ClimateObservation)
            .where(ClimateObservation.is_synthetic.is_(use_synthetic))
        )
        or 0
    )
    districts = int(
        db.scalar(
            select(func.count()).select_from(
                select(DengueCase.district_code)
                .where(DengueCase.is_synthetic.is_(use_synthetic))
                .distinct()
                .subquery()
            )
        )
        or 0
    ) or int(db.scalar(select(func.count()).select_from(GeographicUnit)) or 0)
    weeks = int(
        db.scalar(
            select(func.count()).select_from(
                select(DengueCase.week_start).where(DengueCase.is_synthetic.is_(use_synthetic)).distinct().subquery()
            )
        )
        or 0
    )
    synthetic = total_cases if use_synthetic else 0
    missing_climate = int(
        db.scalar(
            select(func.count())
            .select_from(ClimateObservation)
            .where(
                ClimateObservation.is_synthetic.is_(use_synthetic),
                ClimateObservation.temperature_mean.is_(None),
            )
        )
        or 0
    )
    checks = list(db.scalars(select(DataQualityRecord).order_by(DataQualityRecord.created_at.desc()).limit(20)))
    return {
        "total_cases": total_cases,
        "total_climate": total_climate,
        "districts_covered": districts,
        "weeks_covered": weeks,
        "missing_climate_pct": round(missing_climate / total_climate * 100, 2) if total_climate else 0.0,
        "duplicate_rows": 0,
        "anomalies": sum(record.rows_flagged for record in checks if record.check_name == "anomalias_zscore"),
        "synthetic_share_pct": round(synthetic / total_cases * 100, 2) if total_cases else 0.0,
        "checks": [
            {
                "id": record.id,
                "dataset_name": record.dataset_name,
                "check_name": record.check_name,
                "severity": record.severity,
                "rows_checked": record.rows_checked,
                "rows_flagged": record.rows_flagged,
                "details": record.details,
                "created_at": record.created_at,
            }
            for record in checks
        ],
    }


def list_cases(
    db: Session,
    *,
    district: Optional[str] = None,
    start_week: Optional[Any] = None,
    end_week: Optional[Any] = None,
    page: int = 1,
    size: int = 50,
) -> Dict[str, Any]:
    use_synthetic = resolve_data_mode(db) == "synthetic"
    stmt = select(DengueCase).where(DengueCase.is_synthetic.is_(use_synthetic))
    if district:
        stmt = stmt.where(DengueCase.district_code == district)
    if start_week:
        stmt = stmt.where(DengueCase.week_start >= start_week)
    if end_week:
        stmt = stmt.where(DengueCase.week_start <= end_week)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    units = {u.code: u for u in db.scalars(select(GeographicUnit))}
    rows = [
        {
            "district_code": row.district_code,
            "district_name": units[row.district_code].name if row.district_code in units else row.district_code,
            "epi_year": row.epi_year,
            "epi_week": row.epi_week,
            "week_start": row.week_start,
            "confirmed_cases": row.confirmed_cases,
            "probable_cases": row.probable_cases,
            "hospitalized": row.hospitalized,
            "deaths": row.deaths,
            "is_synthetic": row.is_synthetic,
            "data_quality_flag": row.data_quality_flag,
        }
        for row in db.scalars(
            stmt.order_by(DengueCase.week_start.desc(), DengueCase.district_code).offset((page - 1) * size).limit(size)
        )
    ]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def list_climate(
    db: Session,
    *,
    district: Optional[str] = None,
    page: int = 1,
    size: int = 50,
) -> Dict[str, Any]:
    use_synthetic = resolve_data_mode(db) == "synthetic"
    stmt = select(ClimateObservation).where(ClimateObservation.is_synthetic.is_(use_synthetic))
    if district:
        stmt = stmt.where(ClimateObservation.district_code == district)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = [
        {
            "district_code": row.district_code,
            "epi_year": row.epi_year,
            "epi_week": row.epi_week,
            "week_start": row.week_start,
            "temperature_mean": row.temperature_mean,
            "temperature_max": row.temperature_max,
            "temperature_min": row.temperature_min,
            "humidity_mean": row.humidity_mean,
            "precipitation_mm": row.precipitation_mm,
            "provider": row.provider,
            "is_synthetic": row.is_synthetic,
        }
        for row in db.scalars(
            stmt.order_by(ClimateObservation.week_start.desc()).offset((page - 1) * size).limit(size)
        )
    ]
    return {"items": rows, "total": total, "page": page, "size": size, "pages": max(1, (total + size - 1) // size)}


def list_sources(db: Session) -> List[Dict[str, Any]]:
    rows = []
    for source in db.scalars(select(DataSource).order_by(DataSource.kind, DataSource.code)):
        rows.append(
            {
                "id": source.id,
                "code": source.code,
                "name": source.name,
                "kind": source.kind,
                "provider": source.provider,
                "url": source.url,
                "license": source.license,
                "access_method": source.access_method,
                "integration_status": source.integration_status,
                "is_synthetic": source.is_synthetic,
                "refresh_frequency": source.refresh_frequency,
                "last_success_at": source.last_success_at,
                "last_error": source.last_error,
                "notes": source.license_notes,
            }
        )
    return rows
