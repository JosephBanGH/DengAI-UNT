"""Orquestación de la ingesta real (MINSA/CDC Perú + NASA POWER) y procedencia.

Reglas de honestidad:
* Todo dato externo se descarga primero a ``data/raw`` con checksum y manifiesto.
* La base de datos guarda ``is_synthetic=False`` solo para filas efectivamente
  derivadas de la fuente oficial.
* El modo efectivo del pipeline (``real``/``synthetic``) se resuelve de forma
  homogénea: nunca se mezclan ambos orígenes en el mismo panel.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.integrations import minsa_dengue, nasa_power, raw_store
from app.ml.features import load_panel, resolve_data_mode
from app.models.epi import ClimateObservation, DataSource, DengueCase, ImportJob
from app.models.geo import GeographicUnit

logger = logging.getLogger("dengai.real_data")

REAL_SOURCE_CODES = ("MINSA_VIGILANCIA", "NASA_POWER")


def _raw_path(entry: Dict[str, Any]):
    return settings.raw_dir.parent.parent / entry["path"]


def ingest_real_data(
    db: Session,
    *,
    actor: str = "system",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    district_codes: Optional[Sequence[str]] = None,
    include_cases: bool = True,
    include_climate: bool = True,
    force_download: bool = False,
) -> Dict[str, Any]:
    """Ejecuta la ingesta real completa y devuelve evidencia por paso.

    No se declara éxito si un paso falla: cada resultado incluye estado, conteos y
    errores. La ingesta de casos reemplaza explícitamente el rango solicitado para
    evitar colisiones con el conjunto sintético.
    """
    start_year = int(start_year or settings.MINSA_START_YEAR)
    end_year = int(end_year or settings.MINSA_END_YEAR)
    if end_year < start_year:
        raise ValueError("end_year debe ser mayor o igual que start_year.")

    result: Dict[str, Any] = {
        "requested": {
            "start_year": start_year,
            "end_year": end_year,
            "districts": list(district_codes) if district_codes else "all_catalog",
            "include_cases": include_cases,
            "include_climate": include_climate,
        },
        "steps": {},
    }

    if include_cases:
        entry = minsa_dengue.download_raw(force=force_download)
        path = _raw_path(entry)
        logger.info("Ingiriendo MINSA desde %s", path)
        result["steps"]["minsa"] = minsa_dengue.ingest(
            db,
            path=path,
            start_year=start_year,
            end_year=end_year,
            actor=actor,
            district_codes=district_codes,
        )
        result["steps"]["minsa"]["raw"] = {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "bytes": entry["bytes"],
            "url": entry["url"],
            "fetched_at": entry["fetched_at"],
        }

    if include_climate:
        stmt = select(GeographicUnit).order_by(GeographicUnit.code)
        units: List[GeographicUnit] = list(db.scalars(stmt))
        if district_codes:
            wanted = set(district_codes)
            units = [unit for unit in units if unit.code in wanted]
        result["steps"]["nasa_power"] = nasa_power.ingest_nasa_power(
            db,
            units=units,
            start=date(start_year, 1, 1),
            end=date(end_year, 12, 31),
            actor=actor,
        )

    result["effective_data_mode"] = resolve_data_mode(db)
    result["provenance"] = provenance(db)
    return result


def provenance(db: Session) -> Dict[str, Any]:
    """Resumen auditable del origen real/sintético de casos y clima."""
    cases_total = int(db.scalar(select(func.count()).select_from(DengueCase)) or 0)
    cases_real = int(
        db.scalar(select(func.count()).select_from(DengueCase).where(DengueCase.is_synthetic.is_(False))) or 0
    )
    climate_total = int(db.scalar(select(func.count()).select_from(ClimateObservation)) or 0)
    climate_real = int(
        db.scalar(
            select(func.count()).select_from(ClimateObservation).where(ClimateObservation.is_synthetic.is_(False))
        )
        or 0
    )

    case_range = db.execute(
        select(func.min(DengueCase.week_start), func.max(DengueCase.week_start)).where(
            DengueCase.is_synthetic.is_(False)
        )
    ).one()
    climate_range = db.execute(
        select(func.min(ClimateObservation.week_start), func.max(ClimateObservation.week_start)).where(
            ClimateObservation.is_synthetic.is_(False)
        )
    ).one()

    case_by_source = [
        {"source_code": code, "rows": int(rows)}
        for code, rows in db.execute(
            select(DataSource.code, func.count(DengueCase.id))
            .join(DengueCase, DengueCase.source_id == DataSource.id)
            .group_by(DataSource.code)
            .order_by(DataSource.code)
        ).all()
    ]
    climate_by_provider = [
        {"provider": provider, "rows": int(rows), "is_synthetic": bool(synthetic)}
        for provider, synthetic, rows in db.execute(
            select(
                ClimateObservation.provider,
                ClimateObservation.is_synthetic,
                func.count(ClimateObservation.id),
            )
            .group_by(ClimateObservation.provider, ClimateObservation.is_synthetic)
            .order_by(ClimateObservation.provider)
        ).all()
    ]

    last_jobs = [
        {
            "id": job.id,
            "source_code": job.source_id,
            "dataset_name": job.dataset_name,
            "kind": job.kind,
            "status": job.status,
            "rows_total": job.rows_total,
            "rows_inserted": job.rows_inserted,
            "rows_updated": job.rows_updated,
            "is_synthetic": job.is_synthetic,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error_message": job.error_message,
        }
        for job in db.scalars(
            select(ImportJob)
            .where(ImportJob.is_synthetic.is_(False))
            .order_by(ImportJob.started_at.desc())
            .limit(10)
        )
    ]

    manifest = raw_store.read_manifest()
    return {
        "cases": {
            "total": cases_total,
            "real": cases_real,
            "synthetic": cases_total - cases_real,
            "real_share_pct": round(cases_real / cases_total * 100, 2) if cases_total else 0.0,
            "real_week_range": [
                str(case_range[0]) if case_range[0] else None,
                str(case_range[1]) if case_range[1] else None,
            ],
            "by_source": case_by_source,
        },
        "climate": {
            "total": climate_total,
            "real": climate_real,
            "synthetic": climate_total - climate_real,
            "real_share_pct": round(climate_real / climate_total * 100, 2) if climate_total else 0.0,
            "real_week_range": [
                str(climate_range[0]) if climate_range[0] else None,
                str(climate_range[1]) if climate_range[1] else None,
            ],
            "by_provider": climate_by_provider,
        },
        "raw_manifest": manifest[-20:],
        "raw_entries": len(manifest),
        "last_real_imports": last_jobs,
        "effective_data_mode": resolve_data_mode(db),
    }


def panel_summary(db: Session, *, mode: Optional[str] = None) -> Dict[str, Any]:
    """Resumen del panel que realmente usaría el entrenamiento."""
    panel = load_panel(db, mode=mode)
    panel_frame = panel.panel
    per_district = [
        {
            "district_code": code,
            "district_name": panel.district_names.get(code, code),
            "rows": int(count),
            "real_rows": int(
                (~panel_frame.loc[panel_frame["district_code"] == code, "is_synthetic"].astype(bool)).sum()
            ),
        }
        for code, count in panel_frame.groupby("district_code").size().items()
    ]
    return {
        "data_mode": panel.data_mode,
        "is_synthetic": panel.is_synthetic,
        "rows": int(len(panel_frame)),
        "districts": len(panel.districts),
        "weeks": len(panel.weeks),
        "start_week": str(panel.weeks[0]) if panel.weeks else None,
        "end_week": str(panel.weeks[-1]) if panel.weeks else None,
        "source_summary": panel.source_summary,
        "per_district": per_district,
    }
