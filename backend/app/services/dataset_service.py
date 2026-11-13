"""Conjuntos de datos versionados y diccionario de datos (M06/M16)."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError, ValidationError
from app.ml.features import build_dataset, build_features, load_panel
from app.models.epi import Dataset
from app.services import audit as audit_service

FEATURE_DOCS: Dict[str, Dict[str, str]] = {
    "confirmed_cases": {"description": "Casos confirmados de dengue reportados en la semana", "unit": "casos", "source": "MINSA/CDC o sintético"},
    "cases_lag_1": {"description": "Casos confirmados una semana antes del origen", "unit": "casos", "source": "derivada"},
    "cases_mean_4": {"description": "Promedio de casos de las 4 semanas previas", "unit": "casos", "source": "derivada"},
    "cases_std_4": {"description": "Desviación estándar de casos en 4 semanas", "unit": "casos", "source": "derivada"},
    "neighbour_cases_lag_1": {"description": "Casos promedio de distritos vecinos en la semana previa", "unit": "casos", "source": "derivada"},
    "temperature_mean": {"description": "Temperatura media semanal a 2 m", "unit": "°C", "source": "NASA POWER"},
    "humidity_mean": {"description": "Humedad relativa media semanal", "unit": "%", "source": "NASA POWER"},
    "precipitation_mm": {"description": "Precipitación acumulada semanal", "unit": "mm", "source": "CHIRPS / NASA POWER"},
    "altitude_km": {"description": "Altitud del distrito en kilómetros", "unit": "km", "source": "catálogo geográfico"},
    "population_density": {"description": "Densidad poblacional del distrito", "unit": "hab/km²", "source": "WorldPop / catálogo"},
    "log_population": {"description": "Logaritmo natural de la población", "unit": "-", "source": "derivada"},
    "sin_week": {"description": "Codificación cíclica de la semana epidemiológica (seno)", "unit": "-", "source": "derivada"},
    "cos_week": {"description": "Codificación cíclica de la semana epidemiológica (coseno)", "unit": "-", "source": "derivada"},
    "time_index": {"description": "Índice temporal de la semana (tendencia)", "unit": "-", "source": "derivada"},
    "target": {"description": "Casos confirmados en la semana objetivo (t + h)", "unit": "casos", "source": "objetivo"},
}


def build_and_register(db: Session, *, name: str, description: str, max_lag: int, actor: str) -> Dict[str, Any]:
    panel = load_panel(db)
    frame = build_features(panel, horizon=1, max_lag=max_lag)
    frame = frame.dropna(subset=["target"])
    if frame.empty:
        raise ValidationError("No hay observaciones suficientes para construir el conjunto de datos.")
    columns = [c for c in frame.columns if c not in ("target", "target_week", "origin_week")]
    checksum = hashlib.sha256(
        json.dumps([str(v) for v in frame["confirmed_cases"].tolist()], sort_keys=True).encode()
    ).hexdigest()

    slug = name.lower().replace(" ", "-")[:60] + "-" + checksum[:6]
    existing = db.scalar(select(Dataset).where(Dataset.slug == slug))
    feature_schema = {
        "columns": [c for c in frame.columns if c not in ("target", "target_week", "origin_week", "district_code")],
        "target": "target",
        "unit": "distrito-semana",
        "max_lag": max_lag,
        "rows": int(len(frame)),
    }
    data_dictionary = {
        column: FEATURE_DOCS.get(
            column,
            {
                "description": f"Variable derivada {column}",
                "unit": "",
                "source": "derivada" if "lag" in column or "mean" in column else "plataforma",
            },
        )
        for column in feature_schema["columns"]
    }
    payload = {
        "slug": slug,
        "name": name,
        "version": "1.0.0",
        "description": description,
        "rows": int(len(frame)),
        "districts": int(frame["district_code"].nunique()),
        "weeks": int(frame["origin_week"].nunique()),
        "start_week": frame["origin_week"].min(),
        "end_week": frame["origin_week"].max(),
        "is_synthetic": bool(panel.is_synthetic),
        "source_summary": panel.source_summary,
        "feature_schema": feature_schema,
        "data_dictionary": data_dictionary,
        "checksum": checksum,
        "status": "READY",
        "created_by": actor,
    }
    if existing is None:
        record = Dataset(**payload)
        db.add(record)
    else:
        record = existing
        for key, value in payload.items():
            setattr(record, key, value)
    db.commit()
    audit_service.record(
        db, event_type=audit_service.AuditEventType.CREATE, module="M06", action="build_dataset",
        result="SUCCESS", user_email=actor, resource_type="dataset", resource_id=record.id,
        description=f"Conjunto de datos construido: {name} ({len(frame)} filas, {payload['districts']} distritos).",
    )
    db.commit()
    return _payload(record)


def _payload(record: Dataset) -> Dict[str, Any]:
    return {
        "id": record.id,
        "slug": record.slug,
        "name": record.name,
        "version": record.version,
        "description": record.description,
        "rows": record.rows,
        "districts": record.districts,
        "weeks": record.weeks,
        "start_week": record.start_week,
        "end_week": record.end_week,
        "is_synthetic": record.is_synthetic,
        "status": record.status,
        "checksum": record.checksum,
        "created_at": record.created_at,
        "created_by": record.created_by,
        "source_summary": record.source_summary,
        "feature_schema": record.feature_schema,
    }


def list_datasets(db: Session) -> List[Dict[str, Any]]:
    return [_payload(record) for record in db.scalars(select(Dataset).order_by(Dataset.created_at.desc()))]


def get_dataset(db: Session, dataset_id: int) -> Dict[str, Any]:
    record = db.get(Dataset, dataset_id)
    if record is None:
        raise NotFoundError("El conjunto de datos no existe.")
    return _payload(record)


def data_dictionary(db: Session, dataset_id: int) -> List[Dict[str, Any]]:
    record = db.get(Dataset, dataset_id)
    if record is None:
        raise NotFoundError("El conjunto de datos no existe.")
    dictionary = record.data_dictionary or {}
    return [
        {
            "column": column,
            "dtype": "float64",
            "description": meta.get("description", ""),
            "unit": meta.get("unit", ""),
            "source": meta.get("source", ""),
            "nullable": column != "target",
        }
        for column, meta in sorted(dictionary.items())
    ]


def update_data_dictionary_entry(
    db: Session,
    dataset_id: int,
    column: str,
    *,
    description: str,
    unit: str,
    source: str,
    actor: str,
) -> Dict[str, Any]:
    """Actualiza la definición de metadatos de una variable en el diccionario de datos."""
    record = db.get(Dataset, dataset_id)
    if record is None:
        raise NotFoundError("El conjunto de datos no existe.")
    current_dict = dict(record.data_dictionary or {})
    current_dict[column] = {
        "description": description,
        "unit": unit,
        "source": source,
    }
    record.data_dictionary = current_dict
    db.commit()
    audit_service.record(
        db,
        event_type=audit_service.AuditEventType.UPDATE,
        module="M16",
        action="update_data_dictionary",
        result="SUCCESS",
        user_email=actor,
        resource_type="dataset_dictionary",
        resource_id=record.id,
        description=f"Metadatos de variable '{column}' actualizados en conjunto {record.name}.",
        context={"column": column, "unit": unit, "source": source},
    )
    db.commit()
    return {
        "column": column,
        "dtype": "float64",
        "description": description,
        "unit": unit,
        "source": source,
        "nullable": column != "target",
    }

