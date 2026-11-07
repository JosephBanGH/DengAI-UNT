"""Almacenamiento de artefactos en MinIO (opcional, activado por configuración).

Si MinIO no está habilitado o el cliente no está instalado, las funciones devuelven
``None``/``available=False`` sin interrumpir el entrenamiento. La fuente de verdad
sigue siendo el artefacto local con checksum; MinIO es una copia auditable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings

logger = logging.getLogger("dengai.minio")


def _client():
    if not settings.MINIO_ENABLED:
        return None
    try:
        from minio import Minio  # type: ignore
    except ImportError:
        logger.info("Cliente MinIO no instalado; se omite la copia de artefactos.")
        return None
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def ensure_bucket() -> bool:
    client = _client()
    if client is None:
        return False
    try:
        if not client.bucket_exists(settings.MINIO_BUCKET):
            client.make_bucket(settings.MINIO_BUCKET)
        return True
    except Exception:
        logger.exception("No se pudo verificar/crear el bucket MinIO")
        return False


def upload_file(local_path: str | Path, *, object_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Sube un archivo y devuelve ``bucket/object`` si la operación fue verificable."""
    client = _client()
    if client is None:
        return None
    path = Path(local_path)
    if not path.exists():
        return None
    object_name = object_name or f"artifacts/{path.name}"
    try:
        if not ensure_bucket():
            return None
        client.fput_object(settings.MINIO_BUCKET, object_name, str(path))
        stat = client.stat_object(settings.MINIO_BUCKET, object_name)
        return {
            "bucket": settings.MINIO_BUCKET,
            "object": object_name,
            "uri": f"s3://{settings.MINIO_BUCKET}/{object_name}",
            "size": int(stat.size),
            "etag": stat.etag,
        }
    except Exception:
        logger.exception("No se pudo subir %s a MinIO", path)
        return None


def health() -> Dict[str, Any]:
    client = _client()
    if client is None:
        return {
            "enabled": bool(settings.MINIO_ENABLED),
            "available": False,
            "reason": "MinIO deshabilitado o cliente no instalado.",
            "endpoint": settings.MINIO_ENDPOINT,
        }
    try:
        exists = client.bucket_exists(settings.MINIO_BUCKET)
        return {
            "enabled": True,
            "available": True,
            "bucket": settings.MINIO_BUCKET,
            "bucket_exists": bool(exists),
            "endpoint": settings.MINIO_ENDPOINT,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "endpoint": settings.MINIO_ENDPOINT,
        }
