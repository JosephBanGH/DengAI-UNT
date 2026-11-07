"""Capa raw inmutable: guarda bytes descargados con checksum SHA-256 y manifiesto.

Regla de honestidad: todo dato externo se persiste primero en ``data/raw`` con su
URL, licencia, fecha de descarga y checksum. La ingesta a la base de datos solo lee
desde esta capa, de modo que cualquier resultado es reproducible y auditable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("dengai.raw")

_LOCK = threading.Lock()


def manifest_path() -> Path:
    return settings.raw_dir / "manifest.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest() -> List[Dict[str, Any]]:
    path = manifest_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Manifiesto raw ilegible; se inicia uno nuevo")
        return []
    return payload if isinstance(payload, list) else []


def append_manifest(entry: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        entries = read_manifest()
        entries.append(entry)
        manifest_path().write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def save_raw_bytes(
    *,
    source: str,
    filename: str,
    content: bytes,
    url: str,
    license: str,
    provider: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persiste ``content`` sin modificaciones y registra su procedencia."""
    directory = settings.raw_dir / source
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    digest = hashlib.sha256(content).hexdigest()
    if path.exists():
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing == digest:
            logger.info("Raw ya existe e idéntico: %s", path)
        else:
            # Nunca se sobrescribe un raw con contenido distinto: se versiona.
            path = directory / f"{path.stem}-{digest[:12]}{path.suffix}"
            path.write_bytes(content)
    else:
        path.write_bytes(content)
    entry = {
        "source": source,
        "provider": provider,
        "url": url,
        "license": license,
        "path": str(path.relative_to(settings.raw_dir.parent.parent)),
        "filename": path.name,
        "bytes": len(content),
        "sha256": digest,
        "fetched_at": _now(),
        "metadata": metadata or {},
    }
    return append_manifest(entry)


def save_raw_json(
    *,
    source: str,
    filename: str,
    payload: Any,
    url: str,
    license: str,
    provider: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return save_raw_bytes(
        source=source,
        filename=filename,
        content=content,
        url=url,
        license=license,
        provider=provider,
        metadata=metadata,
    )


def find_entry(source: str, *, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
    entries = [item for item in read_manifest() if item.get("source") == source]
    if filename:
        entries = [item for item in entries if item.get("filename") == filename]
    return entries[-1] if entries else None
