"""Utilidades de configuración reproducible y límites de cómputo."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_reproducible_seed(seed: int = 42) -> None:
    """Fija semillas en todas las librerías disponibles para reproducibilidad."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # algunas ops de CPU no tienen kernel determinista
        if torch.cuda.is_available():  # pragma: no cover - no hay GPU en el entorno
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def torch_info() -> dict:
    try:
        import torch

        return {"available": True, "version": torch.__version__, "device": "cpu", "cuda": bool(torch.cuda.is_available())}
    except Exception as exc:  # pragma: no cover
        return {"available": False, "version": None, "device": None, "error": str(exc)}


def configure_threads(threads: Optional[int] = None) -> int:
    """Limita hilos para que el entrenamiento no degrade el servidor web."""
    value = threads or max(1, min(4, (os.cpu_count() or 4) - 1))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(value))
    try:
        import torch

        torch.set_num_threads(value)
    except Exception:
        pass
    return value
