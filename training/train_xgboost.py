"""Script CLI independiente para entrenar y optimizar XGBoost (Fase 2)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.common import cli_entrypoint  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    return cli_entrypoint(
        "xgboost",
        "Entrenamiento y optimización Optuna de XGBoost (--epochs mapea a n_estimators si no se indica --n-estimators).",
        allow_n_estimators=True,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
