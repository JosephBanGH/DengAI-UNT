"""Contratos comunes de los modelos de pronóstico."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ModelData:
    """Contenedor alineado de entradas por muestra (distrito, semana de origen)."""

    X: np.ndarray                      # (N, F) variables tabulares estandarizadas
    seq: np.ndarray                    # (N, T, F) secuencia temporal de la misma unidad
    grid: Optional[np.ndarray]         # (N, T, H, W, C) tensor espacio-temporal
    cell: Optional[np.ndarray]         # (N,) índice plano de la celda del distrito
    y: np.ndarray                      # (N,) casos observados en t+h
    feature_names: Sequence[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(len(self.y))


class BaseForecaster:
    """Interfaz mínima: entrenar, predecir e informar importancia de variables."""

    key: str = "base"
    name: str = "Modelo base"
    requires: str = "tabular"  # tabular | sequence | grid
    family: str = "baseline"

    def fit(
        self,
        train: ModelData,
        val: Optional[ModelData] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        trial: Any = None,
        epochs: int = 60,
    ) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, data: ModelData) -> np.ndarray:
        raise NotImplementedError

    def feature_importance(self) -> Optional[np.ndarray]:
        return None

    def training_history(self) -> List[Dict[str, float]]:
        return []

    def description(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "family": self.family, "requires": self.requires}
