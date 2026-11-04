"""Registro de algoritmos: metadatos, espacios de búsqueda y fábrica de modelos."""
from __future__ import annotations

from typing import Any, Dict, List

from app.ml.models.base import BaseForecaster
from app.ml.models.torch_forecasters import (
    CNNLSTMForecaster,
    ConvLSTMAttentionForecaster,
    ConvLSTMForecaster,
    LSTMForecaster,
)
from app.ml.models.xgboost_forecaster import XGBoostForecaster
from app.ml.reproducibility import torch_available

ALGORITHMS: List[Dict[str, Any]] = [
    {
        "key": "xgboost",
        "name": "XGBoost",
        "family": "gradient_boosting",
        "role": "Modelo individual 1",
        "description": "Predicción tabular de incidencia a partir de rezagos, clima y variables territoriales.",
        "spatial": False,
        "temporal": True,
        "attention": False,
        "backend": "xgboost",
        "default_horizons": [1, 2, 4, 8],
    },
    {
        "key": "lstm",
        "name": "LSTM",
        "family": "recurrent",
        "role": "Modelo individual 2",
        "description": "Captura dependencias temporales sobre la secuencia de 12 semanas por distrito.",
        "spatial": False,
        "temporal": True,
        "attention": False,
        "backend": "torch",
        "default_horizons": [1, 2, 4],
    },
    {
        "key": "convlstm",
        "name": "ConvLSTM",
        "family": "convolutional_recurrent",
        "role": "Modelo individual 3",
        "description": "Captura dependencias espacio-temporales sobre una rejilla de distritos y semanas.",
        "spatial": True,
        "temporal": True,
        "attention": False,
        "backend": "torch",
        "default_horizons": [1, 2, 4],
    },
    {
        "key": "cnn_lstm",
        "name": "CNN-LSTM",
        "family": "hybrid",
        "role": "Modelo híbrido 1",
        "description": "Extracción espacial con convoluciones por paso temporal y modelado secuencial con LSTM.",
        "spatial": True,
        "temporal": True,
        "attention": False,
        "backend": "torch",
        "default_horizons": [1, 2, 4],
    },
    {
        "key": "convlstm_attention",
        "name": "ConvLSTM-Attention",
        "family": "hybrid_attention",
        "role": "Modelo híbrido 2",
        "description": "ConvLSTM con atención temporal y espacial para identificar distritos y semanas relevantes.",
        "spatial": True,
        "temporal": True,
        "attention": True,
        "backend": "torch",
        "default_horizons": [1, 2, 4],
    },
]

_FACTORY = {
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
    "convlstm": ConvLSTMForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "convlstm_attention": ConvLSTMAttentionForecaster,
}

SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "xgboost": {
        "n_estimators": {"type": "int", "low": 200, "high": 900, "step": 50},
        "max_depth": {"type": "int", "low": 3, "high": 9},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "float", "low": 0.5, "high": 8.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-4, "high": 5.0, "log": True},
        "gamma": {"type": "float", "low": 1e-4, "high": 2.0, "log": True},
    },
    "lstm": {
        "hidden_size": {"type": "categorical", "choices": [32, 64, 96, 128]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "lr": {"type": "float", "low": 3e-4, "high": 5e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "convlstm": {
        "hidden_channels": {"type": "categorical", "choices": [8, 16, 24]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "kernel_size": {"type": "categorical", "choices": [3]},
        "lr": {"type": "float", "low": 3e-4, "high": 5e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    },
    "cnn_lstm": {
        "cnn_filters": {"type": "categorical", "choices": [8, 16, 32]},
        "hidden_size": {"type": "categorical", "choices": [32, 64, 96]},
        "num_layers": {"type": "int", "low": 1, "high": 2},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "lr": {"type": "float", "low": 3e-4, "high": 5e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    },
    "convlstm_attention": {
        "hidden_channels": {"type": "categorical", "choices": [8, 16, 24]},
        "num_layers": {"type": "int", "low": 1, "high": 2},
        "attention_heads": {"type": "categorical", "choices": [1, 2, 4]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.3},
        "lr": {"type": "float", "low": 3e-4, "high": 5e-3, "log": True},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
    },
}


def algorithm_metadata() -> List[Dict[str, Any]]:
    torch_ok = torch_available()
    items = []
    for item in ALGORITHMS:
        entry = dict(item)
        if entry["backend"] == "torch" and not torch_ok:
            entry["status"] = "UNAVAILABLE"
            entry["status_reason"] = "PyTorch no está instalado en este entorno."
        else:
            entry["status"] = "READY"
            entry["status_reason"] = ""
        items.append(entry)
    return items


def get_algorithm(key: str) -> Dict[str, Any]:
    for item in ALGORITHMS:
        if item["key"] == key:
            return item
    raise KeyError(f"Algoritmo desconocido: {key}")


def get_search_space(key: str) -> Dict[str, Any]:
    return SEARCH_SPACES.get(key, {})


def create_model(key: str) -> BaseForecaster:
    if key not in _FACTORY:
        raise KeyError(f"Algoritmo desconocido: {key}")
    if key != "xgboost" and not torch_available():
        raise RuntimeError("PyTorch no está disponible: no es posible entrenar modelos neuronales.")
    return _FACTORY[key]()
