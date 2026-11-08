"""Modelo individual 1: XGBoost para predicción tabular de incidencia."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from app.ml.models.base import BaseForecaster, ModelData


class XGBoostForecaster(BaseForecaster):
    key = "xgboost"
    name = "XGBoost"
    family = "gradient_boosting"
    requires = "tabular"

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.06,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "objective": "count:poisson",
        "random_state": 42,
        "n_jobs": 4,
    }

    def __init__(self) -> None:
        self.model = None
        self.params: Dict[str, Any] = {}
        self._importance: Optional[np.ndarray] = None

    def fit(
        self,
        train: ModelData,
        val: Optional[ModelData] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        trial: Any = None,
        epochs: int = 60,
    ) -> "XGBoostForecaster":
        import xgboost as xgb

        gpu_defaults: Dict[str, Any] = {"tree_method": "hist"}
        try:
            import torch

            if torch.cuda.is_available():
                gpu_defaults["device"] = "cuda"
        except Exception:
            pass

        self.params = {**self.DEFAULT_PARAMS, **gpu_defaults, **(params or {})}
        eval_set = [(train.X, train.y)]
        if val is not None and len(val) > 0:
            eval_set.append((val.X, val.y))

        callbacks = []
        if trial is not None and val is not None and len(val) > 0:
            from optuna import TrialPruned

            class _OptunaPruningCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
                    val_metrics = evals_log.get("validation_1")
                    if not val_metrics:
                        return False
                    metric_values = next(iter(val_metrics.values()))
                    if not metric_values:
                        return False
                    score = metric_values[-1]
                    if isinstance(score, tuple):
                        score = score[0]
                    # Escalar step cada 15 árboles a la escala 1..25 para compatibilidad con n_warmup_steps=5
                    if (epoch + 1) % 15 == 0:
                        step = (epoch + 1) // 15
                        trial.report(float(score), step=step)
                        if trial.should_prune():
                            raise TrialPruned()
                    return False

            callbacks.append(_OptunaPruningCallback())

        self.model = xgb.XGBRegressor(
            **self.params,
            early_stopping_rounds=40 if len(eval_set) > 1 else None,
            eval_metric="mae",
            callbacks=callbacks if callbacks else None,
        )
        self.model.fit(
            train.X,
            np.clip(train.y, 0, None),
            eval_set=eval_set if len(eval_set) > 1 else None,
            verbose=False,
        )
        booster = self.model.get_booster()
        gains = booster.get_score(importance_type="gain")
        names = list(train.feature_names)
        self._importance = np.array([float(gains.get(n, 0.0)) for n in names], dtype=float)
        return self

    def predict(self, data: ModelData) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo no está entrenado")
        pred = self.model.predict(data.X)
        return np.clip(pred, 0, None)

    def feature_importance(self) -> Optional[np.ndarray]:
        return self._importance
