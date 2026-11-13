"""Construcción del panel distrito-semana, variables predictoras y particiones sin fuga.

Reglas de diseño (M06 del prompt maestro):
* La unidad de análisis es distrito-semana epidemiológica.
* Para pronosticar la semana ``t+h`` solo se usan observaciones con fecha ``<= t``.
* La normalización se ajusta únicamente con el conjunto de entrenamiento.
* La validación es temporal (walk-forward) y geográfica (distritos no vistos).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.epi import ClimateObservation, DengueCase, EnvironmentalVariable
from app.models.geo import GeographicUnit

LAG_WEEKS = 12
GRID_SHAPE = (4, 5)  # (filas, columnas) del tensor espacial para ConvLSTM


@dataclass
class PanelResult:
    panel: pd.DataFrame
    districts: List[str]
    district_names: Dict[str, str]
    weeks: List[date]
    is_synthetic: bool
    source_summary: Dict[str, object] = field(default_factory=dict)
    data_mode: str = "synthetic"


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    meta_train: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    horizon: int
    is_synthetic: bool
    notes: Dict[str, object] = field(default_factory=dict)
    seq_train: Optional[np.ndarray] = None
    seq_val: Optional[np.ndarray] = None
    seq_test: Optional[np.ndarray] = None
    grid_train: Optional[np.ndarray] = None
    grid_val: Optional[np.ndarray] = None
    grid_test: Optional[np.ndarray] = None
    cell_train: Optional[np.ndarray] = None
    cell_val: Optional[np.ndarray] = None
    cell_test: Optional[np.ndarray] = None
    grid_shape: Tuple[int, int] = GRID_SHAPE

    def summary(self) -> Dict[str, object]:
        return {
            "n_train": int(len(self.y_train)),
            "n_val": int(len(self.y_val)),
            "n_test": int(len(self.y_test)),
            "n_features": len(self.feature_names),
            "horizon": self.horizon,
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
        }


def _has_real_cases(db: Session) -> bool:
    return db.scalar(select(DengueCase.id).where(DengueCase.is_synthetic.is_(False)).limit(1)) is not None


def _has_real_climate(db: Session) -> bool:
    return (
        db.scalar(select(ClimateObservation.id).where(ClimateObservation.is_synthetic.is_(False)).limit(1))
        is not None
    )


def resolve_data_mode(db: Session, *, only_synthetic: Optional[bool] = None, mode: Optional[str] = None) -> str:
    """Resuelve el modo efectivo: ``real`` o ``synthetic``.

    ``auto`` (valor por defecto) usa datos reales solo si existen **casos y clima**
    reales; de lo contrario recurre al conjunto sintético. Los modos explícitos
    nunca mezclan ambos orígenes y fallan si falta la cobertura mínima.
    """
    if only_synthetic is not None:
        return "synthetic" if only_synthetic else "real"
    requested = (mode or settings.DATA_MODE or "auto").lower()
    if requested == "synthetic":
        return "synthetic"
    if requested == "real":
        if not _has_real_cases(db):
            raise ValueError(
                "DATA_MODE=real pero no hay casos reales en la base de datos. "
                "Ejecute la ingesta real (MINSA + NASA POWER) o use DATA_MODE=auto."
            )
        if not _has_real_climate(db):
            raise ValueError(
                "DATA_MODE=real pero no hay clima real (NASA POWER). "
                "Ejecute la ingesta climática o use DATA_MODE=auto."
            )
        return "real"
    if requested != "auto":
        raise ValueError(f"DATA_MODE inválido: {requested!r}. Use auto, synthetic o real.")
    return "real" if (_has_real_cases(db) and _has_real_climate(db)) else "synthetic"


def load_panel(
    db: Session,
    *,
    only_synthetic: Optional[bool] = None,
    mode: Optional[str] = None,
) -> PanelResult:
    """Carga casos + clima en formato largo (una fila por distrito-semana).

    El origen se elige de forma homogénea: nunca se mezclan filas reales y
    sintéticas en el mismo panel, para no contaminar métricas ni conclusiones.
    """
    data_mode = resolve_data_mode(db, only_synthetic=only_synthetic, mode=mode)
    use_synthetic = data_mode == "synthetic"
    case_stmt = select(DengueCase).where(DengueCase.is_synthetic == use_synthetic)
    cases = pd.DataFrame(
        [
            {
                "district_code": row.district_code,
                "week_start": row.week_start,
                "epi_year": row.epi_year,
                "epi_week": row.epi_week,
                "confirmed_cases": row.confirmed_cases,
                "hospitalized": row.hospitalized,
                "is_synthetic": row.is_synthetic,
            }
            for row in db.scalars(case_stmt)
        ]
    )
    if cases.empty:
        raise ValueError(
            f"No hay datos de casos {'reales' if data_mode == 'real' else 'sintéticos'} cargados. "
            "Importe un archivo o ejecute el poblado/ingesta correspondiente."
        )

    climate = pd.DataFrame(
        [
            {
                "district_code": row.district_code,
                "week_start": row.week_start,
                "temperature_mean": row.temperature_mean,
                "temperature_max": row.temperature_max,
                "temperature_min": row.temperature_min,
                "humidity_mean": row.humidity_mean,
                "precipitation_mm": row.precipitation_mm,
                "wind_speed": row.wind_speed,
                "solar_radiation": row.solar_radiation,
                "is_synthetic": row.is_synthetic,
            }
            for row in db.scalars(
                select(ClimateObservation).where(ClimateObservation.is_synthetic == use_synthetic)
            )
        ]
    )

    units = list(db.scalars(select(GeographicUnit)))
    district_names = {u.code: u.name for u in units}
    uframe = pd.DataFrame(
        [
            {
                "district_code": u.code,
                "altitude_m": u.altitude_m,
                "population": u.population,
                "area_km2": u.area_km2,
                "latitude": u.centroid_lat,
                "longitude": u.centroid_lon,
                "province": u.province,
            }
            for u in units
        ]
    )

    panel = cases.merge(climate, on=["district_code", "week_start"], how="left", suffixes=("", "_climate"))
    panel = panel.merge(uframe, on="district_code", how="left")
    panel = panel.sort_values(["district_code", "week_start"]).reset_index(drop=True)

    env = pd.DataFrame(
        [
            {
                "district_code": row.district_code,
                "epi_year": row.epi_year,
                "epi_week": row.epi_week,
                "name": row.name,
                "value": row.value,
            }
            for row in db.scalars(
                select(EnvironmentalVariable).where(
                    EnvironmentalVariable.name == "population_density",
                    EnvironmentalVariable.is_synthetic == use_synthetic,
                )
            )
        ]
    )
    if not env.empty:
        env = env.rename(columns={"value": "population_density"}).drop(columns=["name"])
        env = env.groupby(["district_code", "epi_year", "epi_week"], as_index=False)["population_density"].mean()
        panel = panel.merge(env, on=["district_code", "epi_year", "epi_week"], how="left")
    else:
        panel["population_density"] = panel["population"] / panel["area_km2"].replace(0, np.nan)

    panel["population_density"] = panel.groupby("district_code")["population_density"].ffill()
    panel["population_density"] = panel["population_density"].fillna(
        panel["population"] / panel["area_km2"].replace(0, np.nan)
    )

    # Imputación temporal estrictamente causal (solo forward-fill por distrito, sin usar futuro).
    # Cualquier NaN inicial previo a la primera observación se imputa después con la media de entrenamiento.
    numeric_cols = [
        "temperature_mean", "temperature_max", "temperature_min", "humidity_mean",
        "precipitation_mm", "wind_speed", "solar_radiation",
    ]
    panel[numeric_cols] = panel.groupby("district_code")[numeric_cols].ffill()

    if panel["is_synthetic"].fillna(False).astype(bool).nunique() > 1:
        raise ValueError("El panel contiene una mezcla de filas reales y sintéticas; revise la ingesta.")
    is_synthetic = use_synthetic
    districts = sorted(panel["district_code"].unique().tolist())
    weeks = sorted(panel["week_start"].unique().tolist())
    source_summary = {
        "data_mode": data_mode,
        "case_rows": int(len(cases)),
        "climate_rows": int(len(climate)),
        "synthetic_rows": int(panel["is_synthetic"].fillna(False).astype(bool).sum()),
        "real_rows": int((~panel["is_synthetic"].fillna(False).astype(bool)).sum()),
        "missing_climate_pct": round(float(panel[numeric_cols].isna().any(axis=1).mean() * 100), 3),
        "districts": len(districts),
        "weeks": len(weeks),
        "start_week": str(pd.Timestamp(weeks[0]).date()) if weeks else None,
        "end_week": str(pd.Timestamp(weeks[-1]).date()) if weeks else None,
    }
    return PanelResult(
        panel=panel,
        districts=districts,
        district_names=district_names,
        weeks=weeks,
        is_synthetic=is_synthetic,
        source_summary=source_summary,
        data_mode=data_mode,
    )


def _spatial_neighbours(panel: pd.DataFrame) -> pd.DataFrame:
    """Media de casos de los demás distritos por semana (proxy de propagación espacial)."""
    weekly = panel.pivot_table(index="week_start", columns="district_code", values="confirmed_cases", aggfunc="sum")
    total = weekly.sum(axis=1)
    count = weekly.notna().sum(axis=1).replace(0, np.nan)
    neighbour_mean = (total - weekly).div(count - 1)
    long = neighbour_mean.stack().rename("neighbour_cases").reset_index()
    long.columns = ["week_start", "district_code", "neighbour_cases"]
    return long


def build_features(panel_result: PanelResult, *, horizon: int, max_lag: int = LAG_WEEKS) -> pd.DataFrame:
    """Genera el marco supervisado: una fila por (distrito, semana de origen)."""
    if int(horizon) < 1:
        raise ValueError(f"El horizonte debe ser >= 1 para garantizar causalidad temporal; recibido {horizon}.")
    panel = panel_result.panel.copy()
    panel = panel.sort_values(["district_code", "week_start"])

    grouped = panel.groupby("district_code", group_keys=False)

    for lag in range(1, max_lag + 1):
        panel[f"cases_lag_{lag}"] = grouped["confirmed_cases"].shift(lag)

    for window in (4, 8, 12):
        panel[f"cases_mean_{window}"] = grouped["confirmed_cases"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 3)).mean()
        )
    panel["cases_std_4"] = grouped["confirmed_cases"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    panel["cases_diff_1"] = panel["cases_lag_1"] - panel["cases_lag_2"]
    panel["cases_growth_4"] = (panel["cases_lag_1"] + 1) / (panel["cases_mean_4"] + 1)

    for col in ("temperature_mean", "humidity_mean", "precipitation_mm"):
        for lag in range(0, 5):
            panel[f"{col}_lag_{lag}"] = grouped[col].shift(lag if lag else 0)
    for window in (4, 12):
        panel[f"precipitation_mean_{window}"] = grouped["precipitation_mm"].transform(
            lambda s: s.rolling(window, min_periods=2).mean()
        )
        panel[f"temperature_mean_{window}"] = grouped["temperature_mean"].transform(
            lambda s: s.rolling(window, min_periods=2).mean()
        )

    panel = panel.merge(_spatial_neighbours(panel_result.panel), on=["week_start", "district_code"], how="left")
    panel["neighbour_cases_lag_1"] = panel.groupby("district_code")["neighbour_cases"].shift(1)

    week_of_year = pd.to_datetime(panel["week_start"]).dt.isocalendar().week.astype(float)
    panel["sin_week"] = np.sin(2 * np.pi * week_of_year / 52.0)
    panel["cos_week"] = np.cos(2 * np.pi * week_of_year / 52.0)
    panel["sin_week_2"] = np.sin(4 * np.pi * week_of_year / 52.0)
    panel["cos_week_2"] = np.cos(4 * np.pi * week_of_year / 52.0)
    panel["altitude_km"] = panel["altitude_m"] / 1000.0
    panel["log_population"] = np.log1p(panel["population"])
    panel["log_density"] = np.log1p(panel["population_density"])

    weeks_sorted = sorted(panel["week_start"].unique())
    week_index = {w: i for i, w in enumerate(weeks_sorted)}
    panel["time_index"] = panel["week_start"].map(week_index).astype(float)

    district_order = {code: i for i, code in enumerate(sorted(panel["district_code"].unique()))}
    panel["district_idx"] = panel["district_code"].map(district_order).astype(int)
    panel["week_idx"] = panel["week_start"].map(week_index).astype(int)

    # Objetivo desplazado: casos en t+h (nunca se usa como predictor).
    panel["target"] = panel.groupby("district_code")["confirmed_cases"].shift(-horizon)
    panel["target_week"] = panel.groupby("district_code")["week_start"].shift(-horizon)
    panel["origin_week"] = panel["week_start"]
    panel["horizon"] = horizon
    return panel


FEATURE_EXCLUDE = {
    "district_code", "week_start", "epi_year", "epi_week", "confirmed_cases",
    "hospitalized", "is_synthetic", "is_synthetic_climate", "target", "target_week",
    "origin_week", "horizon", "province", "neighbour_cases", "district_idx", "week_idx",
}


def feature_columns(frame: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in frame.columns
        if c not in FEATURE_EXCLUDE and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return sorted(cols)


def split_temporal(
    frame: pd.DataFrame,
    *,
    validation_weeks: int,
    test_weeks: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Partición estrictamente temporal: pasado -> validación -> prueba."""
    frame = frame.dropna(subset=["target"]).copy()
    origin_weeks = sorted(frame["origin_week"].unique())
    if len(origin_weeks) < (validation_weeks + test_weeks + 20):
        raise ValueError(
            f"Se requieren al menos {validation_weeks + test_weeks + 20} semanas con datos; hay {len(origin_weeks)}."
        )
    test_start = origin_weeks[-test_weeks]
    val_start = origin_weeks[-(test_weeks + validation_weeks)]
    # Sin fuga temporal: el objetivo de una fila de entrenamiento no puede caer en
    # el periodo de validación, ni el de validación en el periodo de prueba.
    train = frame[(frame["origin_week"] < val_start) & (frame["target_week"] < val_start)]
    val = frame[
        (frame["origin_week"] >= val_start)
        & (frame["origin_week"] < test_start)
        & (frame["target_week"] < test_start)
    ]
    test = frame[frame["origin_week"] >= test_start]
    notes = {
        "train_end": str(pd.Timestamp(val_start).date()),
        "train_target_end": str(pd.Timestamp(train["target_week"].max()).date()) if len(train) else None,
        "val_range": [str(pd.Timestamp(val_start).date()), str(pd.Timestamp(test_start).date())],
        "val_target_end": str(pd.Timestamp(val["target_week"].max()).date()) if len(val) else None,
        "test_start": str(pd.Timestamp(test_start).date()),
        "strategy": "temporal_walk_forward",
    }
    return train, val, test, notes


def walk_forward_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 5,
    validation_weeks: int = 13,
    test_weeks: int = 26,
    horizon: int = 1,
) -> List[Dict[str, object]]:
    """Folds walk-forward con ventana expansiva y un bloque de prueba final intocable.

    Cada fold cumple ``max(target_week del train) < min(origin_week de validación)``:
    no hay solapamiento de objetivos entre entrenamiento y validación. El bloque de
    prueba final nunca se usa para seleccionar hiperparámetros ni para construir folds.
    """
    frame = frame.dropna(subset=["target"]).copy()
    weeks = sorted(frame["origin_week"].unique())
    required = test_weeks + n_folds * validation_weeks + 20
    if len(weeks) < required:
        raise ValueError(
            f"Se requieren al menos {required} semanas de origen para {n_folds} folds walk-forward; "
            f"hay {len(weeks)}."
        )
    test_start = weeks[-test_weeks]
    history = [week for week in weeks if week < test_start]
    fold_span = n_folds * validation_weeks
    if len(history) <= fold_span + horizon:
        raise ValueError("No hay suficiente historia previa al bloque de prueba para construir folds.")

    folds: List[Dict[str, object]] = []
    first_val_idx = len(history) - fold_span
    for fold_idx in range(n_folds):
        val_start_idx = first_val_idx + fold_idx * validation_weeks
        val_end_idx = val_start_idx + validation_weeks
        val_start = history[val_start_idx]
        val_end = history[val_end_idx - 1]
        train_weeks = history[:val_start_idx]
        val_weeks = history[val_start_idx:val_end_idx]
        train = frame[
            frame["origin_week"].isin(train_weeks) & (frame["target_week"] < val_start)
        ]
        val = frame[
            frame["origin_week"].isin(val_weeks) & (frame["target_week"] < test_start)
        ]
        if train.empty or val.empty:
            continue
        folds.append(
            {
                "fold": fold_idx + 1,
                "train": train,
                "validation": val,
                "train_weeks": len(train_weeks),
                "validation_weeks": len(val_weeks),
                "train_origin_range": [
                    str(pd.Timestamp(min(train_weeks)).date()),
                    str(pd.Timestamp(max(train_weeks)).date()),
                ],
                "validation_origin_range": [
                    str(pd.Timestamp(val_start).date()),
                    str(pd.Timestamp(val_end).date()),
                ],
                "train_target_end": str(pd.Timestamp(train["target_week"].max()).date()),
                "validation_target_start": str(pd.Timestamp(val["target_week"].min()).date()),
                "test_start": str(pd.Timestamp(test_start).date()),
                "horizon": horizon,
            }
        )
    if not folds:
        raise ValueError("No se pudo construir ningún fold walk-forward válido.")
    return folds


def split_geographic(frame: pd.DataFrame, *, holdout_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Validación geográfica: distritos completos quedan fuera del entrenamiento."""
    frame = frame.dropna(subset=["target"]).copy()
    districts = sorted(frame["district_code"].unique())
    if len(districts) < 6:
        raise ValueError("Se requieren al menos 6 distritos para validación geográfica.")
    n_holdout = max(2, int(round(len(districts) * holdout_ratio)))
    # Selección determinista: distritos equiespaciados alfabéticamente (reproducible, sin mirar el objetivo).
    holdout = set(districts[:: max(1, len(districts) // n_holdout)][:n_holdout])
    test = frame[frame["district_code"].isin(holdout)]
    remaining = frame[~frame["district_code"].isin(holdout)]
    val_districts = set(sorted(remaining["district_code"].unique())[-max(1, n_holdout - 1):])
    val = remaining[remaining["district_code"].isin(val_districts)]
    train = remaining[~remaining["district_code"].isin(val_districts)]
    notes = {
        "holdout_districts": sorted(holdout),
        "val_districts": sorted(val_districts),
        "strategy": "geografica",
    }
    return train, val, test, notes


def fit_scaler(train: pd.DataFrame, columns: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    matrix = train[list(columns)].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return mean, std


def to_matrix(frame: pd.DataFrame, columns: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    matrix = frame[list(columns)].to_numpy(dtype=float)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    return (matrix - mean) / std


def build_dataset(
    panel_result: PanelResult,
    *,
    horizon: int = 1,
    validation_weeks: int = 26,
    test_weeks: int = 26,
    strategy: str = "walk_forward",
    max_lag: int = LAG_WEEKS,
) -> DatasetBundle:
    frame = build_features(panel_result, horizon=horizon, max_lag=max_lag)
    if strategy == "geographic":
        train_raw, val_raw, test_raw, notes = split_geographic(frame)
    else:
        train_raw, val_raw, test_raw, notes = split_temporal(
            frame, validation_weeks=validation_weeks, test_weeks=test_weeks
        )
    cols = feature_columns(frame)
    mean, std = fit_scaler(train_raw, cols)

    meta_cols = ["district_code", "origin_week", "target_week", "target", "epi_year", "epi_week"]
    bundle = DatasetBundle(
        X_train=to_matrix(train_raw, cols, mean, std),
        y_train=train_raw["target"].to_numpy(dtype=float),
        X_val=to_matrix(val_raw, cols, mean, std),
        y_val=val_raw["target"].to_numpy(dtype=float),
        X_test=to_matrix(test_raw, cols, mean, std),
        y_test=test_raw["target"].to_numpy(dtype=float),
        feature_names=cols,
        meta_train=train_raw[meta_cols].reset_index(drop=True),
        meta_val=val_raw[meta_cols].reset_index(drop=True),
        meta_test=test_raw[meta_cols].reset_index(drop=True),
        scaler_mean=mean,
        scaler_std=std,
        horizon=horizon,
        is_synthetic=panel_result.is_synthetic,
        notes={**notes, "validation_weeks": validation_weeks, "test_weeks": test_weeks},
    )
    _attach_sequences(bundle, frame, panel_result, cols, mean, std, train_raw, val_raw, test_raw)
    return bundle


def _attach_sequences(
    bundle: DatasetBundle,
    frame: pd.DataFrame,
    panel_result: PanelResult,
    columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    window: int = 12,
) -> None:
    """Añade ventanas (T, F) y tensores espaciales alineados con las filas supervisadas."""
    context = _window_context(frame, panel_result, columns, mean, std, window)
    bundle.seq_train, bundle.grid_train, bundle.cell_train = _windows_for(train_raw, context)
    bundle.seq_val, bundle.grid_val, bundle.cell_val = _windows_for(val_raw, context)
    bundle.seq_test, bundle.grid_test, bundle.cell_test = _windows_for(test_raw, context)
    bundle.notes["sequence_window"] = window
    bundle.notes["grid_shape"] = list(context["grid_shape"])
    bundle.notes["district_grid"] = {code: list(pos) for code, pos in context["mapping"].items()}


CHANNEL_NAMES = ("confirmed_cases", "temperature_mean", "humidity_mean", "precipitation_mm")


def _window_context(
    frame: pd.DataFrame,
    panel_result: PanelResult,
    columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    window: int = 12,
) -> Dict[str, Any]:
    """Precalcula los cubos de características y del tensor espacial una sola vez."""
    n_d = int(frame["district_idx"].max()) + 1
    n_w = int(frame["week_idx"].max()) + 1
    cube_features = np.zeros((n_d, n_w, len(columns)), dtype=np.float32)
    matrix = to_matrix(frame, columns, mean, std)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    cube_features[frame["district_idx"].to_numpy(), frame["week_idx"].to_numpy()] = matrix.astype(np.float32)

    rows, cols = GRID_SHAPE
    cube_grid = np.zeros((n_w, rows, cols, len(CHANNEL_NAMES) + 1), dtype=np.float32)
    mapping = district_grid(panel_result.districts, _centroids(panel_result))
    week_lookup = {week: idx for idx, week in enumerate(panel_result.weeks)}
    col_idx = {col: idx for idx, col in enumerate(columns)}
    stat_keys = ("cases_lag_1", "temperature_mean", "humidity_mean", "precipitation_mm")
    ch_mean = np.array(
        [float(mean[col_idx[k]]) if k in col_idx else 0.0 for k in stat_keys],
        dtype=np.float32,
    )
    ch_std = np.array(
        [float(std[col_idx[k]]) if k in col_idx and std[col_idx[k]] > 1e-9 else 1.0 for k in stat_keys],
        dtype=np.float32,
    )
    for _, record in panel_result.panel.iterrows():
        position = mapping.get(record["district_code"])
        week_position = week_lookup.get(record["week_start"])
        if position is None or week_position is None:
            continue
        r, c = position
        raw_cases = float(record["confirmed_cases"]) if pd.notna(record["confirmed_cases"]) else ch_mean[0]
        cube_grid[week_position, r, c, 0] = (raw_cases - ch_mean[0]) / ch_std[0]
        for ch_idx, name in enumerate(CHANNEL_NAMES[1:], start=1):
            value = record.get(name)
            raw_val = float(value) if pd.notna(value) else float(ch_mean[ch_idx])
            cube_grid[week_position, r, c, ch_idx] = (raw_val - ch_mean[ch_idx]) / ch_std[ch_idx]
        cube_grid[week_position, r, c, -1] = 1.0

    return {
        "cube_features": cube_features,
        "cube_grid": cube_grid,
        "mapping": mapping,
        "grid_shape": (rows, cols),
        "window": window,
    }


def _windows_for(subset: pd.DataFrame, context: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cube_features = context["cube_features"]
    cube_grid = context["cube_grid"]
    window = int(context["window"])
    rows, cols = context["grid_shape"]
    mapping = context["mapping"]
    n_f = cube_features.shape[2]
    channels = cube_grid.shape[3]

    seq = np.zeros((len(subset), window, n_f), dtype=np.float32)
    grid = np.zeros((len(subset), window, rows, cols, channels), dtype=np.float32)
    cells = np.zeros((len(subset),), dtype=np.int64)
    for i, record in enumerate(subset.itertuples(index=False)):
        d = int(record.district_idx)
        w = int(record.week_idx)
        start = max(0, w - window + 1)
        chunk = cube_features[d, start : w + 1]
        if len(chunk) < window:
            chunk = np.concatenate([np.repeat(chunk[:1], window - len(chunk), axis=0), chunk], axis=0)
        seq[i] = chunk
        spatial = cube_grid[start : w + 1]
        if len(spatial) < window:
            spatial = np.concatenate([np.repeat(spatial[:1], window - len(spatial), axis=0), spatial], axis=0)
        grid[i] = spatial
        r, c = mapping.get(record.district_code, (0, 0))
        cells[i] = r * cols + c
    return seq, grid, cells


def make_sample_tensors(
    panel_result: PanelResult,
    frame: pd.DataFrame,
    subset: pd.DataFrame,
    columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    window: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Versión pública usada por la inferencia (mismo tratamiento que en entrenamiento)."""
    context = _window_context(frame, panel_result, columns, mean, std, window)
    seq, grid, cell = _windows_for(subset, context)
    return seq, grid, cell, context


def _centroids(panel_result: PanelResult) -> Dict[str, Tuple[float, float]]:
    panel = panel_result.panel
    result: Dict[str, Tuple[float, float]] = {}
    for code, group in panel.groupby("district_code"):
        result[code] = (float(group["latitude"].iloc[0] or 0.0), float(group["longitude"].iloc[0] or 0.0))
    return result


# --------------------------------------------------------------------------------------
# Tensores espacio-temporales para ConvLSTM / CNN-LSTM
# --------------------------------------------------------------------------------------
def district_grid(district_codes: Sequence[str], centroids: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[int, int]]:
    """Asigna cada distrito a una celda única de la rejilla según su centroide (sin colisiones)."""
    rows, cols = GRID_SHAPE
    known = [code for code in district_codes if code in centroids]
    if known:
        lats = np.array([float(centroids[code][0]) for code in known])
        lons = np.array([float(centroids[code][1]) for code in known])
    else:
        lats = np.array([0.0])
        lons = np.array([0.0])
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    lon_min, lon_max = float(np.min(lons)), float(np.max(lons))
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    used: set[Tuple[int, int]] = set()
    mapping: Dict[str, Tuple[int, int]] = {}
    for code in district_codes:
        if code not in centroids:
            r_target, c_target = 0.0, 0.0
        else:
            lat, lon = centroids[code]
            r_target = 0.0 if lat_max - lat_min < 1e-9 else (lat_max - lat) / (lat_max - lat_min) * (rows - 1)
            c_target = 0.0 if lon_max - lon_min < 1e-9 else (lon - lon_min) / (lon_max - lon_min) * (cols - 1)
        available = [cell for cell in all_cells if cell not in used] or all_cells
        chosen = min(available, key=lambda rc: ((rc[0] - r_target) ** 2 + (rc[1] - c_target) ** 2, rc[0], rc[1]))
        used.add(chosen)
        mapping[code] = chosen
    return mapping


def build_tensor_arrays(
    panel_result: PanelResult,
    *,
    horizons: Sequence[int] = (1, 2, 4, 8),
    window: int = 12,
    channels: Sequence[str] = ("confirmed_cases", "temperature_mean", "humidity_mean", "precipitation_mm"),
) -> Dict[str, object]:
    """Construye tensores (N, T, H, W, C) y vectores objetivo multihorizonte."""
    panel = panel_result.panel.sort_values(["district_code", "week_start"])
    weeks = sorted(panel["week_start"].unique())
    districts = panel_result.districts
    centroids = {
        row["district_code"]: (float(row["latitude"] or 0), float(row["longitude"] or 0))
        for _, row in panel.drop_duplicates("district_code").iterrows()
    }
    mapping = district_grid(districts, centroids)
    rows, cols = GRID_SHAPE

    def _grid_for(week_value) -> np.ndarray:
        cube = np.zeros((rows, cols, len(channels)), dtype=np.float32)
        mask = np.zeros((rows, cols), dtype=np.float32)
        subset = panel[panel["week_start"] == week_value]
        for _, record in subset.iterrows():
            r, c = mapping[record["district_code"]]
            cube[r, c, 0] = float(record["confirmed_cases"])
            for idx, ch in enumerate(channels[1:], start=1):
                value = record.get(ch)
                cube[r, c, idx] = float(value) if pd.notna(value) else 0.0
            mask[r, c] = 1.0
        return np.concatenate([cube, mask[..., None]], axis=2)

    samples = []
    targets = {h: [] for h in horizons}
    meta = []
    for idx in range(window, len(weeks)):
        origin = weeks[idx]
        seq = np.stack([_grid_for(weeks[j]) for j in range(idx - window, idx)], axis=0)
        entry = {"origin_week": origin, "district_order": districts}
        ok = True
        per_h = {}
        for h in horizons:
            if idx + h >= len(weeks):
                ok = False
                break
            target_week = weeks[idx + h]
            subset = panel[(panel["week_start"] == target_week) & (panel["district_code"].isin(districts))]
            values = subset.set_index("district_code")["confirmed_cases"].reindex(districts)
            if values.isna().any():
                ok = False
                break
            per_h[h] = values.to_numpy(dtype=np.float32)
        if not ok:
            continue
        samples.append(seq)
        for h in horizons:
            targets[h].append(per_h[h])
        meta.append(entry)

    X = np.stack(samples, axis=0) if samples else np.zeros((0, window, rows, cols, len(channels) + 1), dtype=np.float32)
    y = {h: np.stack(v, axis=0) if v else np.zeros((0, len(districts)), dtype=np.float32) for h, v in targets.items()}
    return {"X": X, "y": y, "meta": meta, "districts": districts, "grid_shape": (rows, cols), "mapping": mapping, "channels": list(channels)}
