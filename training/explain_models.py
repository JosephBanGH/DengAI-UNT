"""Script CLI de explicabilidad XAI (Fase 2 - Sesión 3).

Genera explicaciones locales y temporales reales (sin fallbacks silenciosos por error):
- SHAP (TreeExplainer para XGBoost, KernelExplainer adaptado para PyTorch).
- LIME tabular local (LimeTabularExplainer con fondo estricto de entrenamiento).
- Importancia acumulada de rezagos (t-1 a t-12) para casos y clima.
- Pesos reales de atención temporal multi-cabezal y mapa espacial 4x5 mapeado a
  los distritos reales de La Libertad (ConvLSTM-Attention).
- Exportación en PNG (300 DPI), SVG vectorial y JSON en ``data/reports/xai/``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import site
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.common import PROJECT_DIR, configure_cli_logging, load_training_panel  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.ml.explain import (  # noqa: E402
    DISTRICT_TO_UBIGEO,
    explain_sample,
    resolve_district_code,
)
from app.ml.features import GRID_SHAPE, build_features, split_temporal  # noqa: E402
from app.ml.reproducibility import configure_threads, set_reproducible_seed  # noqa: E402
from app.ml.trainer import load_bundle, resolve_artifact_path  # noqa: E402

logger = logging.getLogger("dengai.training.explain")

DEFAULT_XAI_DIR = "data/reports/xai"
ALL_ALGORITHMS = ("xgboost", "lstm", "convlstm", "cnn_lstm", "convlstm_attention")
DEFAULT_AUTO_ALGORITHMS = ("xgboost", "convlstm_attention")


def _import_pyplot():
    """Importa matplotlib.pyplot con backend Agg y directorio cache escribible."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dengai_mpl"))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        saved = list(sys.path)
        try:
            for sp in site.getsitepackages():
                if sp not in sys.path:
                    sys.path.append(sp)
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            return plt
        finally:
            sys.path[:] = saved


def _resolve_model_paths(
    *,
    model_path: Optional[str],
    algorithms: Sequence[str],
    horizon: int,
) -> List[Tuple[str, Path]]:
    if model_path:
        resolved = resolve_artifact_path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"No se encontró el artefacto indicado en --model-path: {model_path}")
        return [("custom", resolved)]

    requested: List[str] = []
    for item in algorithms:
        norm = item.strip().lower()
        if norm == "auto":
            for alg in DEFAULT_AUTO_ALGORITHMS:
                if alg not in requested:
                    requested.append(alg)
        elif norm == "all":
            for alg in ALL_ALGORITHMS:
                if alg not in requested:
                    requested.append(alg)
        elif norm in ALL_ALGORITHMS:
            if norm not in requested:
                requested.append(norm)
        else:
            raise ValueError(f"Algoritmo inválido: {item!r}. Opciones: auto, all, {', '.join(ALL_ALGORITHMS)}.")

    resolved_list: List[Tuple[str, Path]] = []
    for alg in requested:
        candidates = [
            PROJECT_DIR / "data" / "models" / f"{alg}-h{horizon}-optuna.joblib",
        ]
        phase2_dir = PROJECT_DIR / "data" / "models" / "phase2"
        if phase2_dir.is_dir():
            candidates.extend(sorted(phase2_dir.glob(f"{alg}-h{horizon}-*.joblib"), reverse=True))
        root_dir = PROJECT_DIR / "data" / "models"
        if root_dir.is_dir():
            candidates.extend(sorted(root_dir.glob(f"{alg}-h{horizon}-*.joblib"), reverse=True))

        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"No se encontró un modelo entrenado para '{alg}' (h={horizon}) en data/models/. "
                f"Indique --model-path explícitamente."
            )
        resolved_list.append((alg, found))
    return resolved_list


def _select_week_row(frame: pd.DataFrame, district_code: str, week_spec: str) -> Tuple[pd.DataFrame, str]:
    sub = frame[frame["district_code"] == district_code].copy()
    if sub.empty:
        raise ValueError(f"El distrito {district_code!r} no tiene registros en el panel cargado.")

    spec = (week_spec or "latest").strip()
    if spec.lower() == "latest":
        target_date = sub["origin_week"].max()
        row = sub[sub["origin_week"] == target_date]
        y = int(row["epi_year"].iloc[0])
        w = int(row["epi_week"].iloc[0])
        return row.head(1), f"{y}{w:02d}"

    m_iso = re.fullmatch(r"(\d{4})-?W(\d{1,2})", spec, flags=re.IGNORECASE)
    if m_iso:
        year, epi_week = int(m_iso.group(1)), int(m_iso.group(2))
        row = sub[(sub["epi_year"] == year) & (sub["epi_week"] == epi_week)]
        if row.empty:
            raise ValueError(f"No existe la semana {year}-W{epi_week:02d} para {district_code}.")
        return row.head(1), f"{year}{epi_week:02d}"

    m_compact = re.fullmatch(r"(\d{4})(\d{2})", spec)
    if m_compact:
        year, epi_week = int(m_compact.group(1)), int(m_compact.group(2))
        row = sub[(sub["epi_year"] == year) & (sub["epi_week"] == epi_week)]
        if row.empty:
            raise ValueError(f"No existe la semana epidemiológica {spec} para {district_code}.")
        return row.head(1), f"{year}{epi_week:02d}"

    try:
        parsed_date = pd.Timestamp(spec).date()
    except Exception as exc:
        raise ValueError(
            f"Formato de --week inválido: {week_spec!r}. Use 'latest', YYYYWW (p. ej. 202415), YYYY-Www o YYYY-MM-DD."
        ) from exc

    row = sub[pd.to_datetime(sub["origin_week"]).dt.date == parsed_date]
    if row.empty:
        raise ValueError(f"No existe la fecha {parsed_date} en el panel para {district_code}.")
    y = int(row["epi_year"].iloc[0])
    w = int(row["epi_week"].iloc[0])
    return row.head(1), f"{y}{w:02d}"


def _save_figure(fig, base_path: Path) -> Dict[str, str]:
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return {"png": str(png_path), "svg": str(svg_path)}


def _plot_shap_lime(
    plt,
    result: Dict[str, Any],
    *,
    algorithm: str,
    district_label: str,
    week_label: str,
    out_base: Path,
) -> Dict[str, str]:
    shap_factors = list(result["factors"])[::-1]
    lime_data = result.get("lime") or {}
    lime_factors = list(lime_data.get("factors") or [])[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=150)

    # Panel izquierdo: SHAP
    ax0 = axes[0]
    if shap_factors:
        labels0 = [f["label"] for f in shap_factors]
        vals0 = [float(f["contribution"]) for f in shap_factors]
        colors0 = ["#c0392b" if v > 0 else "#2471a3" for v in vals0]
        ax0.barh(range(len(labels0)), vals0, color=colors0, alpha=0.88)
        ax0.set_yticks(range(len(labels0)))
        ax0.set_yticklabels(labels0, fontsize=8.5)
        ax0.axvline(0.0, color="#2c3e50", linewidth=0.9, linestyle="--")
    ax0.set_xlabel("Contribución SHAP (casos)")
    ax0.set_title(
        f"{result['method']} · Predicción={result['prediction']:.2f} (base={result['base_value']:.2f})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax0.grid(axis="x", alpha=0.25)

    # Panel derecho: LIME
    ax1 = axes[1]
    if lime_factors:
        labels1 = [f["label"] for f in lime_factors]
        vals1 = [float(f["contribution"]) for f in lime_factors]
        colors1 = ["#d35400" if v > 0 else "#1f618d" for v in vals1]
        ax1.barh(range(len(labels1)), vals1, color=colors1, alpha=0.88)
        ax1.set_yticks(range(len(labels1)))
        ax1.set_yticklabels(labels1, fontsize=8.5)
        ax1.axvline(0.0, color="#2c3e50", linewidth=0.9, linestyle="--")
    r2_txt = f"R²={lime_data.get('r2_score'):.3f}" if lime_data.get("r2_score") is not None else "R²=N/D"
    loc_pred = lime_data.get("local_prediction")
    loc_txt = f"{loc_pred:.2f}" if loc_pred is not None else "N/D"
    ax1.set_xlabel("Peso local LIME (casos)")
    ax1.set_title(
        f"LIME Tabular (fondo train) · Pred. local={loc_txt} · {r2_txt}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Explicabilidad Local [{algorithm.upper()}] · {district_label} · Semana {week_label}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = _save_figure(fig, out_base)
    plt.close(fig)
    return paths


def _plot_lags_and_temporal(
    plt,
    result: Dict[str, Any],
    *,
    algorithm: str,
    district_label: str,
    week_label: str,
    out_base: Path,
) -> Dict[str, str]:
    lags = result["lag_importance"]["lags"]
    x_labels = [item["label"] for item in lags]
    cases_abs = np.array([item["cases_abs"] for item in lags], dtype=float)
    climate_abs = np.array([item["climate_abs"] for item in lags], dtype=float)
    cum_total = np.array([item["cumulative_total_abs"] for item in lags], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=150)

    # Panel izquierdo: Rezagos t-1 .. t-12 (casos vs clima) + acumulada
    ax0 = axes[0]
    x = np.arange(len(x_labels))
    ax0.bar(x, cases_abs, label="Casos (|atribución|)", color="#2e86c1", alpha=0.88)
    ax0.bar(x, climate_abs, bottom=cases_abs, label="Clima (|atribución|)", color="#e67e22", alpha=0.88)
    ax0.set_xticks(x)
    ax0.set_xticklabels(x_labels, fontsize=8.5)
    ax0.set_ylabel("Importancia por rezago (|Δ casos|)")
    ax0.set_title("Importancia Acumulada de Rezagos (t-1 a t-12)", fontsize=10.5, fontweight="bold")
    ax0.grid(axis="y", alpha=0.25)

    ax0_r = ax0.twinx()
    ax0_r.plot(x, cum_total, color="#7d3c98", marker="o", linewidth=2.0, label="Acumulada total")
    ax0_r.set_ylabel("Importancia acumulada", color="#7d3c98")
    ax0_r.tick_params(axis="y", labelcolor="#7d3c98")

    h1, l1 = ax0.get_legend_handles_labels()
    h2, l2 = ax0_r.get_legend_handles_labels()
    ax0.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)

    # Panel derecho: Perturbación temporal y/o pesos reales de atención temporal
    ax1 = axes[1]
    temporal_pert = result.get("temporal_perturbation") or []
    attention = result.get("attention") or {}
    attn_temp = attention.get("temporal_weights") or []

    if attn_temp:
        t_labels = [item["label"] for item in attn_temp]
        tx = np.arange(len(t_labels))
        weights = [item["weight"] for item in attn_temp]
        ax1.bar(tx, weights, color="#16a085", alpha=0.85, label="Atención temporal media")
        n_heads = int(attention.get("n_heads") or 0)
        if n_heads > 1:
            for head_idx in range(n_heads):
                hw = [item["head_weights"][head_idx] for item in attn_temp]
                ax1.plot(tx, hw, marker=".", linewidth=1.3, linestyle="--", label=f"Cabeza {head_idx + 1}")
        ax1.set_xticks(tx)
        ax1.set_xticklabels(t_labels, fontsize=8.5)
        ax1.set_ylabel("Peso de atención (softmax)")
        ax1.set_title("Atención Temporal Multi-Cabezal (ConvLSTM-Attention)", fontsize=10.5, fontweight="bold")
        if temporal_pert:
            ax1_r = ax1.twinx()
            p_vals = [item["abs_delta"] for item in temporal_pert]
            ax1_r.plot(tx, p_vals, color="#c0392b", marker="s", linewidth=1.5, label="|Δ perturbación|")
            ax1_r.set_ylabel("|Δ predicción por neutralización|", color="#c0392b")
            ax1_r.tick_params(axis="y", labelcolor="#c0392b")
            hh1, ll1 = ax1.get_legend_handles_labels()
            hh2, ll2 = ax1_r.get_legend_handles_labels()
            ax1.legend(hh1 + hh2, ll1 + ll2, loc="upper left", fontsize=8)
        else:
            ax1.legend(loc="upper left", fontsize=8.5)
    elif temporal_pert:
        t_labels = [f"t{item['step']}" if item["step"] < 0 else "t" for item in temporal_pert]
        tx = np.arange(len(t_labels))
        deltas = [item["delta"] for item in temporal_pert]
        colors = ["#c0392b" if d > 0 else "#2471a3" for d in deltas]
        ax1.bar(tx, deltas, color=colors, alpha=0.88)
        ax1.axhline(0.0, color="#2c3e50", linewidth=0.9, linestyle="--")
        ax1.set_xticks(tx)
        ax1.set_xticklabels(t_labels, fontsize=8.5)
        ax1.set_ylabel("Δ predicción al neutralizar semana")
        ax1.set_title("Importancia Temporal por Ventana (Perturbación)", fontsize=10.5, fontweight="bold")
    else:
        summary = result["lag_importance"]["summary"]
        cats = ["Casos\n(t-1..t-12)", "Clima\n(t-1..t-12)", "Clima\n(t0)", "Ventanas\nclima", "Ventanas\ncasos"]
        vals = [
            summary["total_cases_lags_1_12"],
            summary["total_climate_lags_1_12"],
            summary["contemporaneous_climate_t0"],
            summary["rolling_climate_windows"],
            summary["rolling_cases_windows"],
        ]
        ax1.bar(cats, vals, color=["#2e86c1", "#e67e22", "#f39c12", "#d35400", "#1f618d"], alpha=0.88)
        ax1.set_ylabel("Atribución absoluta acumulada")
        ax1.set_title("Desglose Acumulado Casos vs. Clima", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Dinámica Temporal y Rezagos [{algorithm.upper()}] · {district_label} · Semana {week_label}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = _save_figure(fig, out_base)
    plt.close(fig)
    return paths


def _plot_spatial_attention(
    plt,
    attention: Dict[str, Any],
    *,
    algorithm: str,
    district_code: str,
    district_label: str,
    week_label: str,
    out_base: Path,
) -> Dict[str, str]:
    from matplotlib.patches import Rectangle

    matrix = np.asarray(attention["spatial_matrix"], dtype=float)
    rows, cols = matrix.shape
    districts = list(attention["spatial_districts"])
    cell_info = {(item["row"], item["col"]): item for item in districts}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=150)

    # Panel izquierdo: Mapa de calor 4x5 mapeado a los distritos reales
    ax0 = axes[0]
    im = ax0.imshow(matrix, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Peso de atención espacial")

    for r in range(rows):
        for c in range(cols):
            info = cell_info.get((r, c))
            val = float(matrix[r, c])
            txt_color = "white" if val > (matrix.max() * 0.55) else "#1b2631"
            if info is not None:
                short_code = info["district_code"].replace("PE-LL-", "")
                full_name = info["district_name"]
                if len(full_name) > 15 and " " in full_name:
                    parts = full_name.split(" ")
                    name_short = f"{' '.join(parts[:-1])}\n{parts[-1]}"
                else:
                    name_short = full_name[:16]
                label = f"{short_code}\n{name_short}\n{val:.3f}"
                if info["district_code"] == district_code:
                    rect = Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="#00b4d8", linewidth=2.6)
                    ax0.add_patch(rect)
            else:
                label = f"(vacía)\n{val:.3f}"
                txt_color = "#566573"
            ax0.text(c, r, label, ha="center", va="center", fontsize=7.5, color=txt_color, fontweight="bold" if info else "normal")

    ax0.set_xticks(range(cols))
    ax0.set_yticks(range(rows))
    ax0.set_xlabel("Columna espacial (Oeste → Este)")
    ax0.set_ylabel("Fila espacial (Norte → Sur)")
    ax0.set_title(
        f"Grilla Espacial {rows}×{cols} ({attention['n_occupied_cells']} distritos reales)",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel derecho: Ranking de distritos por atención espacial
    ax1 = axes[1]
    ordered = list(districts)[::-1]
    y_labels = [f"{d['district_name']} ({d['district_code']})" for d in ordered]
    weights = [float(d["weight"]) for d in ordered]
    colors = ["#00b4d8" if d["district_code"] == district_code else "#e67e22" for d in ordered]
    ax1.barh(range(len(ordered)), weights, color=colors, alpha=0.9)
    ax1.set_yticks(range(len(ordered)))
    ax1.set_yticklabels(y_labels, fontsize=8.5)
    ax1.set_xlabel("Peso de atención espacial (softmax)")
    ax1.set_title("Contribución Espacial por Distrito (Objetivo en cian)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"Atención Espacial Real [{algorithm.upper()}] · {district_label} · Semana {week_label}",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    paths = _save_figure(fig, out_base)
    plt.close(fig)
    return paths


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_xai_cli(args: argparse.Namespace) -> Dict[str, Any]:
    configure_threads()
    set_reproducible_seed(int(args.seed))
    plt = _import_pyplot()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_training_panel(args.data_mode)
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        district_code = resolve_district_code(args.district, db=db)

    model_entries = _resolve_model_paths(
        model_path=args.model_path,
        algorithms=args.models,
        horizon=int(args.horizon),
    )

    frame_cache: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    generated_reports: List[Dict[str, Any]] = []

    for requested_alg, artifact_path in model_entries:
        bundle = load_bundle(str(artifact_path))
        model = bundle["model"]
        algorithm = str(bundle.get("algorithm") or getattr(model, "key", requested_alg))
        horizon = int(bundle.get("horizon") or args.horizon)
        columns = list(bundle["feature_names"])
        mean = np.asarray(bundle["scaler_mean"], dtype=float)
        std = np.asarray(bundle["scaler_std"], dtype=float)
        window = int(bundle.get("sequence_window", 12))
        grid_shape = tuple(bundle.get("grid_shape", GRID_SHAPE))

        if horizon not in frame_cache:
            frame = build_features(panel, horizon=horizon)
            train_raw, _, _, _ = split_temporal(
                frame,
                validation_weeks=int(settings.ML_VALIDATION_WEEKS),
                test_weeks=int(settings.ML_TEST_WEEKS),
            )
            frame_cache[horizon] = (frame, train_raw)
        frame, train_raw = frame_cache[horizon]

        subset, week_label = _select_week_row(frame, district_code, args.week)
        result = explain_sample(
            model=model,
            panel=panel,
            frame=frame,
            subset=subset,
            background_frame=train_raw,
            columns=columns,
            mean=mean,
            std=std,
            window=window,
            district_grid=bundle.get("district_grid"),
            grid_shape=grid_shape,
            top_k=int(args.top_k),
            shap_samples=int(args.shap_samples),
            lime_samples=int(args.lime_samples),
            include_lime=True,
            raise_on_error=True,
            seed=int(args.seed),
        )

        if result.get("lime") is None:
            raise RuntimeError(f"LIME no devolvió resultados para {algorithm}.")
        if result["method"] != "SHAP":
            raise RuntimeError(f"SHAP falló para {algorithm}; se obtuvo método {result['method']!r}.")

        district_name = result["district_name"]
        ubigeo = result.get("ubigeo") or DISTRICT_TO_UBIGEO.get(district_code)
        district_label = f"{district_name} ({district_code}" + (f" / UBIGEO {ubigeo})" if ubigeo else ")")
        stem = f"xai_{algorithm}_{district_code}_{week_label}"

        artifacts: Dict[str, Any] = {}
        artifacts["shap_lime"] = _plot_shap_lime(
            plt,
            result,
            algorithm=algorithm,
            district_label=district_label,
            week_label=week_label,
            out_base=out_dir / f"{stem}_shap_lime",
        )
        artifacts["lags_temporal"] = _plot_lags_and_temporal(
            plt,
            result,
            algorithm=algorithm,
            district_label=district_label,
            week_label=week_label,
            out_base=out_dir / f"{stem}_lags",
        )
        if result.get("attention") is not None:
            artifacts["spatial_attention"] = _plot_spatial_attention(
                plt,
                result["attention"],
                algorithm=algorithm,
                district_code=district_code,
                district_label=district_label,
                week_label=week_label,
                out_base=out_dir / f"{stem}_spatial_attention",
            )

        shap_vec = np.asarray(result["shap_contributions"], dtype=float)
        lime_vec = np.asarray(result["lime"]["contributions"], dtype=float)
        payload = {
            "algorithm": algorithm,
            "model_path": str(artifact_path),
            "horizon": horizon,
            "district_code": district_code,
            "ubigeo": ubigeo,
            "district_name": district_name,
            "week_label": week_label,
            "origin_week": str(pd.Timestamp(subset["origin_week"].iloc[0]).date()),
            "data_mode": panel.data_mode,
            "is_synthetic_data": panel.is_synthetic,
            "prediction": result["prediction"],
            "actual_target": result["actual_target"],
            "shap": {
                "method": result["method"],
                "base_value": result["base_value"],
                "top_factors": result["factors"],
                "all_contributions": {
                    name: round(float(shap_vec[i]), 6) for i, name in enumerate(columns)
                },
            },
            "lime": {
                "method": result["lime"]["method"],
                "intercept": result["lime"]["intercept"],
                "local_prediction": result["lime"]["local_prediction"],
                "r2_score": result["lime"]["r2_score"],
                "top_factors": result["lime"]["factors"],
                "all_contributions": {
                    name: round(float(lime_vec[i]), 6) for i, name in enumerate(columns)
                },
            },
            "lag_importance": result["lag_importance"],
            "temporal_perturbation": result["temporal_perturbation"],
            "attention": result["attention"],
            "artifacts": artifacts,
        }

        json_path = out_dir / f"{stem}.json"
        json_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        payload["json_path"] = str(json_path)
        generated_reports.append(payload)
        logger.info("XAI generado [%s | %s | %s] -> %s", algorithm, district_code, week_label, json_path)

    return {
        "district_code": district_code,
        "week_arg": args.week,
        "output_dir": str(out_dir),
        "n_models": len(generated_reports),
        "reports": [
            {
                "algorithm": r["algorithm"],
                "model_path": r["model_path"],
                "prediction": r["prediction"],
                "actual_target": r["actual_target"],
                "shap_method": r["shap"]["method"],
                "lime_r2": r["lime"]["r2_score"],
                "has_attention": r["attention"] is not None,
                "json_path": r["json_path"],
                "artifacts": r["artifacts"],
            }
            for r in generated_reports
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generación de explicabilidad XAI (SHAP, LIME, rezagos y atención real) en Fase 2."
    )
    parser.add_argument(
        "--district",
        type=str,
        default="PE-LL-TRU",
        help="Código de distrito ('PE-LL-TRU') o alias UBIGEO (p. ej. '130101').",
    )
    parser.add_argument(
        "--week",
        type=str,
        default="latest",
        help="Semana epidemiológica de origen: YYYYWW (p. ej. 202415), YYYY-Www, YYYY-MM-DD o 'latest'.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Ruta opcional a un artefacto .joblib específico.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["auto"],
        help="Modelos a explicar si no se pasa --model-path: auto (xgboost + convlstm_attention), all o claves.",
    )
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte en semanas (defecto: 1).")
    parser.add_argument(
        "--data-mode",
        choices=["auto", "real", "synthetic"],
        default="auto",
        help="Modo de datos para load_panel (defecto: auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_XAI_DIR,
        help=f"Directorio de salida para PNG/SVG/JSON (defecto: {DEFAULT_XAI_DIR}).",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Número de factores en los gráficos (defecto: 12).")
    parser.add_argument("--shap-samples", type=int, default=220, help="Muestras para KernelExplainer (defecto: 220).")
    parser.add_argument("--lime-samples", type=int, default=500, help="Muestras para LimeTabularExplainer (defecto: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible (defecto: 42).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_cli_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_xai_cli(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
