"""Orquestador CLI de optimización Optuna para uno o varios modelos (Fase 2)."""
from __future__ import annotations

import copy
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.common import (  # noqa: E402
    build_base_parser,
    configure_cli_logging,
    resolve_safe_output_dir,
    run_training_pipeline,
)

ALL_ALGORITHMS = ("xgboost", "lstm", "convlstm", "cnn_lstm", "convlstm_attention")


def _resolve_algorithms(models_arg: Sequence[str], algorithm_arg: Optional[str]) -> List[str]:
    requested = [algorithm_arg] if algorithm_arg else list(models_arg)
    if "all" in requested:
        return list(ALL_ALGORITHMS)
    unknown = [m for m in requested if m not in ALL_ALGORITHMS]
    if unknown:
        raise ValueError(f"Algoritmo(s) desconocido(s): {unknown}. Opciones válidas: {ALL_ALGORITHMS}")
    return list(dict.fromkeys(requested))


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_cli_logging()
    parser = build_base_parser(
        "Optimización Bayesiana Optuna para uno o varios modelos DengAI-UNT (Fase 2).",
        allow_n_estimators=True,
        allow_models_selection=True,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=ALL_ALGORITHMS,
        help="Alias opcional para seleccionar un único algoritmo.",
    )
    args = parser.parse_args(argv)

    algorithms = _resolve_algorithms(args.models, args.algorithm)
    output_dir = resolve_safe_output_dir(args.output_dir)
    batch_id = args.run_id or f"optuna-{uuid.uuid4().hex[:8]}"

    results: Dict[str, Any] = {}
    for algorithm in algorithms:
        per_args = copy.copy(args)
        per_args.run_id = f"{batch_id}-{algorithm}"
        study_name = f"{args.study_prefix}-{algorithm}-h{args.horizon}"
        summary = run_training_pipeline(
            algorithm,
            per_args,
            study_name=study_name if args.storage else None,
            storage=args.storage,
        )
        results[algorithm] = {
            "run_id": summary["run_id"],
            "best_val_mae": summary["best_val_mae"],
            "best_params": summary["best_params"],
            "metrics": summary["metrics"],
            "artifact_path": summary["artifact_path"],
            "predictions_path": summary["predictions_path"],
            "telemetry_path": summary["telemetry_path"],
        }

    consolidated = {
        "batch_id": batch_id,
        "horizon": int(args.horizon),
        "trials": int(args.trials),
        "epochs": int(args.epochs) if args.epochs is not None else 40,
        "seed": int(args.seed),
        "algorithms": algorithms,
        "results": results,
    }
    consolidated_path = output_dir / f"optuna_summary-h{int(args.horizon)}-{batch_id}.json"
    consolidated_path.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
    consolidated["consolidated_path"] = str(consolidated_path)
    print(json.dumps(consolidated, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
