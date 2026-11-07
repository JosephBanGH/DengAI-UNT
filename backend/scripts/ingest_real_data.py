"""CLI de ingesta real: MINSA/CDC Perú + NASA POWER.

Uso (desde la raíz del repositorio):

    python backend/scripts/ingest_real_data.py --start-year 2015 --end-year 2024
    python backend/scripts/ingest_real_data.py --skip-climate --force

El script guarda los archivos crudos en ``data/raw`` con checksum SHA-256 y deja
evidencia de la ingesta en ``import_jobs`` y ``data_quality_records``. No inventa
datos: si una fuente falla, el paso queda reportado como error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_DIR / "libs", PROJECT_DIR / "backend"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from app.db.base import Base  # noqa: E402
from app.db.seed import ensure_data_sources, ensure_geo_units  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.services.real_data_service import ingest_real_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta real MINSA + NASA POWER")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--districts", nargs="*", default=None)
    parser.add_argument("--skip-cases", action="store_true")
    parser.add_argument("--skip-climate", action="store_true")
    parser.add_argument("--force", action="store_true", help="Redescarga el CSV MINSA aunque exista en raw")
    parser.add_argument("--json", action="store_true", help="Imprime el resultado completo en JSON")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        ensure_geo_units(session)
        ensure_data_sources(session)
        session.commit()
        result = ingest_real_data(
            session,
            actor="cli:ingest_real_data",
            start_year=args.start_year,
            end_year=args.end_year,
            district_codes=args.districts,
            include_cases=not args.skip_cases,
            include_climate=not args.skip_climate,
            force_download=args.force,
        )
    finally:
        session.close()

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"Modo efectivo: {result['effective_data_mode']}")
        for name, step in result.get("steps", {}).items():
            status = step.get("status", "?")
            inserted = step.get("inserted", step.get("rows_inserted", 0))
            updated = step.get("updated", step.get("rows_updated", 0))
            print(f"- {name}: {status} · insertados={inserted} · actualizados={updated}")
            if step.get("errors"):
                for error in step["errors"][:5]:
                    print(f"    error: {error}")
        provenance = result.get("provenance", {})
        cases = provenance.get("cases", {})
        climate = provenance.get("climate", {})
        print(
            f"Casos reales: {cases.get('real', 0)}/{cases.get('total', 0)} · "
            f"clima real: {climate.get('real', 0)}/{climate.get('total', 0)}"
        )
    failed = [
        name
        for name, step in result.get("steps", {}).items()
        if step.get("status") not in {"COMPLETED", None}
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
