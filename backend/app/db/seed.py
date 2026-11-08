"""Poblado inicial determinista: catálogos, roles, usuarios, geografía y datos sintéticos.

Los datos de demostración se generan con semilla fija y se marcan SIEMPRE como
``is_synthetic=True`` en la base de datos, en la API y en la interfaz. No representan
vigilancia epidemiológica real: son un sustituto reproducible mientras no se conecte
una fuente autorizada (MINSA/CDC Perú, CDC, WHO).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.permissions import PERMISSIONS, ROLE_TEMPLATES
from app.db.base import utcnow
from app.models.epi import ClimateObservation, DataSource, DengueCase, EnvironmentalVariable
from app.models.geo import EpiWeek, GeographicUnit
from app.models.security import Permission, Role, SystemConfig, User

logger = logging.getLogger("dengai.seed")

SYNTHETIC_SEED = 20240517
START_YEAR = 2019

# Distritos de La Libertad (Perú) usados como unidad espacial de demostración.
DISTRICTS: List[Dict[str, object]] = [
    {"code": "PE-LL-TRU", "name": "Trujillo", "province": "Trujillo", "alt": 34, "lat": -8.1061, "lon": -79.0287, "pop": 314_242, "area": 45.2},
    {"code": "PE-LL-VLH", "name": "Víctor Larco Herrera", "province": "Trujillo", "alt": 25, "lat": -8.1353, "lon": -79.0479, "pop": 68_437, "area": 18.0},
    {"code": "PE-LL-EPO", "name": "El Porvenir", "province": "Trujillo", "alt": 90, "lat": -8.0928, "lon": -78.9836, "pop": 190_461, "area": 36.7},
    {"code": "PE-LL-LES", "name": "La Esperanza", "province": "Trujillo", "alt": 100, "lat": -8.0683, "lon": -78.9979, "pop": 201_020, "area": 15.5},
    {"code": "PE-LL-FDM", "name": "Florencia de Mora", "province": "Trujillo", "alt": 90, "lat": -8.0825, "lon": -78.9761, "pop": 40_135, "area": 4.2},
    {"code": "PE-LL-HUA", "name": "Huanchaco", "province": "Trujillo", "alt": 20, "lat": -8.0797, "lon": -79.1194, "pop": 58_925, "area": 34.6},
    {"code": "PE-LL-LAR", "name": "Laredo", "province": "Trujillo", "alt": 90, "lat": -8.0892, "lon": -78.9600, "pop": 37_218, "area": 77.5},
    {"code": "PE-LL-MOC", "name": "Moche", "province": "Trujillo", "alt": 20, "lat": -8.1697, "lon": -79.0106, "pop": 40_271, "area": 25.1},
    {"code": "PE-LL-SAL", "name": "Salaverry", "province": "Trujillo", "alt": 10, "lat": -8.2239, "lon": -78.9772, "pop": 19_364, "area": 8.4},
    {"code": "PE-LL-ASC", "name": "Ascope", "province": "Ascope", "alt": 230, "lat": -7.7144, "lon": -79.1078, "pop": 15_432, "area": 92.0},
    {"code": "PE-LL-PAI", "name": "Paiján", "province": "Ascope", "alt": 80, "lat": -7.7333, "lon": -79.3000, "pop": 30_027, "area": 118.6},
    {"code": "PE-LL-CHE", "name": "Chepén", "province": "Chepén", "alt": 130, "lat": -7.2253, "lon": -79.4297, "pop": 40_664, "area": 287.3},
    {"code": "PE-LL-PAC", "name": "Pacasmayo", "province": "Pacasmayo", "alt": 20, "lat": -7.4006, "lon": -79.5714, "pop": 30_642, "area": 30.2},
    {"code": "PE-LL-SDC", "name": "Santiago de Chuco", "province": "Santiago de Chuco", "alt": 3120, "lat": -8.1444, "lon": -78.1753, "pop": 12_145, "area": 142.8},
]

DATA_SOURCES: List[Dict[str, object]] = [
    {
        "code": "MINSA_VIGILANCIA", "name": "Vigilancia epidemiológica MINSA/CDC Perú",
        "kind": "epidemiological", "provider": "MINSA / CDC Perú",
        "url": "https://www.dge.gob.pe/", "license": "Datos abiertos del Estado peruano (uso con atribución)",
        "access_method": "file", "integration_status": "PENDING_CREDENTIALS",
        "refresh_frequency": "weekly",
        "license_notes": "Requiere verificación de términos de uso y desidentificación antes de producción. "
                         "La plataforma solo ingiere datos agregados por distrito-semana.",
    },
    {
        "code": "CDC_DENGUE", "name": "CDC Dengue Data",
        "kind": "epidemiological", "provider": "Centers for Disease Control and Prevention",
        "url": "https://www.cdc.gov/dengue/", "license": "Dominio público (US Government work)",
        "access_method": "file", "integration_status": "AVAILABLE_MANUAL",
        "refresh_frequency": "weekly",
        "license_notes": "Descarga manual en CSV; el adaptador valida el esquema antes de cargar.",
    },
    {
        "code": "WHO_GHO", "name": "WHO Global Health Observatory",
        "kind": "epidemiological", "provider": "World Health Organization",
        "url": "https://www.who.int/data/gho", "license": "CC BY-NC-SA 3.0 IGO",
        "access_method": "api", "integration_status": "AVAILABLE_MANUAL",
        "refresh_frequency": "annual",
        "license_notes": "Uso no comercial; citar la fuente WHO GHO en reportes y publicaciones.",
    },
    {
        "code": "NASA_POWER", "name": "NASA POWER (meteorología)",
        "kind": "climate", "provider": "NASA Langley Research Center",
        "url": "https://power.larc.nasa.gov/", "license": "Datos de acceso abierto NASA (uso libre con atribución)",
        "access_method": "api", "integration_status": "AVAILABLE_LIVE",
        "refresh_frequency": "weekly",
        "license_notes": "API pública sin clave para T2M, RH2M y PRECTOTCORR; el adaptador está implementado "
                         "en app/integrations/nasa_power.py y requiere habilitar LIVE_INTEGRATIONS_ENABLED.",
    },
    {
        "code": "CHIRPS", "name": "CHIRPS (precipitación)",
        "kind": "climate", "provider": "UCSB Climate Hazards Center",
        "url": "https://www.chc.ucsb.edu/data/chirps", "license": "CC0 / uso libre con atribución",
        "access_method": "file", "integration_status": "AVAILABLE_MANUAL",
        "refresh_frequency": "monthly",
        "license_notes": "GeoTIFF/NetCDF; la ingesta requiere el servicio de extracción por distrito (ETL).",
    },
    {
        "code": "WORLDPOP", "name": "WorldPop (población y distribución)",
        "kind": "environmental", "provider": "WorldPop, University of Southampton",
        "url": "https://www.worldpop.org/", "license": "CC BY 4.0",
        "access_method": "file", "integration_status": "AVAILABLE_MANUAL",
        "refresh_frequency": "annual",
        "license_notes": "Población estimada por unidad administrativa; se usa para denominadores de incidencia.",
    },
    {
        "code": "SYNTHETIC_DEMO", "name": "Conjunto sintético de demostración (La Libertad)",
        "kind": "synthetic", "provider": "Generado por la plataforma",
        "url": "", "license": "Sintético, uso interno para pruebas",
        "access_method": "generated", "integration_status": "READY",
        "refresh_frequency": "on_demand", "is_synthetic": True,
        "license_notes": "Datos generados con semilla fija. NO son datos epidemiológicos reales y se "
                         "identifican como sintéticos en toda la interfaz y en los reportes.",
    },
]

DEFAULT_CONFIG: List[Tuple[str, dict, str]] = [
    (
        "risk_thresholds",
        {
            "method": "incidencia_semanal_por_100k",
            "justification": (
                "Umbrales expresados como incidencia semanal por 100 000 habitantes. Los valores "
                "corresponden a la práctica de vigilancia: por debajo de 5 casos/100k (bajo), "
                "5-20 (medio), 20-50 (alto) y más de 50 (crítico, compatible con escenario de brote). "
                "Son parametrizables por el administrador epidemiológico en M15."
            ),
            "low": 5.0,
            "moderate": 20.0,
            "high": 50.0,
        },
        "Umbrales de clasificación de riesgo epidemiológico (M11/M15).",
    ),
    (
        "forecast_defaults",
        {"horizons": [1, 2, 4, 8], "interval_level": 0.9, "alert_min_level": "HIGH"},
        "Parámetros por defecto de pronóstico y alertas.",
    ),
    (
        "validation_policy",
        {
            "strategy": "walk_forward",
            "validation_weeks": settings.ML_VALIDATION_WEEKS,
            "test_weeks": settings.ML_TEST_WEEKS,
            "geographic_validation": True,
            "note": (
                "El conjunto de prueba nunca participa en la selección de hiperparámetros "
                "(Optuna optimiza solo sobre entrenamiento y validación interna)."
            ),
        },
        "Política de validación temporal y geográfica (prevención de data leakage).",
    ),
    (
        "notifications",
        {"channels": ["in_app"], "email_enabled": False, "webhook_url": ""},
        "Configuración de notificaciones de alertas (M11/M12).",
    ),
    (
        "system_flags",
        {
            "synthetic_data_mode": True,
            "assistant_provider": settings.LLM_PROVIDER,
            "show_synthetic_banner": True,
        },
        "Banderas operativas del sistema.",
    ),
]


def _iso_weeks(start_year: int, until: date) -> List[Dict[str, object]]:
    weeks: List[Dict[str, object]] = []
    monday = date.fromisocalendar(start_year, 1, 1)
    while monday <= until:
        iso = monday.isocalendar()
        weeks.append(
            {
                "year": iso.year,
                "week": iso.week,
                "start_date": monday,
                "end_date": monday + timedelta(days=6),
                "label": f"{iso.year}-W{iso.week:02d}",
            }
        )
        monday += timedelta(days=7)
    return weeks


def _voronoi_geometry(districts: List[Dict[str, object]]) -> Dict[str, str]:
    """Genera polígonos de Voronoi alrededor de los centroides, recortados a la extensión.

    No son límites administrativos oficiales: son células de teselación para la
    visualización coroplética cuando no se ha cargado la cartografía oficial.
    """
    try:
        from shapely.geometry import MultiPoint, box
        from shapely.ops import voronoi_diagram

        points = MultiPoint([(float(d["lon"]), float(d["lat"])) for d in districts])
        minx, miny, maxx, maxy = points.bounds
        pad_x = (maxx - minx) * 0.18 + 0.01
        pad_y = (maxy - miny) * 0.18 + 0.01
        envelope = box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
        diagram = voronoi_diagram(points, envelope=envelope)
        result: Dict[str, str] = {}
        cells = list(diagram.geoms) if hasattr(diagram, "geoms") else [diagram]
        used = set()
        for district in districts:
            from shapely.geometry import Point

            point = Point(float(district["lon"]), float(district["lat"]))
            for idx, cell in enumerate(cells):
                if idx in used:
                    continue
                if cell.contains(point) or cell.touches(point):
                    used.add(idx)
                    result[str(district["code"])] = json.dumps(
                        {"type": "Polygon", "coordinates": [list(map(list, cell.exterior.coords))]}
                    )
                    break
        if len(result) == len(districts):
            return result
    except Exception:  # pragma: no cover - respaldo geométrico
        logger.warning("Shapely no disponible; se usan celdas rectangulares de respaldo")
    result = {}
    for district in districts:
        lon, lat = float(district["lon"]), float(district["lat"])
        d = 0.055
        ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]
        result[str(district["code"])] = json.dumps({"type": "Polygon", "coordinates": [ring]})
    return result


def ensure_rbac(db: Session) -> None:
    existing = {p.code: p for p in db.scalars(select(Permission))}
    for code, module, description in PERMISSIONS:
        perm = existing.get(code)
        if perm is None:
            db.add(Permission(code=code, module=module, description=description))
        else:
            perm.module = module
            perm.description = description
    db.flush()

    perms = {p.code: p for p in db.scalars(select(Permission))}
    for slug, template in ROLE_TEMPLATES.items():
        role = db.scalar(select(Role).where(Role.slug == slug))
        if role is None:
            role = Role(slug=slug, name=str(template["name"]), description=str(template["description"]), is_system=True)
            db.add(role)
        else:
            role.name = str(template["name"])
            role.description = str(template["description"])
        role.permissions = [perms[code] for code in template["permissions"] if code in perms]  # type: ignore[index]
    db.flush()


def ensure_users(db: Session) -> None:
    from app.core.security import hash_password

    roles = {r.slug: r for r in db.scalars(select(Role))}
    admin = db.scalar(select(User).where(User.email == settings.BOOTSTRAP_ADMIN_EMAIL))
    if admin is None:
        db.add(
            User(
                email=settings.BOOTSTRAP_ADMIN_EMAIL,
                full_name="Administrador del sistema",
                password_hash=hash_password(settings.BOOTSTRAP_ADMIN_PASSWORD),
                position="Administrador de plataforma",
                organization="UNT - DengAI",
                is_active=True,
                is_superuser=True,
                must_change_password=True,
                roles=[roles["admin_sistema"]],
            )
        )
    if not settings.SEED_DEMO_USERS:
        db.flush()
        return

    demo = [
        ("epidemiologo@dengai.unt.edu.pe", "Epidemiólogo de demostración", "epidemiologo", "Vigilancia epidemiológica"),
        ("analista@dengai.unt.edu.pe", "Analista de datos de demostración", "analista_datos", "Ciencia de datos"),
        ("investigador@dengai.unt.edu.pe", "Investigador de demostración", "investigador", "Investigación"),
        ("auditor@dengai.unt.edu.pe", "Auditor de demostración", "auditor", "Control interno"),
        ("consulta@dengai.unt.edu.pe", "Usuario de consulta de demostración", "consulta", "Consulta externa"),
        ("admin.epi@dengai.unt.edu.pe", "Administrador epidemiológico de demostración", "admin_epidemiologico", "Gerencia de epidemiología"),
    ]
    for email, full_name, role_slug, position in demo:
        if db.scalar(select(User).where(User.email == email)) is None:
            db.add(
                User(
                    email=email,
                    full_name=full_name,
                    password_hash=hash_password(settings.DEMO_USER_PASSWORD),
                    position=position,
                    organization="UNT - DengAI (demostración)",
                    is_active=True,
                    is_superuser=False,
                    must_change_password=True,
                    roles=[roles[role_slug]],
                )
            )
    db.flush()


def ensure_data_sources(db: Session) -> None:
    for item in DATA_SOURCES:
        source = db.scalar(select(DataSource).where(DataSource.code == item["code"]))
        if source is None:
            db.add(
                DataSource(
                    code=str(item["code"]), name=str(item["name"]), kind=str(item["kind"]),
                    provider=str(item["provider"]), url=str(item["url"]), license=str(item["license"]),
                    license_notes=str(item.get("license_notes", "")),
                    access_method=str(item["access_method"]),
                    integration_status=str(item["integration_status"]),
                    refresh_frequency=str(item["refresh_frequency"]),
                    is_synthetic=bool(item.get("is_synthetic", False)),
                )
            )
    db.flush()


def ensure_geo_units(db: Session) -> None:
    if db.scalar(select(func.count()).select_from(GeographicUnit)) or 0:
        return
    geometries = _voronoi_geometry(DISTRICTS)
    for district in DISTRICTS:
        db.add(
            GeographicUnit(
                code=str(district["code"]), name=str(district["name"]), level="district",
                parent_code="PE-LL", province=str(district["province"]), department="La Libertad",
                altitude_m=float(district["alt"]), area_km2=float(district["area"]),
                population=int(district["pop"]), population_year=2024,
                centroid_lat=float(district["lat"]), centroid_lon=float(district["lon"]),
                geometry_geojson=geometries.get(str(district["code"])),
            )
        )
    db.flush()


def ensure_epi_weeks(db: Session) -> List[EpiWeek]:
    if db.scalar(select(func.count()).select_from(EpiWeek)) or 0:
        return list(db.scalars(select(EpiWeek).order_by(EpiWeek.start_date)))
    today = utcnow().date()
    for item in _iso_weeks(START_YEAR, today):
        db.add(EpiWeek(**item))  # type: ignore[arg-type]
    db.flush()
    return list(db.scalars(select(EpiWeek).order_by(EpiWeek.start_date)))


def _synthetic_series(districts: List[GeographicUnit], weeks: List[EpiWeek]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(SYNTHETIC_SEED)
    n_d, n_w = len(districts), len(weeks)
    altitude = np.array([float(d.altitude_m or 0) for d in districts])
    population = np.array([float(d.population or 1) for d in districts])

    week_of_year = np.array([w.week for w in weeks], dtype=float)
    # Estacionalidad: pico entre las semanas 8 y 16 (temporada de lluvias en la costa norte).
    seasonal = 0.55 + 0.45 * np.cos(2 * np.pi * (week_of_year - 12) / 52.0)

    weather = np.zeros((n_w,))
    for idx, week in enumerate(weeks):
        # Ondas plurianuales tipo El Niño (2023 reforzado) con ruido de baja frecuencia.
        year_factor = {2019: 0.85, 2020: 0.55, 2021: 0.75, 2022: 1.05, 2023: 1.55, 2024: 1.15, 2025: 0.95, 2026: 0.9}
        base = year_factor.get(week.year, 1.0)
        weather[idx] = base * (1.0 + 0.12 * math.sin(2 * math.pi * week.week / 26.0))

    temperature = (
        24.5 - altitude[:, None] * 0.0062  # gradiente altitudinal
        + 2.6 * seasonal[None, :]
        + rng.normal(0, 0.8, size=(n_d, n_w))
    )
    humidity = np.clip(78 - altitude[:, None] * 0.004 + 7.0 * seasonal[None, :] + rng.normal(0, 3.0, (n_d, n_w)), 25, 99)
    precipitation = np.clip(
        (6 + 46 * seasonal[None, :]) * weather[None, :] + rng.gamma(2.0, 4.0, size=(n_d, n_w)),
        0,
        None,
    )

    # Riesgo relativo log-lineal: temperatura templada-cálida, humedad y lluvia favorecen al vector.
    temp_effect = np.exp(-((temperature - 26.0) ** 2) / (2 * 4.2 ** 2))
    humidity_effect = np.clip((humidity - 55) / 35, 0, 1.6)
    rain_effect = np.log1p(precipitation / 25.0)
    altitude_penalty = np.clip(1.0 - altitude / 2600.0, 0.05, 1.0)

    rate = (
        (population / 100_000.0)[:, None]
        * 9.0
        * temp_effect
        * (0.45 + humidity_effect)
        * (0.35 + rain_effect)
        * altitude_penalty[:, None]
        * weather[None, :]
        * seasonal[None, :]
    )
    cases = rng.poisson(np.clip(rate, 0.01, None)).astype(float)
    # Brotes localizados: semanas con exceso epidémico en distritos densos.
    outbreak_mask = (rng.random((n_d, n_w)) < 0.012) & (population[:, None] > 35_000)
    cases = cases + np.where(outbreak_mask, rng.integers(25, 140, size=(n_d, n_w)), 0)

    # Escenario de demostración: ascenso epidémico en las últimas semanas de la serie para
    # ejercitar la clasificación de riesgo, los mapas y las alertas tempranas. Forma parte del
    # conjunto SINTÉTICO y no representa un brote real.
    demo_weeks = min(8, n_w)
    dense = population / population.max()
    for offset in range(demo_weeks):
        growth = (offset + 1) / demo_weeks
        affected = rng.random(n_d) < 0.55
        cases[:, n_w - demo_weeks + offset] += np.where(affected, np.round(growth * 55.0 * dense), 0)

    climate = {
        "temperature_mean": temperature,
        "temperature_max": temperature + 3.2 + rng.normal(0, 0.5, (n_d, n_w)),
        "temperature_min": temperature - 3.6 + rng.normal(0, 0.5, (n_d, n_w)),
        "humidity_mean": humidity,
        "precipitation_mm": precipitation,
        "wind_speed": np.clip(2.8 + rng.normal(0, 0.6, (n_d, n_w)), 0.4, 8),
        "solar_radiation": np.clip(18 + 4 * seasonal[None, :] + rng.normal(0, 1.2, (n_d, n_w)), 6, 30),
    }
    return cases, climate


def ensure_synthetic_data(db: Session) -> None:
    if (db.scalar(select(func.count()).select_from(DengueCase)) or 0) > 0:
        return
    districts = list(db.scalars(select(GeographicUnit).order_by(GeographicUnit.code)))
    weeks = list(db.scalars(select(EpiWeek).order_by(EpiWeek.start_date)))
    if not districts or not weeks:
        return
    source = db.scalar(select(DataSource).where(DataSource.code == "SYNTHETIC_DEMO"))
    cases, climate = _synthetic_series(districts, weeks)

    for d_idx, district in enumerate(districts):
        pop = float(district.population or 1)
        for w_idx, week in enumerate(weeks):
            confirmed = int(cases[d_idx, w_idx])
            db.add(
                DengueCase(
                    district_code=district.code, epi_year=week.year, epi_week=week.week,
                    week_start=week.start_date, confirmed_cases=confirmed,
                    probable_cases=int(confirmed * 0.35), hospitalized=int(confirmed * 0.11),
                    deaths=int(confirmed * 0.004), severity="SIN_SIGNOS_ALARMA" if confirmed < 30 else "CON_SIGNOS_ALARMA",
                    source_id=source.id if source else None, is_synthetic=True, data_quality_flag="OK",
                )
            )
            db.add(
                ClimateObservation(
                    district_code=district.code, epi_year=week.year, epi_week=week.week,
                    week_start=week.start_date,
                    temperature_mean=round(float(climate["temperature_mean"][d_idx, w_idx]), 2),
                    temperature_max=round(float(climate["temperature_max"][d_idx, w_idx]), 2),
                    temperature_min=round(float(climate["temperature_min"][d_idx, w_idx]), 2),
                    humidity_mean=round(float(climate["humidity_mean"][d_idx, w_idx]), 2),
                    precipitation_mm=round(float(climate["precipitation_mm"][d_idx, w_idx]), 2),
                    wind_speed=round(float(climate["wind_speed"][d_idx, w_idx]), 2),
                    solar_radiation=round(float(climate["solar_radiation"][d_idx, w_idx]), 2),
                    altitude_m=district.altitude_m, is_synthetic=True, provider="synthetic",
                    source_id=source.id if source else None, data_quality_flag="OK",
                )
            )
            if w_idx % 4 == 0:
                db.add(
                    EnvironmentalVariable(
                        district_code=district.code, epi_year=week.year, epi_week=week.week,
                        name="population_density", value=round(pop / float(district.area_km2 or 1), 2),
                        unit="hab/km2", is_synthetic=True,
                    )
                )
    db.flush()
    logger.info("Datos sintéticos generados: %d distritos x %d semanas", len(districts), len(weeks))


def ensure_config(db: Session) -> None:
    for key, value, description in DEFAULT_CONFIG:
        record = db.scalar(select(SystemConfig).where(SystemConfig.key == key))
        if record is None:
            db.add(SystemConfig(key=key, value=value, description=description, updated_by="system"))


def run_seed(db: Session) -> Dict[str, int]:
    ensure_rbac(db)
    ensure_users(db)
    ensure_data_sources(db)
    ensure_geo_units(db)
    ensure_epi_weeks(db)
    ensure_config(db)
    ensure_synthetic_data(db)
    db.commit()
    return {
        "permissions": db.scalar(select(func.count()).select_from(Permission)) or 0,
        "roles": db.scalar(select(func.count()).select_from(Role)) or 0,
        "users": db.scalar(select(func.count()).select_from(User)) or 0,
        "districts": db.scalar(select(func.count()).select_from(GeographicUnit)) or 0,
        "weeks": db.scalar(select(func.count()).select_from(EpiWeek)) or 0,
        "cases": db.scalar(select(func.count()).select_from(DengueCase)) or 0,
        "climate": db.scalar(select(func.count()).select_from(ClimateObservation)) or 0,
    }
