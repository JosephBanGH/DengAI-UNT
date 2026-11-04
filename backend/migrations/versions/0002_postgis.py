"""Columna geoespacial nativa para PostgreSQL/PostGIS.

En SQLite la geometría vive en ``geographic_units.geometry_geojson``. Cuando la base es
PostgreSQL con PostGIS se añade ``geom geometry(MultiPolygon, 4326)``, un índice GiST y
un trigger que mantiene ambas representaciones sincronizadas.
"""
from __future__ import annotations

from alembic import op

from app.core.config import settings

revision = "0002_postgis"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if not settings.POSTGIS_ENABLED or settings.is_sqlite:
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute(
        """
        ALTER TABLE geographic_units
        ADD COLUMN IF NOT EXISTS geom geometry(MultiPolygon, 4326)
        """
    )
    op.execute(
        """
        UPDATE geographic_units
        SET geom = ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(geometry_geojson), 4326))
        WHERE geometry_geojson IS NOT NULL AND geom IS NULL
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_geographic_units_geom ON geographic_units USING GIST (geom)"
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION sync_geographic_units_geom() RETURNS trigger AS $$
        BEGIN
            IF NEW.geometry_geojson IS NOT NULL THEN
                NEW.geom := ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(NEW.geometry_geojson), 4326));
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_sync_geographic_units_geom ON geographic_units;
        CREATE TRIGGER trg_sync_geographic_units_geom
        BEFORE INSERT OR UPDATE OF geometry_geojson ON geographic_units
        FOR EACH ROW EXECUTE FUNCTION sync_geographic_units_geom();
        """
    )


def downgrade() -> None:
    if not settings.POSTGIS_ENABLED or settings.is_sqlite:
        return
    op.execute("DROP TRIGGER IF EXISTS trg_sync_geographic_units_geom ON geographic_units")
    op.execute("DROP FUNCTION IF EXISTS sync_geographic_units_geom()")
    op.execute("DROP INDEX IF EXISTS ix_geographic_units_geom")
    op.execute("ALTER TABLE geographic_units DROP COLUMN IF EXISTS geom")
