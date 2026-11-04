-- Extensiones requeridas por la plataforma (se ejecutan al inicializar el contenedor).
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Esquema dedicado para la aplicación (opcional; por defecto se usa public).
-- CREATE SCHEMA IF NOT EXISTS dengai;

-- Nota: las tablas las crea Alembic (backend/migrations) o SQLAlchemy al arrancar.
-- Este script solo garantiza las extensiones geoespaciales.
SELECT postgis_version();
