import { useMemo, useState } from "react";
import type { FeatureCollection, GeoFeature } from "../api/types";

const RISK_COLORS: Record<string, string> = {
  CRITICAL: "#b3261e",
  HIGH: "#e07a1f",
  MODERATE: "#e5b52c",
  LOW: "#4c9f70",
  UNKNOWN: "#b8c2cc",
};

interface ChoroplethProps {
  collection: FeatureCollection | null;
  mode: "risk" | "incidence";
  onSelect?: (feature: GeoFeature) => void;
  height?: number;
  selected?: string | null;
}

export function Choropleth({ collection, mode, onSelect, height = 430, selected }: ChoroplethProps) {
  const [hover, setHover] = useState<{ feature: GeoFeature; x: number; y: number } | null>(null);
  const width = 720;

  const projection = useMemo(() => {
    const features = collection?.features ?? [];
    const lons: number[] = [];
    const lats: number[] = [];
    features.forEach((feature) => {
      const geometry = feature.geometry;
      if (!geometry) return;
      const rings =
        geometry.type === "Polygon"
          ? (geometry.coordinates as number[][][])
          : (geometry.coordinates as number[][][][]).flat();
      rings.forEach((ring) => ring.forEach(([lon, lat]) => {
        lons.push(lon);
        lats.push(lat);
      }));
    });
    if (!lons.length) {
      return null;
    }
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const pad = 18;
    const spanX = Math.max(1e-6, maxLon - minLon);
    const spanY = Math.max(1e-6, maxLat - minLat);
    const scale = Math.min((width - pad * 2) / spanX, (height - pad * 2) / spanY);
    const offsetX = (width - spanX * scale) / 2;
    const offsetY = (height - spanY * scale) / 2;
    return {
      project: (lon: number, lat: number): [number, number] => [
        offsetX + (lon - minLon) * scale,
        offsetY + (maxLat - lat) * scale,
      ],
    };
  }, [collection, height]);

  const scaleMax = useMemo(() => {
    if (!collection || mode !== "incidence") return 1;
    const values = collection.features
      .map((feature) => Number(feature.properties.incidence_per_100k ?? 0))
      .filter((value) => Number.isFinite(value));
    return Math.max(1, ...values);
  }, [collection, mode]);

  if (!collection || !projection) {
    return <div className="empty-state"><strong>Sin cartografía disponible</strong>Cargue la geometría de los distritos para ver el mapa.</div>;
  }

  const colorFor = (feature: GeoFeature): string => {
    if (mode === "risk") {
      const level = String(feature.properties.risk_level || "UNKNOWN").toUpperCase();
      return RISK_COLORS[level] || RISK_COLORS.UNKNOWN;
    }
    const value = Number(feature.properties.incidence_per_100k ?? 0);
    const ratio = Math.min(1, value / scaleMax);
    const start = [232, 244, 236];
    const end = [140, 30, 26];
    const channels = start.map((channel, index) => Math.round(channel + (end[index] - channel) * ratio));
    return `rgb(${channels.join(",")})`;
  };

  const pathFor = (feature: GeoFeature): string => {
    const geometry = feature.geometry;
    if (!geometry) return "";
    const rings =
      geometry.type === "Polygon"
        ? (geometry.coordinates as number[][][])
        : (geometry.coordinates as number[][][][]).map((polygon) => polygon[0]);
    return rings
      .map((ring) =>
        ring
          .map(([lon, lat], index) => {
            const [x, y] = projection.project(lon, lat);
            return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
          })
          .join(" ") + " Z",
      )
      .join(" ");
  };

  // Área aproximada en píxeles para evitar etiquetas superpuestas en celdas pequeñas.
  const labelCandidates = useMemo(() => {
    const rows: { code: string; x: number; y: number; label: string; size: number }[] = [];
    (collection?.features ?? []).forEach((feature) => {
      const geometry = feature.geometry;
      if (!geometry) return;
      const ring = (geometry.type === "Polygon"
        ? (geometry.coordinates as number[][][])[0]
        : (geometry.coordinates as number[][][][])[0]?.[0]) as number[][] | undefined;
      if (!ring || ring.length < 3) return;
      const points = ring.map(([lon, lat]) => projection.project(lon, lat));
      let area = 0;
      for (let index = 0; index < points.length - 1; index += 1) {
        area += points[index][0] * points[index + 1][1] - points[index + 1][0] * points[index][1];
      }
      area = Math.abs(area / 2);
      const lat = Number(feature.properties.lat);
      const lon = Number(feature.properties.lon);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
      const [x, y] = projection.project(lon, lat);
      rows.push({ code: String(feature.properties.code), x, y, label: String(feature.properties.name ?? ""), size: area });
    });
    return rows.sort((a, b) => b.size - a.size);
  }, [collection, projection]);

  const visibleLabels = useMemo(() => {
    const accepted: { x: number; y: number; label: string }[] = [];
    labelCandidates.forEach((candidate) => {
      if (candidate.size < 900) return;
      const overlaps = accepted.some((item) => Math.hypot(item.x - candidate.x, item.y - candidate.y) < 34);
      if (!overlaps) accepted.push(candidate);
    });
    return accepted;
  }, [labelCandidates]);

  return (
    <div className="map-wrap" style={{ minHeight: height }}>
      <svg viewBox={`0 0 ${width} ${height}`} className="map-svg" role="img" aria-label="Mapa coroplético de distritos">
        <rect x={0} y={0} width={width} height={height} fill="#eef4f8" />
        {collection.features.map((feature) => {
          const code = String(feature.properties.code);
          const isSelected = selected === code;
          return (
            <path
              key={code}
              d={pathFor(feature)}
              fill={colorFor(feature)}
              stroke={isSelected ? "#10243b" : "#ffffff"}
              strokeWidth={isSelected ? 2.2 : 1}
              opacity={hover && hover.feature.properties.code !== code ? 0.82 : 1}
              style={{ cursor: onSelect ? "pointer" : "default", transition: "opacity 0.15s" }}
              onMouseMove={(event) => {
                const rect = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setHover({
                  feature,
                  x: event.clientX - rect.left,
                  y: event.clientY - rect.top,
                });
              }}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect?.(feature)}
            />
          );
        })}
        {visibleLabels.map((label) => (
          <text key={`label-${label.label}`} x={label.x} y={label.y} textAnchor="middle" className="map-label">
            {label.label.length > 16 ? `${label.label.slice(0, 15)}…` : label.label}
          </text>
        ))}
      </svg>
      {hover ? (
        <div className="map-tooltip" style={{ left: hover.x, top: hover.y }}>
          <strong>{String(hover.feature.properties.name)}</strong>
          {mode === "risk" ? (
            <div>
              Riesgo: {String(hover.feature.properties.risk_level || "sin dato")}
              {hover.feature.properties.predicted_cases != null
                ? ` · ${Number(hover.feature.properties.predicted_cases).toFixed(1)} casos`
                : ""}
            </div>
          ) : (
            <div>{Number(hover.feature.properties.cases ?? 0)} casos · {Number(hover.feature.properties.incidence_per_100k ?? 0).toFixed(1)} /100k</div>
          )}
          <div>Población: {Number(hover.feature.properties.population ?? 0).toLocaleString("es-PE")}</div>
        </div>
      ) : null}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", gap: 12, flexWrap: "wrap" }}>
        {mode === "risk" ? (
          <div className="legend">
            {Object.entries(RISK_COLORS)
              .filter(([level]) => level !== "UNKNOWN")
              .map(([level, color]) => (
                <span key={level} className="legend-item">
                  <span className="legend-swatch" style={{ background: color }} />
                  {level === "CRITICAL" ? "Crítico" : level === "HIGH" ? "Alto" : level === "MODERATE" ? "Medio" : "Bajo"}
                </span>
              ))}
          </div>
        ) : (
          <div className="legend">
            <span className="legend-item">Incidencia por 100 000 hab. (máx. {scaleMax.toFixed(1)})</span>
            <span className="legend-item">
              <span className="legend-swatch" style={{ background: "rgb(232,244,236)" }} /> 0
            </span>
            <span className="legend-item">
              <span className="legend-swatch" style={{ background: "rgb(140,30,26)" }} /> {scaleMax.toFixed(1)}
            </span>
          </div>
        )}
        <span className="hint">Teselación de distritos para visualización; no es cartografía oficial.</span>
      </div>
    </div>
  );
}
