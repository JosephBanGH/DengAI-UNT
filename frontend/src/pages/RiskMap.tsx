import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { FeatureCollection } from "../api/types";
import { PageHeader } from "../components/Layout";
import { Choropleth } from "../components/Choropleth";
import { BarList } from "../components/charts";
import { Card, CardBody, CardHeader, EmptyState, ErrorAlert, Loading, RiskBadge, Tabs, formatDate, formatNumber } from "../components/ui";
import { describeError } from "../context/ToastContext";

type Mode = "risk" | "incidence";

export function RiskMapPage() {
  const [mode, setMode] = useState<Mode>("risk");
  const [risk, setRisk] = useState<FeatureCollection | null>(null);
  const [incidence, setIncidence] = useState<FeatureCollection | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [weeks, setWeeks] = useState(4);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      api.get<FeatureCollection>("/geo/risk-map"),
      api.get<FeatureCollection>("/geo/incidence-map", { weeks }),
    ])
      .then(([riskData, incidenceData]) => {
        if (cancelled) return;
        setRisk(riskData);
        setIncidence(incidenceData);
      })
      .catch((caught) => {
        if (!cancelled) setError(describeError(caught));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [weeks]);

  const collection = mode === "risk" ? risk : incidence;
  const selectedFeature = collection?.features.find((feature) => feature.properties.code === selected) || null;

  const ranking = (collection?.features ?? [])
    .map((feature) => ({
      label: String(feature.properties.name),
      code: String(feature.properties.code),
      value:
        mode === "risk"
          ? Number(feature.properties.incidence_per_100k ?? feature.properties.risk_score ?? 0)
          : Number(feature.properties.incidence_per_100k ?? 0),
      hint:
        mode === "risk"
          ? `${String(feature.properties.risk_level || "sin dato")} · ${formatNumber(Number(feature.properties.predicted_cases ?? 0), 1)} casos pron.`
          : `${feature.properties.cases ?? 0} casos en ${weeks} semanas`,
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  return (
    <>
      <PageHeader
        title="Mapas y análisis territorial"
        description="Distribución espacial del riesgo pronosticado y de la incidencia histórica por distrito. La geometría es una teselación de Voronoi alrededor de los centroides mientras no se cargue la cartografía oficial."
        actions={
          <div className="field" style={{ marginBottom: 0 }}>
            <label htmlFor="weeks">Ventana histórica</label>
            <select id="weeks" className="select" value={weeks} onChange={(event) => setWeeks(Number(event.target.value))}>
              <option value={4}>4 semanas</option>
              <option value={8}>8 semanas</option>
              <option value={12}>12 semanas</option>
              <option value={26}>26 semanas</option>
            </select>
          </div>
        }
      />

      <Tabs
        tabs={[
          { key: "risk", label: "Riesgo pronosticado" },
          { key: "incidence", label: "Incidencia histórica" },
        ]}
        active={mode}
        onChange={(key) => setMode(key as Mode)}
      />

      {error ? <ErrorAlert message={error} /> : null}
      {loading ? <Loading label="Cargando cartografía…" /> : null}

      {!loading && !error ? (
        <div className="grid grid-main">
          <Card>
            <CardHeader
              title={mode === "risk" ? "Mapa de calor de riesgo" : "Distribución histórica de casos"}
              subtitle={
                mode === "risk"
                  ? `Pronóstico generado ${risk?.metadata?.generated_at ? formatDate(risk.metadata.generated_at, true) : "sin pronóstico registrado"}`
                  : `Acumulado de las últimas ${weeks} semanas`
              }
            />
            <CardBody>
              <Choropleth
                collection={collection}
                mode={mode}
                selected={selected}
                onSelect={(feature) => setSelected(String(feature.properties.code))}
              />
            </CardBody>
          </Card>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <Card>
              <CardHeader title="Ranking territorial" subtitle={mode === "risk" ? "Por incidencia pronosticada" : "Por incidencia acumulada"} />
              <CardBody>
                <BarList items={ranking} valueFormatter={(value) => value.toFixed(1)} />
              </CardBody>
            </Card>

            <Card>
              <CardHeader title="Detalle del distrito" subtitle={selectedFeature ? String(selectedFeature.properties.name) : "Seleccione un distrito"} />
              <CardBody>
                {selectedFeature ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Provincia</span>
                      <strong className="small">{String(selectedFeature.properties.province)}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Población</span>
                      <strong className="small">{Number(selectedFeature.properties.population ?? 0).toLocaleString("es-PE")}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Altitud</span>
                      <strong className="small">{formatNumber(Number(selectedFeature.properties.altitude_m ?? 0), 0)} m</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Nivel de riesgo</span>
                      <RiskBadge level={String(selectedFeature.properties.risk_level || "UNKNOWN")} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Casos pronosticados</span>
                      <strong className="small">{formatNumber(Number(selectedFeature.properties.predicted_cases ?? 0), 1)}</strong>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span className="muted small">Incidencia / 100k</span>
                      <strong className="small">{formatNumber(Number(selectedFeature.properties.incidence_per_100k ?? 0), 1)}</strong>
                    </div>
                    {selectedFeature.properties.rationale ? (
                      <p className="hint" style={{ marginTop: 6 }}>{String(selectedFeature.properties.rationale)}</p>
                    ) : null}
                  </div>
                ) : (
                  <EmptyState title="Sin selección" description="Haga clic sobre un distrito del mapa para ver su detalle." />
                )}
              </CardBody>
            </Card>
          </div>
        </div>
      ) : null}
    </>
  );
}
