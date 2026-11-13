import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { DashboardOverview } from "../api/types";
import { PageHeader } from "../components/Layout";
import { LineChart, BarList, Donut } from "../components/charts";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  InterpretabilityCard,
  KpiCard,
  Loading,
  RiskBadge,
  StatusBadge,
  formatDate,
  formatInteger,
  formatNumber,
} from "../components/ui";
import { describeError } from "../context/ToastContext";
import { downloadCsv, downloadSvg, downloadSvgAsPng } from "../utils/export";

export function DashboardPage() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const lineSvgRef = useRef<SVGSVGElement>(null);
  const donutSvgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .get<DashboardOverview>("/dashboard/overview", { weeks: 104 })
      .then((payload) => {
        if (!cancelled) setData(payload);
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
  }, []);

  if (loading) return <Loading label="Calculando indicadores epidemiológicos…" />;
  if (error) return <ErrorAlert message={error} />;
  if (!data) return <EmptyState title="Sin datos" />;

  const series = data.weekly_series.map((point) => ({ x: point.week, y: point.cases }));
  const rolling = data.weekly_series.map((point, index, array) => {
    const window = array.slice(Math.max(0, index - 3), index + 1);
    return { x: point.week, y: Number((window.reduce((sum, item) => sum + item.cases, 0) / window.length).toFixed(1)) };
  });

  return (
    <>
      <PageHeader
        title="Dashboard ejecutivo"
        description="Indicadores epidemiológicos y operacionales de la plataforma. Los filtros territoriales se aplican en los módulos de mapas y datos."
        actions={
          <span className="chip">
            Última actualización de datos: {data.data_freshness ? formatDate(data.data_freshness) : "sin datos"}
          </span>
        }
      />

      <div className="grid grid-kpi" style={{ marginBottom: 16 }}>
        {data.kpis.map((kpi) => (
          <KpiCard
            key={kpi.key}
            label={kpi.label}
            value={kpi.key === "synthetic_share" ? formatNumber(kpi.value, 1) : formatInteger(kpi.value)}
            unit={kpi.unit}
            changePct={kpi.change_pct}
            trend={kpi.trend}
            hint={kpi.hint}
          />
        ))}
      </div>

      <div className="grid grid-main" style={{ marginBottom: 16 }}>
        <Card>
          <CardHeader
            title="Evolución temporal de casos confirmados"
            subtitle="Serie semanal (2 años) y media móvil de 4 semanas"
            actions={
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <div className="legend" style={{ marginRight: 6 }}>
                  <span className="legend-item"><span className="legend-swatch" style={{ background: "#0b5d6e" }} />Casos</span>
                  <span className="legend-item"><span className="legend-swatch" style={{ background: "#17a2b8" }} />Media 4 sem.</span>
                </div>
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "dashboard_serie_temporal_casos.csv",
                      ["semana", "casos_confirmados", "media_movil_4sem"],
                      series.map((pt, i) => [pt.x, pt.y, rolling[i]?.y ?? ""])
                    )
                  }
                  title="Descargar datos en CSV"
                >
                  CSV
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => lineSvgRef.current && downloadSvg(lineSvgRef.current, "dashboard_serie_temporal_casos.svg")}
                  title="Descargar gráfico en formato vectorial SVG"
                >
                  SVG
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => lineSvgRef.current && downloadSvgAsPng(lineSvgRef.current, "dashboard_serie_temporal_casos.png", 4)}
                  title="Descargar gráfico en alta resolución PNG (600 DPI)"
                >
                  PNG (600 DPI)
                </button>
              </div>
            }
          />
          <CardBody>
            <LineChart
              svgRef={lineSvgRef}
              series={[
                { name: "Casos", points: series, color: "#0b5d6e" },
                { name: "Media móvil 4 sem.", points: rolling, color: "#17a2b8" },
              ]}
              height={250}
              labelEvery={12}
            />
            <InterpretabilityCard
              title="Interpretabilidad Epidemiológica: Dinámica Temporal Decenal"
              finding="El panel oficial registró 65 305 casos de dengue entre 2015 y 2024, con el 87.1 % concentrado en el bienio 2023–2024 tras El Niño Costero y el ciclón Yaku, alcanzando un pico de 3 211 casos en 2024-W15."
              mechanism="El retardo de 2 a 6 semanas posterior a las precipitaciones y alza térmica (>24 °C) aceleró la tasa de oviposición y acortó el ciclo gonotrófico de Aedes aegypti a menos de 8 días."
              paperSnippet="La evolución temporal evidenció una transición epidemiológica bifásica: una fase interepidémica con 75,1 % de semanas en cero (2015–2022), seguida de una emergencia hiperendémica explosiva en 2023–2024 desencadenada por anomalías bioclimáticas de El Niño Costero."
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Distribución de riesgo vigente"
            subtitle="Clasificación por incidencia pronosticada"
            actions={
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "dashboard_distribucion_riesgo.csv",
                      ["estrato_riesgo", "cantidad_distritos", "porcentaje"],
                      data.risk_distribution.map((item) => [
                        item.level,
                        item.count,
                        ((item.count / (data.risk_distribution.reduce((s, r) => s + r.count, 0) || 1)) * 100).toFixed(1) + "%",
                      ])
                    )
                  }
                  title="Descargar datos de riesgo en CSV"
                >
                  CSV
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => donutSvgRef.current && downloadSvg(donutSvgRef.current, "dashboard_distribucion_riesgo.svg")}
                  title="Descargar dona en SVG"
                >
                  SVG
                </button>
                <button
                  className="btn btn-sm"
                  onClick={() => donutSvgRef.current && downloadSvgAsPng(donutSvgRef.current, "dashboard_distribucion_riesgo.png", 4)}
                  title="Descargar dona en PNG (600 DPI)"
                >
                  PNG (600 DPI)
                </button>
              </div>
            }
          />
          <CardBody>
            <Donut
              svgRef={donutSvgRef}
              items={data.risk_distribution.map((item) => ({
                label: item.level === "CRITICAL" ? "Crítico" : item.level === "HIGH" ? "Alto" : item.level === "MODERATE" ? "Medio" : item.level === "LOW" ? "Bajo" : item.level,
                value: item.count,
                color:
                  item.level === "CRITICAL" ? "#b3261e" : item.level === "HIGH" ? "#e07a1f" : item.level === "MODERATE" ? "#e5b52c" : "#4c9f70",
              }))}
            />
            <InterpretabilityCard
              title="Interpretabilidad Epidemiológica: Estratificación de Riesgo"
              finding="Estratificación de los 13 distritos de La Libertad según umbrales operacionales de incidencia pronosticada por 100 000 habitantes."
              mechanism="La categorización en cuatro estratos (Bajo <5, Medio 5–20, Alto 20–50, Crítico ≥50) guía el despliegue anticipado de brigadas de fumigación espacial y control focal de larvas."
              paperSnippet="La estratificación operacional en cuatro niveles de riesgo facilita priorizar la asignación territorial de recursos sanitarios con anterioridad a la saturación hospitalaria."
            />
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <Card>
          <CardHeader
            title="Distritos con mayor incidencia (12 semanas)"
            subtitle="Casos confirmados y tasa por 100 000 habitantes"
            actions={
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "dashboard_distritos_mayor_incidencia.csv",
                      ["distrito", "incidencia_por_100k", "casos_12_semanas", "poblacion"],
                      data.districts_top_incidence.map((item) => [
                        item.district_name,
                        item.incidence_per_100k?.toFixed(2) ?? 0,
                        item.cases_12w,
                        item.population,
                      ])
                    )
                  }
                  title="Descargar datos en CSV"
                >
                  CSV
                </button>
                <Link className="btn btn-sm" to="/mapa">Ver mapa</Link>
              </div>
            }
          />
          <CardBody>
            <BarList
              items={data.districts_top_incidence.slice(0, 8).map((item) => ({
                label: item.district_name,
                value: item.incidence_per_100k ?? 0,
                hint: `${formatInteger(item.cases_12w)} casos · ${formatInteger(item.population)} hab.`,
              }))}
              valueFormatter={(value) => value.toFixed(1)}
            />
            <InterpretabilityCard
              title="Interpretabilidad Espacial: Heterogeneidad Distrital"
              finding="Los mayores focos de incidencia acumulada se concentran en el valle de Chepén (933,3) y la conurbación de Trujillo (El Porvenir, La Esperanza, Florencia de Mora)."
              mechanism="La alta densidad urbana combinada con recipientes domésticos desprotegidos y microclima cálido sostienen la reproducción continua de Aedes aegypti."
              paperSnippet="El análisis espacial demostró una heterogeneidad territorial marcada, con mayor concentración endémica en los corredores periurbanos y valles irrigados de la costa liberteña."
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Distritos con mayor riesgo pronosticado"
            subtitle="Clasificación vigente de la última inferencia"
            actions={
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "dashboard_distritos_mayor_riesgo_pronosticado.csv",
                      ["codigo_distrito", "distrito", "nivel_riesgo", "casos_pronosticados", "incidencia_pronosticada_por_100k"],
                      data.districts_top_risk.map((item) => [
                        item.district_code,
                        item.district_name,
                        item.level,
                        item.predicted_cases?.toFixed(2) ?? 0,
                        item.incidence_per_100k?.toFixed(2) ?? 0,
                      ])
                    )
                  }
                  title="Descargar datos de pronóstico en CSV"
                >
                  CSV
                </button>
                <Link className="btn btn-sm" to="/pronosticos">Pronósticos</Link>
              </div>
            }
          />
          <CardBody>
            {data.districts_top_risk.length ? (
              <DataTable
                rows={data.districts_top_risk.slice(0, 8)}
                rowKey={(row) => row.district_code}
                columns={[
                  { key: "district_name", header: "Distrito" },
                  { key: "level", header: "Riesgo", render: (row) => <RiskBadge level={row.level} /> },
                  {
                    key: "predicted_cases",
                    header: "Casos pron.",
                    numeric: true,
                    render: (row) => formatNumber(row.predicted_cases, 1),
                  },
                  {
                    key: "incidence_per_100k",
                    header: "Incidencia",
                    numeric: true,
                    render: (row) => formatNumber(row.incidence_per_100k, 1),
                  },
                ]}
              />
            ) : (
              <EmptyState title="Sin clasificación de riesgo" description="Genere un pronóstico y ejecute la clasificación de riesgo." />
            )}
            <InterpretabilityCard
              title="Explicabilidad del Pronóstico: Ventana de Anticipación"
              finding="Inferencia generada a horizonte h=1 semana mediante el modelo ConvLSTM validado en producción sobre el panel distrital."
              mechanism="La predicción anticipada a 7 días permite ejecutar fumigación focalizada y abatización larvaria antes de que ocurra la transmisión comunitaria secundaria."
              paperSnippet="La inferencia predictiva espacio-temporal proporciona una ventana de alerta temprana accionable de 7 días, superando los desfases de la notificación pasiva habitual."
            />
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Card>
          <CardHeader
            title="Factores climáticos"
            subtitle="Promedio de las últimas 8 semanas"
            actions={
              <button
                className="btn btn-sm"
                onClick={() =>
                  downloadCsv(
                    "dashboard_factores_climaticos.csv",
                    ["variable", "valor_promedio_8sem", "unidad", "variacion_porcentual_vs_historico"],
                    data.climate_summary.map((item) => [
                      item.variable,
                      item.value != null ? item.value.toFixed(2) : "—",
                      item.unit,
                      item.change_pct !== null ? `${item.change_pct.toFixed(1)}%` : "—",
                    ])
                  )
                }
                title="Descargar datos climáticos en CSV"
              >
                CSV
              </button>
            }
          />
          <CardBody>
            {data.climate_summary.map((item) => (
              <div key={item.variable} style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                <span className="small">{item.variable}</span>
                <span>
                  <strong>{formatNumber(item.value, 1)}</strong> <span className="muted small">{item.unit}</span>
                  {item.change_pct !== null ? (
                    <span className={`kpi-delta ${item.change_pct > 0 ? "up" : "down"}`} style={{ marginLeft: 8 }}>
                      {item.change_pct > 0 ? "▲" : "▼"} {Math.abs(item.change_pct).toFixed(1)}%
                    </span>
                  ) : null}
                </span>
              </div>
            ))}
            <p className="hint">
              Comparación con el periodo histórico previo. Los valores provienen de las observaciones almacenadas
              ({data.is_synthetic_data ? "sintéticas de demostración" : "MINSA/CDC Perú y NASA POWER 2015–2024"}).
            </p>
            <InterpretabilityCard
              title="Explicabilidad Bioclimática: Forzamiento Exógeno"
              finding="Monitoreo de temperatura media superficial, humedad relativa y precipitaciones de reanálisis satelital NASA POWER."
              mechanism="Temperaturas superiores a 24 °C aceleran exponencialmente la replicación viral en la glándula salival del mosquito y reducen el intervalo entre picaduras a menos de 48 horas."
              paperSnippet="Los predictores bioclimáticos de reanálisis satelital actuaron como precursores dinámicos determinantes, capturando el acoplamiento biofísico causal de la transmisión del dengue."
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Estado de integraciones" subtitle="Fuentes de datos configuradas" />
          <CardBody>
            {data.integration_status.slice(0, 7).map((item) => (
              <div key={item.code} style={{ display: "flex", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
                <span className="small truncate" title={item.name}>{item.name}</span>
                <StatusBadge status={item.is_synthetic ? "SYNTHETIC" : item.status.split("_")[0]} />
              </div>
            ))}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Rendimiento del modelo en producción" subtitle="Métricas del conjunto de prueba" />
          <CardBody>
            {data.model_performance.length ? (
              data.model_performance.map((model) => (
                <div key={`${model.algorithm}-${model.version}`} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <strong className="small">{model.algorithm}</strong>
                    <span className="chip">h={model.horizon} · {model.version}</span>
                  </div>
                  <div className="small muted">
                    MAE {formatNumber(model.MAE, 2)} · RMSE {formatNumber(model.RMSE, 2)} · AUC-PR{" "}
                    {formatNumber(model.AUC_PR, 3)} · Cobertura {formatNumber((model.coverage ?? 0) * 100, 1)}%
                  </div>
                </div>
              ))
            ) : (
              <EmptyState
                title="Sin modelos en producción"
                description="Entrene un experimento en el Laboratorio de IA y promueva la versión resultante."
              />
            )}
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-2">
        <Card>
          <CardHeader title="Alertas activas" subtitle="Estado de las alertas tempranas" actions={<Link className="btn btn-sm" to="/alertas">Gestionar</Link>} />
          <CardBody>
            <div style={{ display: "flex", gap: 18, alignItems: "center", flexWrap: "wrap" }}>
              <div>
                <div className="kpi-label">Total activas</div>
                <div className="kpi-value">{data.active_alerts}</div>
              </div>
              <div className="legend" style={{ flexDirection: "column", alignItems: "flex-start" }}>
                {data.alerts_by_level.map((item) => (
                  <span key={item.level} className="legend-item">
                    <RiskBadge level={item.level} /> <strong>{item.count}</strong>
                  </span>
                ))}
                {!data.alerts_by_level.length ? <span className="muted small">Sin alertas abiertas.</span> : null}
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Operación de la plataforma" subtitle="Actividad registrada" />
          <CardBody>
            <div className="grid grid-3">
              {Object.entries(data.operations || {}).map(([key, value]) => (
                <div key={key}>
                  <div className="kpi-label">{labelForKey(key)}</div>
                  <div className="kpi-value" style={{ fontSize: 20 }}>{value}</div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      </div>
    </>
  );
}

function labelForKey(key: string): string {
  const labels: Record<string, string> = {
    experiments: "Experimentos",
    reports: "Reportes",
    workflow_runs: "Ejecuciones WF",
    active_sessions: "Sesiones activas",
  };
  return labels[key] || key;
}
