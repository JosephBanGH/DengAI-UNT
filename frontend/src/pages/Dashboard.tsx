import { useEffect, useState } from "react";
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
  KpiCard,
  Loading,
  RiskBadge,
  StatusBadge,
  formatDate,
  formatInteger,
  formatNumber,
} from "../components/ui";
import { describeError } from "../context/ToastContext";

export function DashboardPage() {
  const [data, setData] = useState<DashboardOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
              <div className="legend">
                <span className="legend-item"><span className="legend-swatch" style={{ background: "#0b5d6e" }} />Casos</span>
                <span className="legend-item"><span className="legend-swatch" style={{ background: "#17a2b8" }} />Media 4 sem.</span>
              </div>
            }
          />
          <CardBody>
            <LineChart
              series={[
                { name: "Casos", points: series, color: "#0b5d6e" },
                { name: "Media móvil 4 sem.", points: rolling, color: "#17a2b8" },
              ]}
              height={250}
              labelEvery={12}
            />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Distribución de riesgo vigente" subtitle="Clasificación por incidencia pronosticada" />
          <CardBody>
            <Donut
              items={data.risk_distribution.map((item) => ({
                label: item.level === "CRITICAL" ? "Crítico" : item.level === "HIGH" ? "Alto" : item.level === "MODERATE" ? "Medio" : item.level === "LOW" ? "Bajo" : item.level,
                value: item.count,
                color:
                  item.level === "CRITICAL" ? "#b3261e" : item.level === "HIGH" ? "#e07a1f" : item.level === "MODERATE" ? "#e5b52c" : "#4c9f70",
              }))}
            />
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 16 }}>
        <Card>
          <CardHeader
            title="Distritos con mayor incidencia (12 semanas)"
            subtitle="Casos confirmados y tasa por 100 000 habitantes"
            actions={<Link className="btn btn-sm" to="/mapa">Ver mapa</Link>}
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
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="Distritos con mayor riesgo pronosticado"
            subtitle="Clasificación vigente de la última inferencia"
            actions={<Link className="btn btn-sm" to="/pronosticos">Pronósticos</Link>}
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
          </CardBody>
        </Card>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <Card>
          <CardHeader title="Factores climáticos" subtitle="Promedio de las últimas 8 semanas" />
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
              (sintéticas en esta demostración).
            </p>
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
