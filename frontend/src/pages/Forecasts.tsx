import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Prediction } from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  InterpretabilityCard,
  Loading,
  RiskBadge,
  formatDate,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";
import { downloadCsv } from "../utils/export";

interface ForecastResult {
  run_key: string;
  model_version: string;
  algorithm: string;
  generated_at: string;
  is_synthetic_data: boolean;
  predictions: Prediction[];
  disclaimer?: string;
}

const HORIZONS = [1, 2, 4, 8];

export function ForecastsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [items, setItems] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [horizons, setHorizons] = useState<number[]>([1]);
  const [lastRun, setLastRun] = useState<ForecastResult | null>(null);
  const [filter, setFilter] = useState<number | "">("");
  const [explain, setExplain] = useState<{ district: string; data: any } | null>(null);

  const load = () => {
    setLoading(true);
    api
      .get<{ items: Prediction[] }>("/forecasts/latest", { horizon: filter || undefined })
      .then((data) => setItems(data.items))
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, [filter]);

  const generate = async () => {
    setRunning(true);
    setError(null);
    try {
      const result = await api.post<ForecastResult>("/forecasts/generate", { horizons });
      setLastRun(result);
      toast.push(`Pronósticos generados para ${result.predictions.length} registros distrito-semana.`, "success");
      load();
    } catch (caught) {
      setError(describeError(caught));
      toast.push(describeError(caught), "error");
    } finally {
      setRunning(false);
    }
  };

  const openExplanation = async (districtCode: string) => {
    try {
      const data = await api.get<any>(`/ml/explain/${districtCode}`, { horizon: filter || 1 });
      setExplain({ district: districtCode, data });
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  return (
    <>
      <PageHeader
        title="Pronósticos multihorizonte"
        description="Inferencia con el modelo en producción. Los intervalos se calibran con los residuales del conjunto de validación, nunca con el de prueba."
        actions={
          can("predictions:run") ? (
            <>
              <select
                className="select"
                style={{ width: 200 }}
                multiple
                value={horizons.map(String)}
                onChange={(event) => {
                  const selected = Array.from(event.target.selectedOptions).map((option) => Number(option.value));
                  setHorizons(selected.length ? selected : [1]);
                }}
                title="Mantenga Ctrl para seleccionar varios horizontes"
              >
                {HORIZONS.map((horizon) => (
                  <option key={horizon} value={horizon}>Horizonte {horizon} semana(s)</option>
                ))}
              </select>
              <button className="btn btn-primary" onClick={generate} disabled={running}>
                {running ? "Generando…" : "Generar pronóstico"}
              </button>
            </>
          ) : undefined
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      {lastRun ? (
        <Card className="card-pad" style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center" }}>
            <div>
              <div className="kpi-label">Ejecución</div>
              <div className="mono">{lastRun.run_key}</div>
            </div>
            <div>
              <div className="kpi-label">Modelo</div>
              <div className="small">{lastRun.model_version}</div>
            </div>
            <div>
              <div className="kpi-label">Registros</div>
              <div className="small">{lastRun.predictions.length}</div>
            </div>
            <div style={{ marginLeft: "auto" }} className="hint">
              {lastRun.disclaimer}
            </div>
          </div>
        </Card>
      ) : null}

      <Card>
        <CardHeader
          title="Pronósticos vigentes"
          subtitle="Ordenados por horizonte y casos esperados"
          actions={
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {items.length ? (
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "pronosticos_vigentes_dengai.csv",
                      ["codigo_distrito", "distrito", "semana_objetivo", "horizonte", "casos_pronosticados", "limite_inferior_90", "limite_superior_90", "incidencia_por_100k", "nivel_riesgo"],
                      items.map((r) => [
                        r.district_code,
                        r.district_name,
                        `${r.target_year}-W${String(r.target_week).padStart(2, "0")}`,
                        r.horizon_weeks,
                        r.predicted_cases?.toFixed(2) ?? 0,
                        r.lower_bound?.toFixed(2) ?? 0,
                        r.upper_bound?.toFixed(2) ?? 0,
                        r.incidence_per_100k?.toFixed(2) ?? 0,
                        r.risk_level,
                      ])
                    )
                  }
                  title="Descargar pronósticos en CSV"
                >
                  Descargar CSV (Pronósticos)
                </button>
              ) : null}
              <select className="select" style={{ width: 170 }} value={filter} onChange={(event) => setFilter(event.target.value ? Number(event.target.value) : "")}>
                <option value="">Todos los horizontes</option>
                {HORIZONS.map((horizon) => (
                  <option key={horizon} value={horizon}>h = {horizon}</option>
                ))}
              </select>
            </div>
          }
        />
        <CardBody>
          {loading ? (
            <Loading />
          ) : items.length ? (
            <>
              <DataTable
                rows={items}
                rowKey={(row) => `${row.district_code}-${row.horizon_weeks}`}
                columns={[
                  { key: "district_name", header: "Distrito" },
                  {
                    key: "target",
                    header: "Semana objetivo",
                    render: (row) => `${row.target_year}-W${String(row.target_week).padStart(2, "0")}`,
                  },
                  { key: "horizon_weeks", header: "Horizonte", numeric: true, render: (row) => `h=${row.horizon_weeks}` },
                  { key: "predicted_cases", header: "Casos pron.", numeric: true, render: (row) => formatNumber(row.predicted_cases, 1) },
                  {
                    key: "interval",
                    header: "IC 90%",
                    numeric: true,
                    render: (row) => `${formatNumber(row.lower_bound, 1)} – ${formatNumber(row.upper_bound, 1)}`,
                  },
                  { key: "incidence_per_100k", header: "Incidencia /100k", numeric: true, render: (row) => formatNumber(row.incidence_per_100k, 1) },
                  { key: "risk_level", header: "Riesgo", render: (row) => <RiskBadge level={row.risk_level} /> },
                  {
                    key: "explain",
                    header: "",
                    render: (row) =>
                      can("explain:read") ? (
                        <button className="btn btn-sm btn-ghost" onClick={() => openExplanation(row.district_code)}>
                          Explicar
                        </button>
                      ) : null,
                  },
                ]}
              />
              <InterpretabilityCard
                title="Explicabilidad del Pronóstico Distrital e Incertidumbre"
                finding="Pronósticos cuantitativos acompañados de intervalos de incertidumbre al 90 % calculados mediante distribución condicional empírica."
                mechanism="Permite cuantificar no solo el valor esperado sino la cota superior en escenarios de aceleración epidémica imprevista por lluvias."
                paperSnippet="La estimación de intervalos de predicción al 90 % complementa el pronóstico puntual, otorgando un marco riguroso de gestión de riesgo frente a la variabilidad hidroclimática costera."
              />
            </>
          ) : (
            <EmptyState
              title="Sin pronósticos registrados"
              description="Genere una inferencia con el modelo en producción o entrene y promueva una versión en el Laboratorio de IA."
            />
          )}
        </CardBody>
      </Card>

      {explain ? (
        <div className="modal-backdrop" onClick={() => setExplain(null)} role="presentation">
          <div className="modal" style={{ maxWidth: 680 }} onClick={(event) => event.stopPropagation()}>
            <header className="modal-header">
              <h2 style={{ fontSize: 16 }}>Explicación de la predicción · {explain.data.district_name}</h2>
              <button className="btn btn-ghost btn-sm" style={{ marginLeft: "auto" }} onClick={() => setExplain(null)}>
                Cerrar
              </button>
            </header>
            <div className="modal-body">
              <p className="small muted" style={{ marginBottom: 12 }}>
                Método: {explain.data.method} · Modelo {explain.data.algorithm} {explain.data.model_version} · Predicción{" "}
                {formatNumber(explain.data.prediction, 2)} casos
              </p>
              <table className="data">
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th className="numeric">Contribución</th>
                    <th>Efecto</th>
                    <th className="numeric">Valor</th>
                  </tr>
                </thead>
                <tbody>
                  {explain.data.factors.map((factor: any) => (
                    <tr key={factor.feature}>
                      <td>{factor.label}</td>
                      <td className="numeric">{formatNumber(factor.contribution, 3)}</td>
                      <td>
                        <span className={`badge ${factor.direction === "aumenta" ? "badge-danger" : "badge-ok"}`}>
                          {factor.direction}
                        </span>
                      </td>
                      <td className="numeric">{factor.value === null ? "—" : formatNumber(factor.value, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="small" style={{ marginTop: 12 }}>{explain.data.narrative}</p>
              <p className="hint">{explain.data.caveat}</p>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
