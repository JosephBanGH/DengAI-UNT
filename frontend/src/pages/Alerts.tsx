import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AlertItem } from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  Loading,
  Modal,
  RiskBadge,
  StatusBadge,
  formatDate,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function AlertsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [items, setItems] = useState<AlertItem[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState("OPEN");
  const [level, setLevel] = useState("");
  const [generating, setGenerating] = useState(false);
  const [detail, setDetail] = useState<AlertItem | null>(null);
  const [notes, setNotes] = useState("");

  const load = () => {
    setLoading(true);
    Promise.all([
      api.get<{ items: AlertItem[] }>("/alerts", { status: status || undefined, level: level || undefined, size: 100 }),
      api.get<any>("/alerts/summary"),
    ])
      .then(([list, stats]) => {
        setItems(list.items);
        setSummary(stats);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, [status, level]);

  const generate = async () => {
    setGenerating(true);
    try {
      const result = await api.post<{ created: number }>("/alerts/generate", { horizons: [1, 2], min_level: "MODERATE" });
      toast.push(`${result.created} alertas generadas o vigentes.`, "success");
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    } finally {
      setGenerating(false);
    }
  };

  const update = async (alert: AlertItem, action: string) => {
    try {
      await api.post(`/alerts/${alert.id}`, { action, notes });
      toast.push(`Alerta ${action === "acknowledge" ? "reconocida" : action === "close" ? "cerrada" : "reabierta"}.`, "success");
      setDetail(null);
      setNotes("");
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  return (
    <>
      <PageHeader
        title="Sistema de alertas tempranas"
        description="Alertas generadas a partir de los pronósticos vigentes. Los niveles se definen por incidencia semanal por 100 000 habitantes con umbrales configurables."
        actions={
          can("alerts:write") ? (
            <button className="btn btn-primary" onClick={generate} disabled={generating}>
              {generating ? "Evaluando pronósticos…" : "Generar alertas"}
            </button>
          ) : undefined
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      {summary ? (
        <div className="grid grid-kpi" style={{ marginBottom: 14 }}>
          <div className="card kpi">
            <span className="kpi-label">Total histórico</span>
            <span className="kpi-value">{summary.total}</span>
          </div>
          {Object.entries(summary.by_status || {}).map(([key, value]) => (
            <div className="card kpi" key={key}>
              <span className="kpi-label">{key}</span>
              <span className="kpi-value">{value as number}</span>
            </div>
          ))}
          {Object.entries(summary.by_level || {}).map(([key, value]) => (
            <div className="card kpi" key={key}>
              <span className="kpi-label">Nivel {key}</span>
              <span className="kpi-value">{value as number}</span>
            </div>
          ))}
        </div>
      ) : null}

      <Card>
        <CardHeader
          title="Alertas"
          subtitle="Seguimiento, reconocimiento y cierre"
          actions={
            <>
              <select className="select" style={{ width: 160 }} value={status} onChange={(event) => setStatus(event.target.value)}>
                <option value="">Todos los estados</option>
                <option value="OPEN">Abiertas</option>
                <option value="ACKNOWLEDGED">Reconocidas</option>
                <option value="CLOSED">Cerradas</option>
              </select>
              <select className="select" style={{ width: 150 }} value={level} onChange={(event) => setLevel(event.target.value)}>
                <option value="">Todos los niveles</option>
                <option value="CRITICAL">Crítico</option>
                <option value="HIGH">Alto</option>
                <option value="MODERATE">Medio</option>
                <option value="LOW">Bajo</option>
              </select>
            </>
          }
        />
        <CardBody>
          {loading ? (
            <Loading />
          ) : items.length ? (
            <DataTable
              rows={items}
              rowKey={(row) => row.id}
              columns={[
                { key: "district_name", header: "Distrito" },
                { key: "week", header: "Semana", render: (row) => `${row.epi_year}-W${String(row.epi_week).padStart(2, "0")}` },
                { key: "horizon_weeks", header: "Horizonte", numeric: true, render: (row) => `h=${row.horizon_weeks}` },
                { key: "risk_level", header: "Nivel", render: (row) => <RiskBadge level={row.risk_level} /> },
                { key: "predicted_cases", header: "Casos pron.", numeric: true, render: (row) => formatNumber(row.predicted_cases, 1) },
                {
                  key: "interval",
                  header: "IC 90%",
                  numeric: true,
                  render: (row) => `${formatNumber(row.lower_bound, 1)} – ${formatNumber(row.upper_bound, 1)}`,
                },
                { key: "status", header: "Estado", render: (row) => <StatusBadge status={row.status} /> },
                { key: "generated_at", header: "Generada", render: (row) => formatDate(row.generated_at, true) },
                {
                  key: "actions",
                  header: "",
                  render: (row) => (
                    <button className="btn btn-sm btn-ghost" onClick={() => { setDetail(row); setNotes(row.notes || ""); }}>
                      Gestionar
                    </button>
                  ),
                },
              ]}
            />
          ) : (
            <EmptyState
              title="Sin alertas para el filtro seleccionado"
              description="Genere alertas a partir de los pronósticos vigentes o ajuste los filtros."
            />
          )}
        </CardBody>
      </Card>

      {detail ? (
        <Modal
          title={`Alerta · ${detail.district_name}`}
          onClose={() => setDetail(null)}
          footer={
            can("alerts:write") ? (
              <>
                <button className="btn" onClick={() => update(detail, "reopen")}>Reabrir</button>
                <button className="btn" onClick={() => update(detail, "acknowledge")}>Reconocer</button>
                <button className="btn btn-danger" onClick={() => update(detail, "close")}>Cerrar</button>
              </>
            ) : undefined
          }
        >
          <div className="grid grid-2" style={{ marginBottom: 12 }}>
            <div><div className="kpi-label">Nivel</div><RiskBadge level={detail.risk_level} /></div>
            <div><div className="kpi-label">Estado</div><StatusBadge status={detail.status} /></div>
            <div><div className="kpi-label">Casos pronosticados</div><strong>{formatNumber(detail.predicted_cases, 1)}</strong></div>
            <div><div className="kpi-label">Incidencia /100k</div><strong>{formatNumber(detail.incidence_per_100k, 1)}</strong></div>
          </div>
          <p className="small">{detail.rationale}</p>
          <p className="hint">
            Generada {formatDate(detail.generated_at, true)} · Modelo asociado #{detail.horizon_weeks} ·{" "}
            {detail.is_synthetic_data ? "Datos sintéticos" : "Datos de fuente externa"}
          </p>
          {detail.acknowledged_by ? (
            <p className="small muted">Reconocida por {detail.acknowledged_by} el {formatDate(detail.acknowledged_at, true)}</p>
          ) : null}
          <div className="field" style={{ marginTop: 12 }}>
            <label>Notas de seguimiento</label>
            <textarea className="textarea" value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Registre acciones de verificación o respuesta" />
          </div>
        </Modal>
      ) : null}
    </>
  );
}
