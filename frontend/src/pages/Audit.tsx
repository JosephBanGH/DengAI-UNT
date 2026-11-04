import { useEffect, useState } from "react";
import { api, downloadFile } from "../api/client";
import type { AuditEvent, Page } from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  Loading,
  StatusBadge,
  formatDate,
} from "../components/ui";
import { BarList } from "../components/charts";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function AuditPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [verification, setVerification] = useState<any>(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [eventType, setEventType] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const size = 25;

  const load = () => {
    setLoading(true);
    Promise.all([
      api.get<Page<AuditEvent>>("/audit", { page, size, event_type: eventType || undefined, search: search || undefined }),
      api.get<any>("/audit/stats"),
      api.get<any>("/audit/verify"),
    ])
      .then(([pageData, statsData, verifyData]) => {
        setEvents(pageData.items);
        setTotal(pageData.total);
        setStats(statsData);
        setVerification(verifyData);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, [page, eventType]);

  const pages = Math.max(1, Math.ceil(total / size));

  return (
    <>
      <PageHeader
        title="Bitácora y auditoría del sistema"
        description="Registro encadenado de eventos: cada fila incluye el hash del evento anterior, de modo que cualquier alteración o eliminación rompe la cadena y queda en evidencia."
        actions={
          can("audit:export") ? (
            <button
              className="btn"
              onClick={() =>
                downloadFile("/audit/export", "bitacora_auditoria.csv").catch((caught) =>
                  toast.push(describeError(caught), "error"),
                )
              }
            >
              Exportar CSV
            </button>
          ) : undefined
        }
      />

      {verification ? (
        <div className={`alert ${verification.valid ? "alert-ok" : "alert-error"}`}>
          <strong>{verification.valid ? "Cadena de integridad verificada." : "Se detectó una alteración."}</strong>{" "}
          {verification.message}
        </div>
      ) : null}

      {error ? <ErrorAlert message={error} /> : null}

      <div className="grid grid-main" style={{ marginBottom: 14 }}>
        <Card>
          <CardHeader title="Eventos" subtitle={`${total} registros`} />
          <CardBody>
            <div className="toolbar">
              <div className="field">
                <label>Tipo de evento</label>
                <select className="select" style={{ width: 190 }} value={eventType} onChange={(event) => { setPage(1); setEventType(event.target.value); }}>
                  <option value="">Todos</option>
                  {["LOGIN", "LOGOUT", "MODULE_ENTER", "MODULE_EXIT", "CREATE", "UPDATE", "DELETE", "EXPORT", "REPORT_GENERATED", "DATA_IMPORT", "MODEL_EXECUTION", "ACCESS_DENIED", "PASSWORD_CHANGE"].map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Buscar</label>
                <input className="input" placeholder="Usuario, descripción, recurso…" value={search} onChange={(event) => setSearch(event.target.value)} />
              </div>
              <button className="btn" onClick={() => { setPage(1); load(); }}>Aplicar</button>
            </div>

            {loading ? (
              <Loading />
            ) : events.length ? (
              <>
                <DataTable
                  rows={events}
                  rowKey={(row) => row.id}
                  columns={[
                    { key: "occurred_at", header: "Fecha y hora", render: (row) => formatDate(row.occurred_at, true) },
                    { key: "event_type", header: "Evento" },
                    { key: "module", header: "Módulo" },
                    { key: "user_email", header: "Usuario", render: (row) => <span className="small">{row.user_email || "sistema"}</span> },
                    { key: "description", header: "Descripción", render: (row) => <span className="small">{row.description}</span> },
                    { key: "result", header: "Resultado", render: (row) => <StatusBadge status={row.result} /> },
                    { key: "duration", header: "Duración", numeric: true, render: (row) => (row.duration_ms ? `${row.duration_ms} ms` : "—") },
                    { key: "hash", header: "Hash", render: (row) => <span className="mono" title={row.hash}>{row.hash.slice(0, 10)}…</span> },
                  ]}
                />
                <div style={{ display: "flex", gap: 8, marginTop: 12, alignItems: "center" }}>
                  <button className="btn btn-sm" disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Anterior</button>
                  <span className="small muted">Página {page} de {pages}</span>
                  <button className="btn btn-sm" disabled={page >= pages} onClick={() => setPage((value) => value + 1)}>Siguiente</button>
                </div>
              </>
            ) : (
              <EmptyState title="Sin eventos" description="Ajuste los filtros de búsqueda." />
            )}
          </CardBody>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card>
            <CardHeader title="Resumen" subtitle="Distribución de eventos" />
            <CardBody>
              {stats ? (
                <>
                  <div className="grid grid-2" style={{ marginBottom: 12 }}>
                    <div>
                      <div className="kpi-label">Total</div>
                      <div className="kpi-value" style={{ fontSize: 22 }}>{stats.total_events}</div>
                    </div>
                    <div>
                      <div className="kpi-label">Sesiones activas</div>
                      <div className="kpi-value" style={{ fontSize: 22 }}>{stats.active_sessions}</div>
                    </div>
                  </div>
                  <h3 style={{ fontSize: 12, marginBottom: 6 }}>Por tipo de evento</h3>
                  <BarList items={(stats.by_type || []).slice(0, 6).map((item: any) => ({ label: item.key, value: item.count }))} />
                  <h3 style={{ fontSize: 12, margin: "12px 0 6px" }}>Por módulo</h3>
                  <BarList items={(stats.by_module || []).slice(0, 6).map((item: any) => ({ label: item.key, value: item.count }))} />
                </>
              ) : (
                <Loading />
              )}
            </CardBody>
          </Card>
        </div>
      </div>
    </>
  );
}
