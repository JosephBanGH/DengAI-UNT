import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ConfigEntry, Health } from "../api/types";
import { PageHeader } from "../components/Layout";
import { Card, CardBody, CardHeader, EmptyState, ErrorAlert, Loading, StatusBadge, Tabs } from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function SettingsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [tab, setTab] = useState<"config" | "health" | "stats">("config");
  const [config, setConfig] = useState<ConfigEntry[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  useEffect(() => {
    Promise.all([api.get<ConfigEntry[]>("/admin/config"), api.get<Health>("/health"), api.get<any>("/admin/stats")])
      .then(([configData, healthData, statsData]) => {
        setConfig(configData);
        setHealth(healthData);
        setStats(statsData);
        setDrafts(Object.fromEntries(configData.map((entry) => [entry.key, JSON.stringify(entry.value, null, 2)])));
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  }, []);

  const save = async (entry: ConfigEntry) => {
    try {
      const value = JSON.parse(drafts[entry.key]);
      await api.put(`/admin/config/${entry.key}`, { value });
      toast.push(`Configuración '${entry.key}' actualizada.`, "success");
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  if (loading) return <Loading label="Cargando configuración del sistema…" />;

  return (
    <>
      <PageHeader
        title="Administración y configuración"
        description="Parámetros epidemiológicos, umbrales de riesgo, salud de la plataforma y estadísticas globales."
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Tabs
        tabs={[
          { key: "config", label: "Configuración" },
          { key: "health", label: "Estado de la plataforma" },
          { key: "stats", label: "Estadísticas" },
        ]}
        active={tab}
        onChange={(key) => setTab(key as typeof tab)}
      />

      {tab === "config" ? (
        <div className="grid" style={{ gap: 14 }}>
          {config.map((entry) => (
            <Card key={entry.key}>
              <CardHeader
                title={entry.key}
                subtitle={entry.description}
                actions={
                  can("admin:config") ? (
                    <button className="btn btn-sm btn-primary" onClick={() => save(entry)}>Guardar</button>
                  ) : null
                }
              />
              <CardBody>
                <textarea
                  className="textarea mono"
                  style={{ minHeight: 120 }}
                  value={drafts[entry.key] ?? ""}
                  disabled={!can("admin:config")}
                  onChange={(event) => setDrafts({ ...drafts, [entry.key]: event.target.value })}
                />
                <p className="hint">Última actualización por {entry.updated_by || "system"}.</p>
              </CardBody>
            </Card>
          ))}
          {!config.length ? <EmptyState title="Sin configuración" /> : null}
        </div>
      ) : null}

      {tab === "health" && health ? (
        <div className="grid grid-2">
          <Card>
            <CardHeader title="Servicios" subtitle={`Versión ${health.version} · entorno ${health.environment}`} />
            <CardBody>
              <Row label="Estado" value={<StatusBadge status={health.status === "ok" ? "READY" : "FAILED"} />} />
              <Row label="Base de datos" value={`${health.database} (${health.database_url_scheme})`} />
              <Row label="PostGIS" value={health.postgis ? "habilitado" : "deshabilitado"} />
              <Row label="PyTorch" value={health.ml_backend.torch.available ? `v${health.ml_backend.torch.version}` : "no disponible"} />
              <Row label="XGBoost" value={health.ml_backend.xgboost.available ? `v${health.ml_backend.xgboost.version}` : "no disponible"} />
              {Object.entries(health.integrations).map(([key, value]) => (
                <Row key={key} label={key} value={String(value)} />
              ))}
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Limitaciones declaradas" subtitle="Transparencia sobre el entorno actual" />
            <CardBody>
              <ul className="small" style={{ paddingLeft: 18 }}>
                {health.limitations.map((limitation, index) => (
                  <li key={index} style={{ marginBottom: 8 }}>{limitation}</li>
                ))}
              </ul>
            </CardBody>
          </Card>
        </div>
      ) : null}

      {tab === "stats" && stats ? (
        <Card>
          <CardHeader title="Estadísticas globales" subtitle="Conteo de entidades registradas" />
          <CardBody>
            <div className="grid grid-3">
              {Object.entries(stats).map(([key, value]) => (
                <div key={key}>
                  <div className="kpi-label">{key.replace(/_/g, " ")}</div>
                  <div className="kpi-value" style={{ fontSize: 22 }}>{String(value)}</div>
                </div>
              ))}
            </div>
          </CardBody>
        </Card>
      ) : null}
    </>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
      <span className="small muted">{label}</span>
      <span className="small" style={{ textAlign: "right" }}>{value}</span>
    </div>
  );
}
