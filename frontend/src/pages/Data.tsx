import { useEffect, useRef, useState } from "react";
import { api, downloadFile } from "../api/client";
import type { DataSourceItem, ImportJob, QualitySummary } from "../api/types";
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
  Tabs,
  formatBytes,
  formatDate,
  formatInteger,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

type Tab = "sources" | "import" | "quality" | "cases" | "climate";

export function DataPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("sources");
  const [sources, setSources] = useState<DataSourceItem[]>([]);
  const [quality, setQuality] = useState<QualitySummary | null>(null);
  const [imports, setImports] = useState<ImportJob[]>([]);
  const [cases, setCases] = useState<{ items: any[]; total: number } | null>(null);
  const [climate, setClimate] = useState<{ items: any[]; total: number } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState("epidemiological");
  const [sourceCode, setSourceCode] = useState("");
  const [isSynthetic, setIsSynthetic] = useState(false);
  const [preview, setPreview] = useState<any | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.get<DataSourceItem[]>("/data/sources"),
      api.get<QualitySummary>("/data/quality"),
      api.get<{ items: ImportJob[] }>("/data/imports", { size: 20 }),
    ])
      .then(([sourceData, qualityData, importData]) => {
        setSources(sourceData);
        setQuality(qualityData);
        setImports(importData.items);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (tab === "cases" && !cases) {
      api.get<{ items: any[]; total: number }>("/data/cases", { size: 50 }).then(setCases).catch((caught) => setError(describeError(caught)));
    }
    if (tab === "climate" && !climate) {
      api.get<{ items: any[]; total: number }>("/data/climate", { size: 50 }).then(setClimate).catch((caught) => setError(describeError(caught)));
    }
  }, [tab, cases, climate]);

  const chooseFile = async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    form.append("kind", kind);
    try {
      const result = await api.upload<any>("/data/import/preview", form);
      setPreview(result);
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const runImport = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      toast.push("Seleccione un archivo CSV o JSON.", "warn");
      return;
    }
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    if (sourceCode) form.append("source_code", sourceCode);
    form.append("is_synthetic", String(isSynthetic));
    try {
      const job = await api.upload<ImportJob>(kind === "epidemiological" ? "/data/import/cases" : "/data/import/climate", form);
      toast.push(`Importación ${job.status}: ${job.rows_inserted} insertados, ${job.rows_updated} actualizados.`, "success");
      setPreview(null);
      if (fileRef.current) fileRef.current.value = "";
      const importData = await api.get<{ items: ImportJob[] }>("/data/imports", { size: 20 });
      setImports(importData.items);
      const qualityData = await api.get<QualitySummary>("/data/quality");
      setQuality(qualityData);
      setCases(null);
      setClimate(null);
    } catch (caught) {
      toast.push(describeError(caught), "error");
    } finally {
      setUploading(false);
    }
  };

  const probe = async (code: string) => {
    try {
      const result = await api.post<any>(`/data/sources/${code}/probe`);
      toast.push(`${code}: ${result.message}`, result.reachable ? "success" : "warn");
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  if (loading) return <Loading label="Cargando catálogos de datos…" />;

  return (
    <>
      <PageHeader
        title="Datos, importación y calidad"
        description="Administración de fuentes, carga de archivos distrito-semana, validación, detección de duplicados y reportes de calidad."
        actions={
          <button
            className="btn"
            onClick={async () => {
              try {
                await downloadFile("/data/cases?size=500", "casos.csv");
              } catch {
                toast.push("La exportación de datos se realiza desde Reportes (XLSX).", "warn");
              }
            }}
          >
            Exportar muestra
          </button>
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Tabs
        tabs={[
          { key: "sources", label: "Fuentes de datos" },
          { key: "import", label: "Importación" },
          { key: "quality", label: "Calidad" },
          { key: "cases", label: "Casos" },
          { key: "climate", label: "Clima" },
        ]}
        active={tab}
        onChange={(key) => setTab(key as Tab)}
      />

      {tab === "sources" ? (
        <Card>
          <CardHeader title="Catálogo de fuentes" subtitle="Licencias, método de acceso y estado de integración" />
          <CardBody>
            <DataTable
              rows={sources}
              rowKey={(row) => row.code}
              columns={[
                { key: "name", header: "Fuente", render: (row) => (
                  <div>
                    <div><strong>{row.name}</strong></div>
                    <div className="hint">{row.provider}</div>
                  </div>
                ) },
                { key: "kind", header: "Tipo" },
                { key: "license", header: "Licencia", render: (row) => <span className="small">{row.license}</span> },
                {
                  key: "integration_status",
                  header: "Estado",
                  render: (row) => (
                    <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                      {row.is_synthetic ? <span className="badge badge-warn">Sintético</span> : null}
                      <StatusBadge status={row.integration_status.split("_")[0]} />
                    </div>
                  ),
                },
                {
                  key: "actions",
                  header: "",
                  render: (row) =>
                    can("data:import") ? (
                      <button className="btn btn-sm" onClick={() => probe(row.code)}>Verificar</button>
                    ) : null,
                },
              ]}
            />
            <p className="hint" style={{ marginTop: 10 }}>
              Las fuentes sin API pública verificada se integran por carga de archivos estructurados. NASA POWER
              expone una API abierta y su adaptador está implementado; requiere <span className="mono">LIVE_INTEGRATIONS_ENABLED=1</span>.
            </p>
          </CardBody>
        </Card>
      ) : null}

      {tab === "import" ? (
        <div className="grid grid-2">
          <Card>
            <CardHeader title="Importar archivo" subtitle="CSV o JSON con una fila por distrito-semana" />
            <CardBody>
              <div className="field">
                <label>Tipo de datos</label>
                <select className="select" value={kind} onChange={(event) => setKind(event.target.value)}>
                  <option value="epidemiological">Casos de dengue (epidemiológico)</option>
                  <option value="climate">Variables climáticas</option>
                </select>
              </div>
              <div className="field">
                <label>Fuente</label>
                <select className="select" value={sourceCode} onChange={(event) => setSourceCode(event.target.value)}>
                  <option value="">Sin especificar</option>
                  {sources.map((source) => (
                    <option key={source.code} value={source.code}>{source.name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Archivo</label>
                <input
                  ref={fileRef}
                  className="input"
                  type="file"
                  accept=".csv,.json"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void chooseFile(file);
                  }}
                />
                <span className="hint">
                  Columnas obligatorias: {kind === "epidemiological"
                    ? "district_code, epi_year, epi_week, confirmed_cases"
                    : "district_code, epi_year, epi_week, temperature_mean, humidity_mean, precipitation_mm"}
                </span>
              </div>
              <label className="small" style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 12 }}>
                <input type="checkbox" checked={isSynthetic} onChange={(event) => setIsSynthetic(event.target.checked)} />
                Marcar los registros como sintéticos
              </label>
              <button className="btn btn-primary" onClick={runImport} disabled={uploading}>
                {uploading ? "Importando…" : "Importar"}
              </button>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Previsualización y validación" subtitle="Verificación previa a la carga" />
            <CardBody>
              {preview ? (
                <>
                  <p className="small">
                    <strong>{preview.detected_rows}</strong> filas detectadas · Estado:{" "}
                    <StatusBadge status={preview.ready ? "READY" : "FAILED"} />
                  </p>
                  {preview.validation.length ? (
                    <ul className="small" style={{ paddingLeft: 18 }}>
                      {preview.validation.map((item: any, index: number) => (
                        <li key={index} className={item.level === "ERROR" ? "form-error" : "muted"}>
                          {item.message} ({item.count})
                        </li>
                      ))}
                    </ul>
                  ) : null}
                  <div className="table-wrap" style={{ marginTop: 10 }}>
                    <table className="data">
                      <thead>
                        <tr>{preview.columns.map((column: string) => <th key={column}>{column}</th>)}</tr>
                      </thead>
                      <tbody>
                        {preview.sample_rows.slice(0, 5).map((row: any, index: number) => (
                          <tr key={index}>
                            {preview.columns.map((column: string) => <td key={column}>{String(row[column] ?? "")}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <EmptyState title="Sin archivo seleccionado" description="Seleccione un archivo para validar su estructura antes de importar." />
              )}
            </CardBody>
          </Card>

          <Card style={{ gridColumn: "1 / -1" }}>
            <CardHeader title="Historial de importaciones" subtitle="Trazabilidad de procesos de carga" />
            <CardBody>
              <DataTable
                rows={imports}
                rowKey={(row) => row.job_key}
                columns={[
                  { key: "job_key", header: "Clave", render: (row) => <span className="mono">{row.job_key}</span> },
                  { key: "file_name", header: "Archivo" },
                  { key: "kind", header: "Tipo" },
                  { key: "status", header: "Estado", render: (row) => <StatusBadge status={row.status} /> },
                  { key: "rows_total", header: "Total", numeric: true },
                  { key: "rows_inserted", header: "Insertados", numeric: true },
                  { key: "rows_rejected", header: "Rechazados", numeric: true },
                  { key: "started_at", header: "Fecha", render: (row) => formatDate(row.started_at, true) },
                  { key: "initiated_by", header: "Usuario" },
                ]}
              />
            </CardBody>
          </Card>
        </div>
      ) : null}

      {tab === "quality" && quality ? (
        <>
          <div className="grid grid-kpi" style={{ marginBottom: 14 }}>
            <div className="card kpi"><span className="kpi-label">Registros de casos</span><span className="kpi-value">{formatInteger(quality.total_cases)}</span></div>
            <div className="card kpi"><span className="kpi-label">Registros climáticos</span><span className="kpi-value">{formatInteger(quality.total_climate)}</span></div>
            <div className="card kpi"><span className="kpi-label">Distritos cubiertos</span><span className="kpi-value">{quality.districts_covered}</span></div>
            <div className="card kpi"><span className="kpi-label">Semanas cubiertas</span><span className="kpi-value">{quality.weeks_covered}</span></div>
            <div className="card kpi"><span className="kpi-label">Datos sintéticos</span><span className="kpi-value">{formatNumber(quality.synthetic_share_pct, 1)}<span className="kpi-unit">%</span></span></div>
          </div>
          <Card>
            <CardHeader title="Chequeos de calidad" subtitle="Resultados de las validaciones ejecutadas" />
            <CardBody>
              <DataTable
                rows={quality.checks}
                rowKey={(row) => row.id}
                columns={[
                  { key: "dataset_name", header: "Conjunto" },
                  { key: "check_name", header: "Chequeo" },
                  { key: "severity", header: "Severidad", render: (row) => <StatusBadge status={row.severity} /> },
                  { key: "rows_checked", header: "Revisados", numeric: true },
                  { key: "rows_flagged", header: "Marcados", numeric: true },
                  { key: "created_at", header: "Fecha", render: (row) => formatDate(row.created_at) },
                ]}
              />
            </CardBody>
          </Card>
        </>
      ) : null}

      {tab === "cases" ? (
        <Card>
          <CardHeader title="Casos registrados" subtitle="Unidad de análisis: distrito-semana epidemiológica" />
          <CardBody>
            {cases ? (
              <DataTable
                rows={cases.items}
                rowKey={(row, index) => `${row.district_code}-${row.epi_year}-${row.epi_week}-${index}`}
                columns={[
                  { key: "district_name", header: "Distrito" },
                  { key: "epi_year", header: "Año", numeric: true },
                  { key: "epi_week", header: "Semana", numeric: true },
                  { key: "week_start", header: "Inicio", render: (row) => formatDate(row.week_start) },
                  { key: "confirmed_cases", header: "Confirmados", numeric: true },
                  { key: "hospitalized", header: "Hospitalizados", numeric: true },
                  { key: "deaths", header: "Defunciones", numeric: true },
                  { key: "is_synthetic", header: "Origen", render: (row) => (row.is_synthetic ? <span className="badge badge-warn">Sintético</span> : <span className="badge badge-ok">Fuente</span>) },
                ]}
              />
            ) : (
              <Loading />
            )}
            <p className="hint" style={{ marginTop: 8 }}>Mostrando 50 registros de {cases?.total ?? 0}.</p>
          </CardBody>
        </Card>
      ) : null}

      {tab === "climate" ? (
        <Card>
          <CardHeader title="Observaciones climáticas" subtitle="Variables semanales por distrito" />
          <CardBody>
            {climate ? (
              <DataTable
                rows={climate.items}
                rowKey={(row, index) => `${row.district_code}-${row.epi_year}-${row.epi_week}-${index}`}
                columns={[
                  { key: "district_code", header: "Distrito" },
                  { key: "epi_year", header: "Año", numeric: true },
                  { key: "epi_week", header: "Semana", numeric: true },
                  { key: "temperature_mean", header: "T° media", numeric: true, render: (row) => formatNumber(row.temperature_mean, 1) },
                  { key: "humidity_mean", header: "Humedad", numeric: true, render: (row) => formatNumber(row.humidity_mean, 1) },
                  { key: "precipitation_mm", header: "Precip. mm", numeric: true, render: (row) => formatNumber(row.precipitation_mm, 1) },
                  { key: "provider", header: "Proveedor" },
                ]}
              />
            ) : (
              <Loading />
            )}
            <p className="hint" style={{ marginTop: 8 }}>Mostrando 50 registros de {climate?.total ?? 0}.</p>
          </CardBody>
        </Card>
      ) : null}
    </>
  );
}
