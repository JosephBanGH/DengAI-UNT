import { useEffect, useState } from "react";
import { api, downloadFile } from "../api/client";
import type { ReportItem, ReportType } from "../api/types";
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
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function ReportsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [types, setTypes] = useState<ReportType[]>([]);
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<"all" | "operational" | "managerial">("all");
  const [selectedType, setSelectedType] = useState<string>("");
  const [format, setFormat] = useState("pdf");
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");
  const [generating, setGenerating] = useState(false);

  const load = () => {
    setLoading(true);
    Promise.all([api.get<ReportType[]>("/reports/types"), api.get<{ items: ReportItem[] }>("/reports", { size: 50 })])
      .then(([typeData, reportData]) => {
        setTypes(typeData);
        setReports(reportData.items);
        if (!selectedType && typeData.length) setSelectedType(typeData[0].key);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const generate = async () => {
    if (!selectedType) return;
    setGenerating(true);
    try {
      const payload: Record<string, unknown> = { report_type: selectedType, format, filters: {} };
      if (periodStart && periodEnd) {
        payload.period_start = periodStart;
        payload.period_end = periodEnd;
      }
      const report = await api.post<ReportItem>("/reports", payload);
      toast.push(`Reporte generado: ${report.title} (${formatBytes(report.file_size)}).`, "success");
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    } finally {
      setGenerating(false);
    }
  };

  const filtered = types.filter((type) => category === "all" || type.category === category);
  const activeType = types.find((type) => type.key === selectedType);
  const visibleReports = reports.filter((report) => category === "all" || report.category === category);

  if (loading) return <Loading label="Cargando catálogo de reportes…" />;

  return (
    <>
      <PageHeader
        title="Reportes operacionales y gerenciales"
        description="Generación de documentos PDF, DOCX y XLSX con fecha, filtros, periodo, fuentes de datos, versión del modelo y notas metodológicas."
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Tabs
        tabs={[
          { key: "all", label: "Todos" },
          { key: "operational", label: "Operacionales" },
          { key: "managerial", label: "Gerenciales" },
        ]}
        active={category}
        onChange={(key) => setCategory(key as typeof category)}
      />

      <div className="grid grid-main" style={{ marginBottom: 14 }}>
        <Card>
          <CardHeader title="Catálogo de reportes" subtitle="Seleccione un tipo para generar el documento" />
          <CardBody>
            <div className="grid grid-2">
              {filtered.map((type) => (
                <button
                  key={type.key}
                  className="card card-pad"
                  style={{
                    textAlign: "left",
                    cursor: "pointer",
                    borderColor: selectedType === type.key ? "var(--primary)" : "var(--border)",
                    boxShadow: selectedType === type.key ? "0 0 0 3px rgba(11,93,110,0.12)" : undefined,
                  }}
                  onClick={() => setSelectedType(type.key)}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                    <strong className="small">{type.title}</strong>
                    <span className="badge badge-neutral">{type.category === "managerial" ? "Gerencial" : "Operacional"}</span>
                  </div>
                  <p className="hint" style={{ marginTop: 4 }}>{type.description}</p>
                </button>
              ))}
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Generar documento" subtitle={activeType?.title || "Seleccione un tipo"} />
          <CardBody>
            <div className="field">
              <label>Formato</label>
              <select className="select" value={format} onChange={(event) => setFormat(event.target.value)}>
                <option value="pdf">PDF</option>
                <option value="docx">Word (DOCX)</option>
                <option value="xlsx">Excel (XLSX)</option>
              </select>
            </div>
            <div className="form-row">
              <div className="field">
                <label>Desde (opcional)</label>
                <input className="input" type="date" value={periodStart} onChange={(event) => setPeriodStart(event.target.value)} />
              </div>
              <div className="field">
                <label>Hasta (opcional)</label>
                <input className="input" type="date" value={periodEnd} onChange={(event) => setPeriodEnd(event.target.value)} />
              </div>
            </div>
            <button className="btn btn-primary" style={{ width: "100%" }} onClick={generate} disabled={!can("reports:generate") || generating}>
              {generating ? "Generando…" : "Generar reporte"}
            </button>
            <p className="hint" style={{ marginTop: 10 }}>
              Cada reporte incluye la versión del modelo en producción, las fuentes utilizadas y la advertencia
              sobre el uso de datos sintéticos cuando corresponda.
            </p>
          </CardBody>
        </Card>
      </div>

      <Card>
        <CardHeader title="Historial de reportes" subtitle="Descarga y control de acceso" />
        <CardBody>
          {visibleReports.length ? (
            <DataTable
              rows={visibleReports}
              rowKey={(row) => row.id}
              columns={[
                { key: "title", header: "Reporte", render: (row) => (
                  <div>
                    <div><strong>{row.title}</strong></div>
                    <div className="hint mono">{row.report_key}</div>
                  </div>
                ) },
                { key: "category", header: "Categoría", render: (row) => (row.category === "managerial" ? "Gerencial" : "Operacional") },
                { key: "format", header: "Formato", render: (row) => row.format.toUpperCase() },
                { key: "status", header: "Estado", render: (row) => <StatusBadge status={row.status} /> },
                { key: "file_size", header: "Tamaño", numeric: true, render: (row) => formatBytes(row.file_size) },
                { key: "model_version", header: "Modelo", render: (row) => <span className="small">{row.model_version || "—"}</span> },
                { key: "generated_at", header: "Generado", render: (row) => formatDate(row.generated_at, true) },
                { key: "generated_by", header: "Usuario", render: (row) => <span className="small">{row.generated_by}</span> },
                {
                  key: "actions",
                  header: "",
                  render: (row) =>
                    row.status === "READY" ? (
                      <div style={{ display: "flex", gap: 6 }}>
                        <button
                          className="btn btn-sm"
                          onClick={() =>
                            downloadFile(`/reports/${row.id}/download`, `${row.report_key}.${row.format}`).catch((caught) =>
                              toast.push(describeError(caught), "error"),
                            )
                          }
                        >
                          Descargar
                        </button>
                        {can("reports:generate") ? (
                          <button
                            className="btn btn-sm btn-ghost"
                            onClick={async () => {
                              try {
                                await api.del(`/reports/${row.id}`);
                                toast.push("Reporte eliminado.", "success");
                                load();
                              } catch (caught) {
                                toast.push(describeError(caught), "error");
                              }
                            }}
                          >
                            Eliminar
                          </button>
                        ) : null}
                      </div>
                    ) : null,
                },
              ]}
            />
          ) : (
            <EmptyState title="Sin reportes generados" description="Genere el primer documento desde el panel superior." />
          )}
        </CardBody>
      </Card>
    </>
  );
}
