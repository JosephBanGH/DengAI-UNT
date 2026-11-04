import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { WorkflowItem } from "../api/types";
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
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function WorkflowsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [workflows, setWorkflows] = useState<WorkflowItem[]>([]);
  const [files, setFiles] = useState<any[]>([]);
  const [executions, setExecutions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    Promise.all([
      api.get<WorkflowItem[]>("/workflows"),
      api.get<any[]>("/workflows/files"),
      api.get<{ items: any[] }>("/workflows/executions", { size: 25 }),
    ])
      .then(([workflowData, fileData, executionData]) => {
        setWorkflows(workflowData);
        setFiles(fileData);
        setExecutions(executionData.items);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const run = async (code: string) => {
    setRunning(code);
    try {
      const result = await api.post<any>(`/workflows/${code}/run`);
      toast.push(`${code}: ${result.status} · ${result.items_processed} ítems procesados.`, result.status === "FAILED" ? "error" : "success");
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    } finally {
      setRunning(null);
    }
  };

  if (loading) return <Loading label="Cargando workflows…" />;

  return (
    <>
      <PageHeader
        title="Automatización de procesos (n8n)"
        description="Workflows WF-01 a WF-04 documentados e importables en n8n. La plataforma puede ejecutarlos internamente para operar sin el orquestador externo."
      />

      {error ? <ErrorAlert message={error} /> : null}

      <div className="grid grid-2" style={{ marginBottom: 14 }}>
        {workflows.map((workflow) => (
          <Card key={workflow.code}>
            <CardHeader
              title={`${workflow.code} · ${workflow.name}`}
              subtitle={workflow.schedule}
              actions={
                workflow.last_execution ? <StatusBadge status={workflow.last_execution.status} /> : <span className="badge badge-neutral">Sin ejecutar</span>
              }
            />
            <CardBody>
              <p className="small">{workflow.description}</p>
              <div className="legend" style={{ margin: "10px 0" }}>
                {workflow.steps.map((step) => (
                  <span key={step} className="chip">{step}</span>
                ))}
              </div>
              {workflow.last_execution ? (
                <p className="hint">
                  Última ejecución: {formatDate(workflow.last_execution.started_at, true)} ·{" "}
                  {workflow.last_execution.items_processed} ítems · {workflow.last_execution.duration_ms ?? 0} ms
                </p>
              ) : null}
              <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
                <button
                  className="btn btn-primary btn-sm"
                  disabled={!can("workflows:run") || running === workflow.code}
                  onClick={() => run(workflow.code)}
                >
                  {running === workflow.code ? "Ejecutando…" : "Ejecutar ahora"}
                </button>
                <a className="btn btn-sm" href={`/api/v1/workflows`} onClick={(event) => event.preventDefault()} title={workflow.file}>
                  <span className="mono">{workflow.file}</span>
                </a>
              </div>
            </CardBody>
          </Card>
        ))}
      </div>

      <div className="grid grid-2">
        <Card>
          <CardHeader title="Archivos para n8n" subtitle="Importables desde el editor de workflows" />
          <CardBody>
            {files.length ? (
              <DataTable
                rows={files}
                rowKey={(row) => row.file}
                columns={[
                  { key: "file", header: "Archivo", render: (row) => <span className="mono">{row.file}</span> },
                  { key: "name", header: "Nombre" },
                  { key: "nodes", header: "Nodos", numeric: true },
                ]}
              />
            ) : (
              <EmptyState title="Sin archivos" description="Los workflows deben estar en n8n/workflows/." />
            )}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Ejecuciones recientes" subtitle="Trazabilidad de la automatización" />
          <CardBody>
            {executions.length ? (
              <DataTable
                rows={executions}
                rowKey={(row) => row.id}
                columns={[
                  { key: "workflow_code", header: "WF" },
                  { key: "status", header: "Estado", render: (row) => <StatusBadge status={row.status} /> },
                  { key: "trigger", header: "Disparador" },
                  { key: "items_processed", header: "Ítems", numeric: true },
                  { key: "duration_ms", header: "ms", numeric: true },
                  { key: "started_at", header: "Inicio", render: (row) => formatDate(row.started_at, true) },
                ]}
              />
            ) : (
              <EmptyState title="Sin ejecuciones" description="Ejecute un workflow para registrar la primera corrida." />
            )}
          </CardBody>
        </Card>
      </div>
    </>
  );
}
