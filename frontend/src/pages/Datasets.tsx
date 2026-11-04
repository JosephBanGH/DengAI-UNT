import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetItem } from "../api/types";
import { PageHeader } from "../components/Layout";
import { Card, CardBody, CardHeader, DataTable, EmptyState, ErrorAlert, Loading, Modal, formatDate, formatInteger } from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

export function DatasetsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("Panel distrito-semana");
  const [description, setDescription] = useState("Conjunto de datos con rezagos de 1 a 12 semanas para modelamiento.");
  const [maxLag, setMaxLag] = useState(12);
  const [dictionary, setDictionary] = useState<{ dataset: DatasetItem; rows: any[] } | null>(null);

  const load = () => {
    setLoading(true);
    api.get<DatasetItem[]>("/datasets")
      .then(setItems)
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const build = async () => {
    try {
      const result = await api.post<DatasetItem>("/datasets/build", { name, description, max_lag: maxLag, include_synthetic: true });
      toast.push(`Conjunto construido: ${result.rows} filas, ${result.districts} distritos.`, "success");
      setCreating(false);
      load();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const openDictionary = async (dataset: DatasetItem) => {
    try {
      const rows = await api.get<any[]>(`/datasets/${dataset.id}/dictionary`);
      setDictionary({ dataset, rows });
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  return (
    <>
      <PageHeader
        title="Conjuntos de datos y reproducibilidad"
        description="Versiones del panel distrito-semana con diccionario de datos, esquema de variables y suma de verificación para reproducir los experimentos."
        actions={can("datasets:write") ? <button className="btn btn-primary" onClick={() => setCreating(true)}>Construir conjunto</button> : undefined}
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Card>
        <CardHeader title="Conjuntos registrados" subtitle="Cada versión queda identificada por su checksum" />
        <CardBody>
          {loading ? (
            <Loading />
          ) : items.length ? (
            <DataTable
              rows={items}
              rowKey={(row) => row.id}
              columns={[
                { key: "name", header: "Nombre", render: (row) => (
                  <div>
                    <div><strong>{row.name}</strong></div>
                    <div className="hint mono">{row.slug}</div>
                  </div>
                ) },
                { key: "version", header: "Versión" },
                { key: "rows", header: "Filas", numeric: true, render: (row) => formatInteger(row.rows) },
                { key: "districts", header: "Distritos", numeric: true },
                { key: "weeks", header: "Semanas", numeric: true },
                {
                  key: "period",
                  header: "Periodo",
                  render: (row) => `${row.start_week ? formatDate(row.start_week) : "—"} → ${row.end_week ? formatDate(row.end_week) : "—"}`,
                },
                { key: "is_synthetic", header: "Datos", render: (row) => (row.is_synthetic ? <span className="badge badge-warn">Sintéticos</span> : <span className="badge badge-ok">Fuente</span>) },
                { key: "created_at", header: "Creado", render: (row) => formatDate(row.created_at) },
                {
                  key: "actions",
                  header: "",
                  render: (row) => (
                    <button className="btn btn-sm btn-ghost" onClick={() => openDictionary(row)}>Diccionario</button>
                  ),
                },
              ]}
            />
          ) : (
            <EmptyState
              title="Sin conjuntos de datos"
              description="Construya una versión a partir de los datos cargados para dejar trazabilidad del panel usado en cada experimento."
            />
          )}
        </CardBody>
      </Card>

      {creating ? (
        <Modal
          title="Construir conjunto de datos"
          onClose={() => setCreating(false)}
          footer={
            <>
              <button className="btn" onClick={() => setCreating(false)}>Cancelar</button>
              <button className="btn btn-primary" onClick={build}>Construir</button>
            </>
          }
        >
          <div className="field">
            <label>Nombre</label>
            <input className="input" value={name} onChange={(event) => setName(event.target.value)} />
          </div>
          <div className="field">
            <label>Descripción</label>
            <textarea className="textarea" value={description} onChange={(event) => setDescription(event.target.value)} />
          </div>
          <div className="field">
            <label>Rezagos máximos (semanas)</label>
            <input className="input" type="number" min={1} max={24} value={maxLag} onChange={(event) => setMaxLag(Number(event.target.value))} />
          </div>
          <p className="hint">
            El conjunto se construye con los datos cargados. Si la base contiene únicamente datos sintéticos, la
            versión queda marcada como sintética.
          </p>
        </Modal>
      ) : null}

      {dictionary ? (
        <Modal title={`Diccionario de datos · ${dictionary.dataset.name}`} onClose={() => setDictionary(null)}>
          <p className="hint" style={{ marginBottom: 10 }}>
            {dictionary.rows.length} variables documentadas. Checksum{" "}
            <span className="mono">{dictionary.dataset.checksum.slice(0, 16)}…</span>
          </p>
          <div className="table-wrap" style={{ maxHeight: 420 }}>
            <table className="data">
              <thead>
                <tr>
                  <th>Variable</th>
                  <th>Descripción</th>
                  <th>Unidad</th>
                  <th>Origen</th>
                </tr>
              </thead>
              <tbody>
                {dictionary.rows.map((row) => (
                  <tr key={row.column}>
                    <td className="mono">{row.column}</td>
                    <td className="small">{row.description}</td>
                    <td className="small">{row.unit || "—"}</td>
                    <td className="small">{row.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Modal>
      ) : null}
    </>
  );
}
