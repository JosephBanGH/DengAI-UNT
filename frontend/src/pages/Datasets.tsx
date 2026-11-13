import { useEffect, useState } from "react";
import { api } from "../api/client";
import type {
  DataDictionaryItem,
  DatasetItem,
  DatasetStatisticsResult,
  GroupValidationResult,
} from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  DataTable,
  EmptyState,
  ErrorAlert,
  KpiCard,
  Loading,
  Modal,
  StatusBadge,
  Tabs,
  formatDate,
  formatInteger,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

type Tab = "datasets" | "validation" | "statistics";

export function DatasetsPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("datasets");
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal para construir conjunto
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("Panel distrito-semana");
  const [description, setDescription] = useState("Conjunto de datos con rezagos de 1 a 12 semanas para modelamiento.");
  const [maxLag, setMaxLag] = useState(12);

  // Diccionario de datos
  const [dictionary, setDictionary] = useState<{ dataset: DatasetItem; rows: DataDictionaryItem[] } | null>(null);
  const [editingEntry, setEditingEntry] = useState<{ column: string; description: string; unit: string; source: string } | null>(null);
  const [savingEdit, setSavingEdit] = useState(false);

  // Validación de grupos
  const [validation, setValidation] = useState<GroupValidationResult | null>(null);
  const [valLoading, setValLoading] = useState(false);

  // Estadística descriptiva
  const [statistics, setStatistics] = useState<DatasetStatisticsResult | null>(null);
  const [statLoading, setStatLoading] = useState(false);

  const loadDatasets = () => {
    setLoading(true);
    api.get<DatasetItem[]>("/datasets")
      .then(setItems)
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  const loadValidation = () => {
    setValLoading(true);
    api.get<GroupValidationResult>("/data/group-validation")
      .then(setValidation)
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setValLoading(false));
  };

  const loadStatistics = () => {
    setStatLoading(true);
    api.get<DatasetStatisticsResult>("/data/statistics")
      .then(setStatistics)
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setStatLoading(false));
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (tab === "validation" && !validation) {
      loadValidation();
    } else if (tab === "statistics" && !statistics) {
      loadStatistics();
    }
  }, [tab, validation, statistics]);

  const build = async () => {
    try {
      const result = await api.post<DatasetItem>("/datasets/build", { name, description, max_lag: maxLag, include_synthetic: true });
      toast.push(`Conjunto construido: ${result.rows} filas, ${result.districts} distritos.`, "success");
      setCreating(false);
      loadDatasets();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const openDictionary = async (dataset: DatasetItem) => {
    try {
      const rows = await api.get<DataDictionaryItem[]>(`/datasets/${dataset.id}/dictionary`);
      setDictionary({ dataset, rows });
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const saveDictionaryEntry = async () => {
    if (!dictionary || !editingEntry) return;
    setSavingEdit(true);
    try {
      const updated = await api.put<DataDictionaryItem>(
        `/datasets/${dictionary.dataset.id}/dictionary/${editingEntry.column}`,
        {
          description: editingEntry.description,
          unit: editingEntry.unit,
          source: editingEntry.source,
        }
      );
      toast.push(`Metadatos de '${editingEntry.column}' actualizados.`, "success");
      setDictionary({
        ...dictionary,
        rows: dictionary.rows.map((r) => (r.column === updated.column ? updated : r)),
      });
      setEditingEntry(null);
    } catch (caught) {
      toast.push(describeError(caught), "error");
    } finally {
      setSavingEdit(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Fase 1: Gestión de Datos, Validación y Estadística"
        description="Exploración del panel distrito-semana, editor de metadatos, control estricto de fuga de información (Walk-Forward / Spatial CV) y caracterización estadística explicable."
        actions={
          can("datasets:write") ? (
            <button className="btn btn-primary" onClick={() => setCreating(true)}>
              Construir conjunto versionado
            </button>
          ) : undefined
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Tabs
        tabs={[
          { key: "datasets", label: "Conjuntos Registrados y Metadatos" },
          { key: "validation", label: "Validación de Grupos y Fuga de Datos" },
          { key: "statistics", label: "Estadística Descriptiva y Explicabilidad" },
        ]}
        active={tab}
        onChange={(key) => setTab(key as Tab)}
      />

      {tab === "datasets" ? (
        <Card>
          <CardHeader title="Conjuntos de datos versionados" subtitle="Cada versión queda identificada por su suma de verificación SHA-256 para reproducibilidad" />
          <CardBody>
            {loading ? (
              <Loading />
            ) : items.length ? (
              <DataTable
                rows={items}
                rowKey={(row) => row.id}
                columns={[
                  {
                    key: "name",
                    header: "Nombre",
                    render: (row) => (
                      <div>
                        <div><strong>{row.name}</strong></div>
                        <div className="hint mono">{row.slug}</div>
                      </div>
                    ),
                  },
                  { key: "version", header: "Versión" },
                  { key: "rows", header: "Filas", numeric: true, render: (row) => formatInteger(row.rows) },
                  { key: "districts", header: "Distritos", numeric: true },
                  { key: "weeks", header: "Semanas", numeric: true },
                  {
                    key: "period",
                    header: "Periodo",
                    render: (row) => `${row.start_week ? formatDate(row.start_week) : "—"} → ${row.end_week ? formatDate(row.end_week) : "—"}`,
                  },
                  {
                    key: "is_synthetic",
                    header: "Procedencia",
                    render: (row) => (row.is_synthetic ? <span className="badge badge-warn">Sintético</span> : <span className="badge badge-ok">Fuente Oficial</span>),
                  },
                  { key: "created_at", header: "Creado", render: (row) => formatDate(row.created_at) },
                  {
                    key: "actions",
                    header: "Metadatos",
                    render: (row) => (
                      <button className="btn btn-sm btn-ghost" onClick={() => openDictionary(row)}>
                        Diccionario
                      </button>
                    ),
                  },
                ]}
              />
            ) : (
              <EmptyState
                title="Sin conjuntos de datos versionados"
                description="Construya una versión a partir de los datos cargados para dejar trazabilidad del panel usado en cada experimento."
              />
            )}
          </CardBody>
        </Card>
      ) : null}

      {tab === "validation" ? (
        valLoading || !validation ? (
          <Loading label="Ejecutando auditoría de grupos y chequeo anti-fuga..." />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="kpi-grid">
              <KpiCard
                label="Riesgo de Fuga de Información"
                value={`${validation.leakage_risk.risk_score_pct}%`}
                hint={validation.leakage_risk.risk_level === "LOW" ? "Bajo (0 Fuga Detectada)" : "Riesgo Detectado"}
              />
              <KpiCard
                label="Grupos Espaciales (Distritos)"
                value={validation.groups_summary.districts_count}
                hint={`Balance min/max: ${validation.groups_summary.balance_ratio}`}
              />
              <KpiCard
                label="Grupos Temporales (Semanas)"
                value={validation.groups_summary.weeks_count}
                hint={`${validation.groups_summary.start_week} a ${validation.groups_summary.end_week}`}
              />
              <KpiCard
                label="Chequeos Metodológicos"
                value={`${validation.leakage_risk.passed_checks} / ${validation.leakage_risk.total_checks}`}
                hint="Verificaciones superadas"
              />
            </div>

            <Card>
              <CardHeader
                title="Semáforo de Fuga y Verificaciones Obligatorias del Asesor"
                subtitle="Evaluación formal de no solapamiento temporal, aislamiento geográfico y causalidad de rezagos"
                actions={
                  <button className="btn btn-sm btn-ghost" onClick={loadValidation}>
                    Re-auditar
                  </button>
                }
              />
              <CardBody>
                <DataTable
                  rows={validation.leakage_risk.verifications}
                  rowKey={(row) => row.id}
                  columns={[
                    { key: "id", header: "Código", render: (r) => <strong className="mono">{r.id}</strong> },
                    { key: "title", header: "Criterio de Validación", render: (r) => <strong>{r.title}</strong> },
                    { key: "description", header: "Alcance", render: (r) => <span className="small">{r.description}</span> },
                    {
                      key: "status",
                      header: "Resultado",
                      render: (r) => (
                        <span className={`badge badge-${r.status === "PASSED" ? "ok" : "danger"}`}>
                          {r.status === "PASSED" ? "Superado" : "Fallo"}
                        </span>
                      ),
                    },
                    { key: "details", header: "Evidencia Verificable", render: (r) => <span className="small mono">{r.details}</span> },
                  ]}
                />
              </CardBody>
            </Card>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <Card>
                <CardHeader title="Partición Temporal: Walk-Forward" subtitle="Entrenamiento en pasado, evaluación estricta en futuro" />
                <CardBody>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13 }}>
                    <div><strong>Entrenamiento:</strong> {validation.partitions_audit.temporal_walk_forward.train_period} ({formatInteger(validation.partitions_audit.temporal_walk_forward.train_rows)} filas)</div>
                    <div><strong>Validación:</strong> {validation.partitions_audit.temporal_walk_forward.val_period} ({formatInteger(validation.partitions_audit.temporal_walk_forward.val_rows)} filas)</div>
                    <div><strong>Prueba:</strong> {validation.partitions_audit.temporal_walk_forward.test_period} ({formatInteger(validation.partitions_audit.temporal_walk_forward.test_rows)} filas)</div>
                    <div style={{ marginTop: 6, color: "#16a34a", fontWeight: 600 }}>
                      ✓ Semanas solapadas entre particiones: {validation.partitions_audit.temporal_walk_forward.overlap_weeks}
                    </div>
                  </div>
                </CardBody>
              </Card>

              <Card>
                <CardHeader title="Partición Espacial: Spatial Block CV" subtitle="Evaluación sobre distritos geográficos ciegos" />
                <CardBody>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13 }}>
                    <div><strong>Distritos Train ({validation.partitions_audit.spatial_block_cv.train_districts.length}):</strong> <span className="small">{validation.partitions_audit.spatial_block_cv.train_districts.join(", ")}</span></div>
                    <div><strong>Distritos Val ({validation.partitions_audit.spatial_block_cv.val_districts.length}):</strong> <span className="small">{validation.partitions_audit.spatial_block_cv.val_districts.join(", ")}</span></div>
                    <div><strong>Distritos Test ({validation.partitions_audit.spatial_block_cv.test_districts.length}):</strong> <span className="small">{validation.partitions_audit.spatial_block_cv.test_districts.join(", ")}</span></div>
                    <div style={{ marginTop: 6, color: "#16a34a", fontWeight: 600 }}>
                      ✓ Distritos solapados entre particiones: {validation.partitions_audit.spatial_block_cv.overlap_districts}
                    </div>
                  </div>
                </CardBody>
              </Card>
            </div>

            <Card>
              <CardHeader title="Directivas y Recomendaciones Metodológicas de Reproducibilidad" />
              <CardBody>
                <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, lineHeight: 1.6 }}>
                  {validation.methodological_recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              </CardBody>
            </Card>
          </div>
        )
      ) : null}

      {tab === "statistics" ? (
        statLoading || !statistics ? (
          <Loading label="Calculando estadísticas descriptivas y correlaciones..." />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <Card>
              <CardHeader
                title="Estadísticas Descriptivas Univariadas del Panel"
                subtitle="Indicadores paramétricos y no paramétricos sobre el total de observaciones distrito-semana"
              />
              <CardBody>
                <DataTable
                  rows={statistics.univariate_statistics}
                  rowKey={(r) => r.variable}
                  columns={[
                    { key: "variable", header: "Variable", render: (r) => <strong>{r.variable}</strong> },
                    { key: "unit", header: "Unidad", render: (r) => <span className="small">{r.unit}</span> },
                    { key: "valid_count", header: "N Válido", numeric: true, render: (r) => formatInteger(r.valid_count) },
                    { key: "missing_pct", header: "% Nulos", numeric: true, render: (r) => `${r.missing_pct}%` },
                    { key: "mean", header: "Media", numeric: true, render: (r) => formatNumber(r.mean, 2) },
                    { key: "std", header: "Desv. Est.", numeric: true, render: (r) => formatNumber(r.std, 2) },
                    { key: "median", header: "Mediana", numeric: true, render: (r) => formatNumber(r.median, 2) },
                    { key: "iqr", header: "IQR", numeric: true, render: (r) => formatNumber(r.iqr, 2) },
                    { key: "min", header: "Mín", numeric: true, render: (r) => formatNumber(r.min, 2) },
                    { key: "max", header: "Máx", numeric: true, render: (r) => formatNumber(r.max, 2) },
                    {
                      key: "skewness",
                      header: "Asimetría",
                      numeric: true,
                      render: (r) => (
                        <span style={{ fontWeight: r.skewness && r.skewness > 1 ? 600 : 400 }}>
                          {formatNumber(r.skewness, 2)}
                        </span>
                      ),
                    },
                  ]}
                />
              </CardBody>
            </Card>

            <Card>
              <CardHeader
                title="Matriz de Asociaciones Bivariadas (Clima vs. Casos de Dengue)"
                subtitle="Coeficientes de correlación paramétrica de Pearson y no paramétrica de Spearman con significancia p < 0.05"
              />
              <CardBody>
                <DataTable
                  rows={statistics.correlations}
                  rowKey={(r) => r.feature_x}
                  columns={[
                    { key: "feature_x", header: "Variable Climática / Exógena", render: (r) => <strong>{r.feature_x}</strong> },
                    { key: "feature_y", header: "Variable Objetivo", render: (r) => r.feature_y },
                    { key: "pearson", header: "Pearson (r)", numeric: true, render: (r) => formatNumber(r.pearson, 3) },
                    { key: "pearson_p_value", header: "p-valor (Pearson)", numeric: true, render: (r) => formatNumber(r.pearson_p_value, 5) },
                    { key: "spearman", header: "Spearman (rho)", numeric: true, render: (r) => formatNumber(r.spearman, 3) },
                    {
                      key: "significant_05",
                      header: "Significativo (p < 0.05)",
                      render: (r) => (r.significant_05 ? <span className="badge badge-ok">Significativo</span> : <span className="badge badge-neutral">No sign.</span>),
                    },
                  ]}
                />
              </CardBody>
            </Card>

            <Card>
              <CardHeader
                title="Explicabilidad e Interpretabilidad en Lenguaje Natural"
                subtitle="Síntesis epistemológica automatizada sustentada en los cálculos matemáticos realizados"
              />
              <CardBody>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {statistics.automated_explanations.map((exp, idx) => (
                    <div key={idx} style={{ padding: 12, background: "#f8fafc", borderRadius: 6, borderLeft: "4px solid #0284c7" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <strong style={{ color: "#0369a1" }}>{exp.topic}</strong>
                        <span className="small mono">{exp.indicator}</span>
                      </div>
                      <p style={{ margin: "0 0 6px 0", fontSize: 13, lineHeight: 1.5 }}>
                        <strong>Comportamiento observado:</strong> {exp.behavior}
                      </p>
                      <p style={{ margin: "0 0 6px 0", fontSize: 13, lineHeight: 1.5, color: "#334155" }}>
                        <strong>Implicancia metodológica (IA):</strong> {exp.methodological_implication}
                      </p>
                      <div style={{ fontSize: 11, color: "#64748b", fontStyle: "italic" }}>
                        ⚖️ {exp.epistemic_caution}
                      </div>
                    </div>
                  ))}
                </div>
              </CardBody>
            </Card>
          </div>
        )
      ) : null}

      {/* Modal para construir conjunto */}
      {creating ? (
        <Modal
          title="Construir conjunto de datos versionado"
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
            El conjunto se construye con el panel cargado y genera el checksum SHA-256 para trazabilidad.
          </p>
        </Modal>
      ) : null}

      {/* Modal del Diccionario de Datos */}
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
                  {can("datasets:write") ? <th>Acción</th> : null}
                </tr>
              </thead>
              <tbody>
                {dictionary.rows.map((row) => (
                  <tr key={row.column}>
                    <td className="mono"><strong>{row.column}</strong></td>
                    <td className="small">{row.description}</td>
                    <td className="small">{row.unit || "—"}</td>
                    <td className="small">{row.source}</td>
                    {can("datasets:write") ? (
                      <td>
                        <button
                          className="btn btn-sm btn-ghost"
                          onClick={() =>
                            setEditingEntry({
                              column: row.column,
                              description: row.description,
                              unit: row.unit,
                              source: row.source,
                            })
                          }
                        >
                          Editar
                        </button>
                      </td>
                    ) : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Modal>
      ) : null}

      {/* Modal para editar definición de metadatos de variable */}
      {editingEntry ? (
        <Modal
          title={`Editar Metadatos de Variable: ${editingEntry.column}`}
          onClose={() => setEditingEntry(null)}
          footer={
            <>
              <button className="btn" onClick={() => setEditingEntry(null)}>Cancelar</button>
              <button className="btn btn-primary" onClick={saveDictionaryEntry} disabled={savingEdit}>
                {savingEdit ? "Guardando…" : "Guardar Cambios"}
              </button>
            </>
          }
        >
          <div className="field">
            <label>Descripción Científica</label>
            <textarea
              className="textarea"
              value={editingEntry.description}
              onChange={(e) => setEditingEntry({ ...editingEntry, description: e.target.value })}
            />
          </div>
          <div className="field">
            <label>Unidad de Medida</label>
            <input
              className="input"
              value={editingEntry.unit}
              onChange={(e) => setEditingEntry({ ...editingEntry, unit: e.target.value })}
            />
          </div>
          <div className="field">
            <label>Fuente / Origen</label>
            <input
              className="input"
              value={editingEntry.source}
              onChange={(e) => setEditingEntry({ ...editingEntry, source: e.target.value })}
            />
          </div>
          <p className="hint">
            La modificación se registrará en la bitácora criptográfica encadenada de auditoría (M03).
          </p>
        </Modal>
      ) : null}
    </>
  );
}
