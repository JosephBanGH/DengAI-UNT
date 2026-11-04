import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Algorithm, Comparison, Experiment, ModelVersion } from "../api/types";
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
  Progress,
  StatusBadge,
  Tabs,
  formatDate,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";

type Tab = "algorithms" | "experiments" | "models" | "comparison";

export function MlLabPage() {
  const { can } = useAuth();
  const toast = useToast();
  const [tab, setTab] = useState<Tab>("algorithms");
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [backend, setBackend] = useState<any>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState<Algorithm | null>(null);
  const [nTrials, setNTrials] = useState(8);
  const [horizons, setHorizons] = useState<number[]>([1]);
  const [strategy, setStrategy] = useState("walk_forward");
  const [horizonFilter, setHorizonFilter] = useState(1);

  const loadAll = () => {
    Promise.all([
      api.get<Algorithm[]>("/ml/algorithms"),
      api.get<any>("/ml/backend"),
      api.get<{ items: Experiment[] }>("/ml/experiments", { size: 25 }),
      api.get<ModelVersion[]>("/ml/models"),
    ])
      .then(([algorithmData, backendData, experimentData, modelData]) => {
        setAlgorithms(algorithmData);
        setBackend(backendData);
        setExperiments(experimentData.items);
        setModels(modelData);
      })
      .catch((caught) => setError(describeError(caught)))
      .finally(() => setLoading(false));
  };

  useEffect(loadAll, []);

  // Sondeo del progreso mientras haya experimentos activos.
  useEffect(() => {
    const active = experiments.some((experiment) => ["RUNNING", "QUEUED"].includes(experiment.status));
    if (!active) return;
    const timer = window.setInterval(async () => {
      try {
        const data = await api.get<{ items: Experiment[] }>("/ml/experiments", { size: 25 });
        setExperiments(data.items);
      } catch {
        /* el sondeo no debe romper la vista */
      }
    }, 4000);
    return () => window.clearInterval(timer);
  }, [experiments]);

  useEffect(() => {
    if (tab !== "comparison") return;
    setComparisonError(null);
    api
      .get<Comparison>("/ml/compare", { horizon: horizonFilter })
      .then(setComparison)
      .catch((caught) => setComparisonError(describeError(caught)));
  }, [tab, horizonFilter, experiments]);

  const launch = async () => {
    if (!launching) return;
    try {
      await api.post("/ml/experiments", {
        algorithm: launching.key,
        n_trials: nTrials,
        horizons,
        validation_strategy: strategy,
      });
      toast.push(`Experimento ${launching.name} en ejecución.`, "success");
      setLaunching(null);
      const data = await api.get<{ items: Experiment[] }>("/ml/experiments", { size: 25 });
      setExperiments(data.items);
      setTab("experiments");
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  const promote = async (model: ModelVersion) => {
    try {
      const result = await api.post<{ message: string }>(`/ml/models/${model.id}/promote`);
      toast.push(result.message, "success");
      loadAll();
    } catch (caught) {
      toast.push(describeError(caught), "error");
    }
  };

  if (loading) return <Loading label="Cargando laboratorio de IA…" />;

  return (
    <>
      <PageHeader
        title="Laboratorio de modelos de inteligencia artificial"
        description="Entrenamiento de los cinco algoritmos del protocolo, optimización con Optuna (TPE + Median Pruner), validación temporal y geográfica, y promoción de versiones a producción."
        actions={
          <span className="chip">
            Backend: {backend?.torch?.available ? `PyTorch ${backend.torch.version}` : "sin PyTorch"} · XGBoost{" "}
            {backend?.xgboost?.version}
          </span>
        }
      />

      {error ? <ErrorAlert message={error} /> : null}

      <Tabs
        tabs={[
          { key: "algorithms", label: "Algoritmos" },
          { key: "experiments", label: "Experimentos" },
          { key: "models", label: "Versiones de modelo" },
          { key: "comparison", label: "Comparación estadística" },
        ]}
        active={tab}
        onChange={(key) => setTab(key as Tab)}
      />

      {tab === "algorithms" ? (
        <div className="grid grid-3">
          {algorithms.map((algorithm) => (
            <Card key={algorithm.key}>
              <CardHeader
                title={algorithm.name}
                subtitle={algorithm.role}
                actions={<StatusBadge status={algorithm.status} />}
              />
              <CardBody>
                <p className="small" style={{ minHeight: 54 }}>{algorithm.description}</p>
                <div className="legend" style={{ marginBottom: 12 }}>
                  {algorithm.temporal ? <span className="chip">Temporal</span> : null}
                  {algorithm.spatial ? <span className="chip">Espacial</span> : null}
                  {algorithm.attention ? <span className="chip">Atención</span> : null}
                  <span className="chip">{algorithm.backend}</span>
                </div>
                {algorithm.status !== "READY" ? (
                  <p className="form-error">{algorithm.status_reason}</p>
                ) : null}
                <button
                  className="btn btn-primary btn-sm"
                  style={{ width: "100%" }}
                  disabled={!can("ml:run") || algorithm.status !== "READY"}
                  onClick={() => {
                    setLaunching(algorithm);
                    setHorizons(algorithm.default_horizons.slice(0, 1));
                    setNTrials(algorithm.key === "xgboost" ? 10 : 4);
                  }}
                >
                  Entrenar
                </button>
              </CardBody>
            </Card>
          ))}
        </div>
      ) : null}

      {tab === "experiments" ? (
        <Card>
          <CardHeader title="Experimentos" subtitle="Progreso de optimización y entrenamiento" />
          <CardBody>
            {experiments.length ? (
              <DataTable
                rows={experiments}
                rowKey={(row) => row.id}
                columns={[
                  { key: "name", header: "Experimento", render: (row) => (
                    <div>
                      <Link to={`/laboratorio/${row.id}`}><strong>{row.name}</strong></Link>
                      <div className="hint mono">{row.run_id}</div>
                    </div>
                  ) },
                  { key: "algorithm", header: "Algoritmo" },
                  { key: "status", header: "Estado", render: (row) => (
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <StatusBadge status={row.status} />
                    </div>
                  ) },
                  { key: "progress", header: "Progreso", render: (row) => (
                    <div style={{ minWidth: 120 }}>
                      <Progress value={row.progress} />
                      <span className="hint">{row.stage}</span>
                    </div>
                  ) },
                  { key: "trials", header: "Trials", numeric: true, render: (row) => `${row.trials_completed}/${row.n_trials}` },
                  { key: "mae", header: "MAE (h1)", numeric: true, render: (row) => formatNumber(row.metrics?.["1"]?.metrics?.MAE ?? null, 3) },
                  { key: "duration", header: "Duración", numeric: true, render: (row) => (row.duration_seconds ? `${row.duration_seconds.toFixed(0)} s` : "—") },
                  { key: "started", header: "Inicio", render: (row) => formatDate(row.started_at, true) },
                ]}
              />
            ) : (
              <EmptyState title="Sin experimentos" description="Seleccione un algoritmo en la pestaña Algoritmos para iniciar el primero." />
            )}
          </CardBody>
        </Card>
      ) : null}

      {tab === "models" ? (
        <Card>
          <CardHeader title="Versiones de modelo" subtitle="Artefactos registrados por horizonte; solo una versión por algoritmo y horizonte está en producción" />
          <CardBody>
            {models.length ? (
              <DataTable
                rows={models}
                rowKey={(row) => row.id}
                columns={[
                  { key: "algorithm", header: "Algoritmo" },
                  { key: "version", header: "Versión", render: (row) => <span className="mono">{row.version}</span> },
                  { key: "status", header: "Estado", render: (row) => <StatusBadge status={row.status} /> },
                  { key: "production", header: "Producción", render: (row) => (row.is_production ? <span className="badge badge-ok">Activo</span> : <span className="muted small">—</span>) },
                  { key: "MAE", header: "MAE", numeric: true, render: (row) => formatNumber(row.metrics?.MAE ?? null, 3) },
                  { key: "RMSE", header: "RMSE", numeric: true, render: (row) => formatNumber(row.metrics?.RMSE ?? null, 3) },
                  { key: "coverage", header: "Cobertura", numeric: true, render: (row) => formatNumber((row.metrics?.coverage ?? 0) * 100, 1) + "%" },
                  { key: "created", header: "Creado", render: (row) => formatDate(row.created_at, true) },
                  {
                    key: "actions",
                    header: "",
                    render: (row) =>
                      can("ml:promote") && !row.is_production ? (
                        <button className="btn btn-sm" onClick={() => promote(row)}>Promover</button>
                      ) : null,
                  },
                ]}
              />
            ) : (
              <EmptyState title="Sin modelos" description="Entrene un experimento para generar la primera versión." />
            )}
          </CardBody>
        </Card>
      ) : null}

      {tab === "comparison" ? (
        <div className="grid" style={{ gap: 14 }}>
          <Card>
            <CardHeader
              title="Comparación de modelos"
              subtitle="Métricas sobre el conjunto de prueba y pruebas de Friedman, Wilcoxon-Holm y Diebold-Mariano"
              actions={
                <select className="select" style={{ width: 150 }} value={horizonFilter} onChange={(event) => setHorizonFilter(Number(event.target.value))}>
                  {[1, 2, 4, 8].map((horizon) => (
                    <option key={horizon} value={horizon}>Horizonte {horizon}</option>
                  ))}
                </select>
              }
            />
            <CardBody>
              {comparisonError ? (
                <div className="alert alert-warn">{comparisonError}</div>
              ) : !comparison ? (
                <Loading />
              ) : (
                <>
                  {comparison.winner ? (
                    <div className="alert alert-info">
                      <strong>Mejor modelo: {comparison.winner}.</strong> {comparison.winner_basis}
                    </div>
                  ) : null}
                  <DataTable
                    rows={comparison.ranking}
                    rowKey={(row) => row.algorithm}
                    columns={[
                      { key: "algorithm", header: "Algoritmo" },
                      { key: "MAE", header: "MAE", numeric: true, render: (row) => formatNumber(row.MAE, 4) },
                      { key: "RMSE", header: "RMSE", numeric: true, render: (row) => formatNumber(row.RMSE, 4) },
                    ]}
                  />
                  <h3 style={{ marginTop: 18, marginBottom: 8, fontSize: 14 }}>Pruebas estadísticas</h3>
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr>
                          <th>Prueba</th>
                          <th>Comparación</th>
                          <th className="numeric">Estadístico</th>
                          <th className="numeric">p</th>
                          <th>Conclusión</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparison.tests.map((test, index) => (
                          <tr key={index}>
                            <td>{test.test_name}</td>
                            <td className="small">{test.comparison || "global"}</td>
                            <td className="numeric">{test.statistic != null ? formatNumber(test.statistic, 3) : "—"}</td>
                            <td className="numeric">
                              {test.p_value_adjusted != null
                                ? formatNumber(test.p_value_adjusted, 4)
                                : test.p_value != null
                                  ? formatNumber(test.p_value, 4)
                                  : "—"}
                            </td>
                            <td className="small">
                              {test.applicable === false
                                ? test.reason
                                : test.significant_holm ?? test.significant
                                  ? "Diferencia significativa"
                                  : "Sin diferencia significativa"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="hint" style={{ marginTop: 10 }}>{comparison.caveat}</p>
                </>
              )}
            </CardBody>
          </Card>
        </div>
      ) : null}

      {launching ? (
        <Modal
          title={`Entrenar ${launching.name}`}
          onClose={() => setLaunching(null)}
          footer={
            <>
              <button className="btn" onClick={() => setLaunching(null)}>Cancelar</button>
              <button className="btn btn-primary" onClick={launch}>Iniciar experimento</button>
            </>
          }
        >
          <p className="small muted" style={{ marginBottom: 12 }}>{launching.description}</p>
          <div className="form-row">
            <div className="field">
              <label>Trials de Optuna</label>
              <input className="input" type="number" min={1} max={150} value={nTrials} onChange={(event) => setNTrials(Number(event.target.value))} />
              <span className="hint">El protocolo exige entre 50 y 150 trials; en esta demostración se permiten menos por presupuesto de CPU.</span>
            </div>
            <div className="field">
              <label>Estrategia de validación</label>
              <select className="select" value={strategy} onChange={(event) => setStrategy(event.target.value)}>
                <option value="walk_forward">Temporal walk-forward</option>
                <option value="geographic">Geográfica (distritos no vistos)</option>
              </select>
            </div>
          </div>
          <div className="field">
            <label>Horizontes (semana(s))</label>
            <div style={{ display: "flex", gap: 8 }}>
              {[1, 2, 4, 8].map((horizon) => (
                <label key={horizon} className="chip" style={{ cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={horizons.includes(horizon)}
                    onChange={(event) => {
                      setHorizons((current) =>
                        event.target.checked ? [...current, horizon].sort((a, b) => a - b) : current.filter((value) => value !== horizon),
                      );
                    }}
                  />
                  h={horizon}
                </label>
              ))}
            </div>
          </div>
          <p className="hint">
            El conjunto de prueba no participa en la selección de hiperparámetros. El entrenamiento final usa
            entrenamiento + validación y se evalúa una sola vez sobre prueba.
          </p>
        </Modal>
      ) : null}
    </>
  );
}
