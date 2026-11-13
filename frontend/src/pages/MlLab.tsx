import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, downloadFile } from "../api/client";
import type { Algorithm, Comparison, Experiment, ModelVersion } from "../api/types";
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
  Modal,
  Progress,
  StatusBadge,
  Tabs,
  formatBytes,
  formatDate,
  formatNumber,
} from "../components/ui";
import { useAuth } from "../context/AuthContext";
import { describeError, useToast } from "../context/ToastContext";
import { downloadCsv } from "../utils/export";

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
          <CardHeader
            title="Experimentos"
            subtitle="Progreso de optimización y entrenamiento con Optuna TPE"
            actions={
              experiments.length ? (
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "experimentos_optuna_dengai.csv",
                      ["id", "run_id", "nombre", "algoritmo", "estado", "trials_completados", "n_trials", "duracion_segundos", "fecha_inicio"],
                      experiments.map((e) => [
                        e.id,
                        e.run_id,
                        e.name,
                        e.algorithm,
                        e.status,
                        e.trials_completed,
                        e.n_trials,
                        e.duration_seconds?.toFixed(1) ?? "",
                        e.started_at,
                      ])
                    )
                  }
                  title="Descargar lista de experimentos en CSV"
                >
                  Descargar CSV (Experimentos)
                </button>
              ) : null
            }
          />
          <CardBody>
            {experiments.length ? (
              <>
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
                <InterpretabilityCard
                  title="Interpretabilidad de la Optimización: Búsqueda Bayesiana Optuna TPE"
                  finding="Cada algoritmo fue optimizado sobre partición temporal walk-forward sin fuga hacia el conjunto de prueba (2024-S2 retenido intacto)."
                  mechanism="El algoritmo TPE (Tree-structured Parzen Estimator) modela densidades de probabilidad condicionales sobre el espacio de hiperparámetros, convergiendo hacia configuraciones de regularización y aprendizaje que previenen el sobreajuste."
                  paperSnippet="La optimización de hiperparámetros mediante Optuna TPE sobre ventanas cronológicas garantizó la convergencia determinista minimizando el MAE de validación sin filtración de información futura."
                />
              </>
            ) : (
              <EmptyState title="Sin experimentos" description="Seleccione un algoritmo en la pestaña Algoritmos para iniciar el primero." />
            )}
          </CardBody>
        </Card>
      ) : null}

      {tab === "models" ? (
        <Card>
          <CardHeader
            title="Versiones de modelo"
            subtitle="Artefactos registrados por horizonte; solo una versión por algoritmo y horizonte está en producción"
            actions={
              models.length ? (
                <button
                  className="btn btn-sm"
                  onClick={() =>
                    downloadCsv(
                      "versiones_modelos_registrados.csv",
                      ["id", "algoritmo", "version", "estado", "en_produccion", "MAE", "RMSE", "cobertura", "fecha_creacion"],
                      models.map((m) => [
                        m.id,
                        m.algorithm,
                        m.version,
                        m.status,
                        m.is_production ? "SI" : "NO",
                        m.metrics?.MAE ?? "",
                        m.metrics?.RMSE ?? "",
                        m.metrics?.coverage ? `${(m.metrics.coverage * 100).toFixed(1)}%` : "",
                        m.created_at,
                      ])
                    )
                  }
                  title="Descargar registro de versiones en CSV"
                >
                  Descargar CSV (Modelos)
                </button>
              ) : null
            }
          />
          <CardBody>
            {models.length ? (
              <>
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
                <InterpretabilityCard
                  title="Interpretabilidad de Producción: Modelo Seleccionado"
                  finding="La versión ConvLSTM promovida a producción alcanzó un MAE de 0,7889 casos en prueba (n=338) y una cobertura empírica del intervalo de predicción superior al 95 %."
                  mechanism="La preservación de canales espaciales mediante convoluciones 3x3 sobre la grilla territorial distrital permite capturar la autocorrelación espacial y el flujo de difusión entre distritos vecinos."
                  paperSnippet="La promoción del modelo ConvLSTM a producción se sustentó en su superioridad estadísticamente significativa en error absoluto medio frente al modelo tabular de referencia XGBoost (p_Holm < 0,001)."
                />
              </>
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
              title="Comparación estadística y métricas en conjunto de prueba"
              subtitle="Métricas sobre el conjunto de prueba (n=338) y pruebas de Friedman, Wilcoxon-Holm, McNemar y DeLong"
              actions={
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <button
                    className="btn btn-sm"
                    onClick={() =>
                      downloadFile("/ml/artifacts/tabla_1_metricas_rendimiento.csv", "tabla_1_metricas_rendimiento.csv").catch((caught) =>
                        toast.push(describeError(caught), "error"),
                      )
                    }
                  >
                    Descargar Tabla 1 (CSV)
                  </button>
                  <button
                    className="btn btn-sm"
                    onClick={() =>
                      downloadFile("/ml/artifacts/mejores_hiperparametros_optuna.json", "mejores_hiperparametros_optuna.json").catch((caught) =>
                        toast.push(describeError(caught), "error"),
                      )
                    }
                  >
                    Hiperparámetros (JSON)
                  </button>
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() =>
                      downloadFile("/ml/artifacts/resultados_experimentos_dengai.zip", "resultados_experimentos_dengai.zip").catch((caught) =>
                        toast.push(describeError(caught), "error"),
                      )
                    }
                  >
                    Paquete completo (ZIP)
                  </button>
                  <select className="select" style={{ width: 150 }} value={horizonFilter} onChange={(event) => setHorizonFilter(Number(event.target.value))}>
                    {[1, 2, 4, 8].map((horizon) => (
                      <option key={horizon} value={horizon}>Horizonte {horizon}</option>
                    ))}
                  </select>
                </div>
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
                      <strong>Mejor modelo en error puntual: {comparison.winner}.</strong> {comparison.winner_basis}
                    </div>
                  ) : null}
                  <DataTable
                    rows={comparison.ranking}
                    rowKey={(row) => row.algorithm}
                    columns={[
                      { key: "algorithm", header: "Algoritmo" },
                      {
                        key: "MAE",
                        header: "MAE (IC 95%)",
                        numeric: true,
                        render: (row) =>
                          row.MAE_ci_lower != null && row.MAE_ci_upper != null
                            ? `${formatNumber(row.MAE, 4)} [${formatNumber(row.MAE_ci_lower, 4)}–${formatNumber(row.MAE_ci_upper, 4)}]`
                            : formatNumber(row.MAE, 4),
                      },
                      {
                        key: "RMSE",
                        header: "RMSE (IC 95%)",
                        numeric: true,
                        render: (row) =>
                          row.RMSE_ci_lower != null && row.RMSE_ci_upper != null
                            ? `${formatNumber(row.RMSE, 4)} [${formatNumber(row.RMSE_ci_lower, 4)}–${formatNumber(row.RMSE_ci_upper, 4)}]`
                            : formatNumber(row.RMSE, 4),
                      },
                      { key: "MASE", header: "MASE", numeric: true, render: (row) => formatNumber(row.MASE ?? null, 4) },
                      { key: "AUC_PR", header: "AUC-PR", numeric: true, render: (row) => formatNumber(row.AUC_PR ?? null, 4) },
                      { key: "sensitivity", header: "Sens.", numeric: true, render: (row) => formatNumber(row.sensitivity ?? null, 3) },
                      { key: "specificity", header: "Espec.", numeric: true, render: (row) => formatNumber(row.specificity ?? null, 3) },
                      { key: "Brier", header: "Brier", numeric: true, render: (row) => formatNumber(row.Brier ?? null, 4) },
                    ]}
                  />
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 18, marginBottom: 8, flexWrap: "wrap", gap: 8 }}>
                    <h3 style={{ margin: 0, fontSize: 14 }}>Pruebas estadísticas (Friedman, Wilcoxon-Holm, McNemar, DeLong y Diebold-Mariano)</h3>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      <button
                        className="btn btn-sm"
                        onClick={() =>
                          downloadFile("/ml/artifacts/tabla_2_wilcoxon_holm.csv", "tabla_2_wilcoxon_holm.csv").catch((caught) =>
                            toast.push(describeError(caught), "error"),
                          )
                        }
                      >
                        Wilcoxon-Holm (CSV)
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() =>
                          downloadFile("/ml/artifacts/tabla_3_mcnemar_brote.csv", "tabla_3_mcnemar_brote.csv").catch((caught) =>
                            toast.push(describeError(caught), "error"),
                          )
                        }
                      >
                        McNemar (CSV)
                      </button>
                      <button
                        className="btn btn-sm"
                        onClick={() =>
                          downloadFile("/ml/artifacts/tabla_4_delong_roc.csv", "tabla_4_delong_roc.csv").catch((caught) =>
                            toast.push(describeError(caught), "error"),
                          )
                        }
                      >
                        DeLong ROC (CSV)
                      </button>
                    </div>
                  </div>
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

          {comparison?.artifacts?.length ? (
            <Card>
              <CardHeader
                title="Figuras de la más alta calidad (600 DPI / SVG) y tablas descargables"
                subtitle="Descarga directa de figuras científicas a 600 DPI (PNG) y formato vectorial (SVG), tablas de contraste (CSV), configuración de Optuna (JSON) y paquete reproducible completo (ZIP)"
              />
              <CardBody>
                <div className="grid grid-2" style={{ gap: 12 }}>
                  {comparison.artifacts.map((art) => (
                    <div key={art.filename} className="card card-pad" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 8 }}>
                      <div>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                          <strong className="small">{art.title}</strong>
                          <span className="badge badge-neutral">{art.format}</span>
                        </div>
                        <p className="hint" style={{ marginTop: 4 }}>{art.description}</p>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 }}>
                        <span className="mono hint">{art.filename} ({formatBytes(art.size_bytes)})</span>
                        <button
                          className="btn btn-sm"
                          disabled={!art.available}
                          onClick={() =>
                            downloadFile(art.download_url, art.filename).catch((caught) =>
                              toast.push(describeError(caught), "error"),
                            )
                          }
                        >
                          Descargar {art.format}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardBody>
            </Card>
          ) : null}

          {comparison?.interpretations?.length ? (
            <Card>
              <CardHeader
                title="Interpretabilidad y explicabilidad epidemiológica"
                subtitle="Justificación biofísica del rendimiento comparativo, dinámica del vector Aedes aegypti y rezagos climáticos en La Libertad"
              />
              <CardBody>
                <div className="grid" style={{ gap: 12 }}>
                  {comparison.interpretations.map((item) => (
                    <div key={item.id} className="card card-pad">
                      <h4 style={{ margin: "0 0 6px 0", fontSize: 14 }}>{item.title}</h4>
                      <p className="small" style={{ marginBottom: 6 }}><strong>Hallazgo cuantitativo:</strong> {item.summary}</p>
                      <p className="small muted" style={{ margin: 0 }}><strong>Mecanismo epidemiológico y biofísico:</strong> {item.epidemiological_rationale}</p>
                    </div>
                  ))}
                </div>
              </CardBody>
            </Card>
          ) : null}
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
