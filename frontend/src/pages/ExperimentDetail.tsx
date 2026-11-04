import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { Experiment } from "../api/types";
import { PageHeader } from "../components/Layout";
import {
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  ErrorAlert,
  Loading,
  Progress,
  StatusBadge,
  formatDate,
  formatNumber,
} from "../components/ui";
import { describeError } from "../context/ToastContext";

const METRIC_ORDER = ["MAE", "RMSE", "MAPE", "MASE", "coverage", "WIS", "AUC_PR", "sensitivity", "specificity", "Brier"];

export function ExperimentDetailPage() {
  const { id } = useParams();
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = () =>
      api
        .get<Experiment>(`/ml/experiments/${id}`)
        .then((data) => {
          if (!cancelled) {
            setExperiment(data);
            setLoading(false);
          }
        })
        .catch((caught) => {
          if (!cancelled) {
            setError(describeError(caught));
            setLoading(false);
          }
        });
    void fetchOnce();
    const timer = window.setInterval(() => {
      void fetchOnce();
    }, 3500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [id]);

  if (loading) return <Loading label="Cargando experimento…" />;
  if (error) return <ErrorAlert message={error} />;
  if (!experiment) return <EmptyState title="Experimento no encontrado" />;

  const horizons = experiment.metrics ? Object.keys(experiment.metrics) : [];

  return (
    <>
      <PageHeader
        title={experiment.name}
        description={`Experimento ${experiment.run_id} · ${experiment.algorithm} · validación ${experiment.validation_strategy}`}
        actions={
          <>
            <StatusBadge status={experiment.status} />
            <Link className="btn" to="/laboratorio">Volver al laboratorio</Link>
          </>
        }
      />

      <div className="grid grid-main" style={{ marginBottom: 14 }}>
        <Card>
          <CardHeader title="Estado de la ejecución" subtitle={experiment.stage} />
          <CardBody>
            <Progress value={experiment.progress} />
            <div className="grid grid-3" style={{ marginTop: 14 }}>
              <Info label="Trials completados" value={`${experiment.trials_completed}/${experiment.n_trials}`} />
              <Info label="Inicio" value={formatDate(experiment.started_at, true)} />
              <Info label="Fin" value={experiment.finished_at ? formatDate(experiment.finished_at, true) : "en curso"} />
              <Info label="Duración" value={experiment.duration_seconds ? `${experiment.duration_seconds.toFixed(1)} s` : "—"} />
              <Info label="Creado por" value={experiment.created_by} />
              <Info label="Datos" value={experiment.is_synthetic_data ? "Sintéticos" : "Fuente externa"} />
            </div>
            {experiment.error_message ? <div className="alert alert-error" style={{ marginTop: 12 }}>{experiment.error_message}</div> : null}
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Registro de experimento" subtitle="Trazabilidad MLflow / interno" />
          <CardBody>
            <Info label="run_id" value={experiment.run_id} mono />
            <Info label="MLflow run" value={experiment.mlflow_run_id || "registro interno (MLflow no instalado)"} mono />
            <Info label="Horizontes" value={JSON.stringify(experiment.horizons)} />
            <p className="hint" style={{ marginTop: 10 }}>
              La optimización se ejecuta estrictamente sobre entrenamiento y validación interna; el conjunto de
              prueba se evalúa una única vez al final.
            </p>
          </CardBody>
        </Card>
      </div>

      {horizons.length ? (
        horizons.map((horizon) => {
          const payload = experiment.metrics?.[horizon];
          if (!payload) return null;
          return (
            <Card key={horizon} style={{ marginBottom: 14 }}>
              <CardHeader
                title={`Horizonte ${horizon} semana(s)`}
                subtitle={`Entrenamiento ${payload.n_train} · Validación ${payload.n_val} · Prueba ${payload.n_test} observaciones`}
              />
              <CardBody>
                <div className="grid grid-3">
                  {METRIC_ORDER.filter((metric) => payload.metrics[metric] !== undefined && payload.metrics[metric] !== null).map((metric) => (
                    <div key={metric} className="card card-pad">
                      <div className="kpi-label">{metric.replace("_", "-")}</div>
                      <div className="kpi-value" style={{ fontSize: 20 }}>
                        {formatNumber(payload.metrics[metric] as number, metric === "coverage" || metric === "AUC_PR" || metric === "Brier" || metric === "sensitivity" || metric === "specificity" ? 3 : 2)}
                      </div>
                    </div>
                  ))}
                </div>
                <h3 style={{ marginTop: 16, marginBottom: 8, fontSize: 13 }}>Intervalos de confianza bootstrap (MAE)</h3>
                <p className="small">
                  [{formatNumber(payload.metrics.MAE_ci_lower ?? null, 3)}, {formatNumber(payload.metrics.MAE_ci_upper ?? null, 3)}]
                </p>
                <h3 style={{ marginTop: 16, marginBottom: 8, fontSize: 13 }}>Mejores hiperparámetros</h3>
                <pre className="mono" style={{ background: "var(--surface-2)", padding: 12, borderRadius: 8, overflow: "auto", fontSize: 11.5 }}>
                  {JSON.stringify(payload.best_params, null, 2)}
                </pre>
                <h3 style={{ marginTop: 16, marginBottom: 8, fontSize: 13 }}>Particiones</h3>
                <pre className="mono" style={{ background: "var(--surface-2)", padding: 12, borderRadius: 8, overflow: "auto", fontSize: 11.5 }}>
                  {JSON.stringify({ ...payload.notes, district_grid: undefined }, null, 2)}
                </pre>
              </CardBody>
            </Card>
          );
        })
      ) : (
        <Card>
          <CardBody>
            <EmptyState title="Sin métricas todavía" description="El experimento aún no ha finalizado o falló antes de la evaluación." />
          </CardBody>
        </Card>
      )}
    </>
  );
}

function Info({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div className="kpi-label">{label}</div>
      <div className={mono ? "mono" : "small"} style={{ wordBreak: "break-all" }}>{value}</div>
    </div>
  );
}
