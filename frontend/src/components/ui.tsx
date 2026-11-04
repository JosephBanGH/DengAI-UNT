import type { CSSProperties, ReactNode } from "react";
import { Icon } from "./Icons";

export function Card({ children, className = "", style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  return <section className={`card ${className}`} style={style}>{children}</section>;
}

export function CardHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="card-header">
      <div>
        <div className="card-title">{title}</div>
        {subtitle ? <div className="card-subtitle">{subtitle}</div> : null}
      </div>
      {actions ? <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>{actions}</div> : null}
    </header>
  );
}

export function CardBody({ children, className = "", style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  return <div className={`card-body ${className}`} style={style}>{children}</div>;
}

export function KpiCard({
  label,
  value,
  unit,
  hint,
  changePct,
  trend,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  hint?: string | null;
  changePct?: number | null;
  trend?: string | null;
}) {
  const trendClass = trend === "up" ? "up" : trend === "down" ? "down" : "flat";
  return (
    <div className="card kpi">
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">
        {value}
        {unit ? <span className="kpi-unit">{unit}</span> : null}
      </span>
      {typeof changePct === "number" ? (
        <span className={`kpi-delta ${trendClass}`}>
          {trend === "up" ? "▲" : trend === "down" ? "▼" : "■"} {Math.abs(changePct).toFixed(1)}% vs. periodo previo
        </span>
      ) : null}
      {hint ? <span className="kpi-hint">{hint}</span> : null}
    </div>
  );
}

const RISK_LABEL: Record<string, string> = {
  CRITICAL: "Crítico",
  HIGH: "Alto",
  MODERATE: "Medio",
  LOW: "Bajo",
  UNKNOWN: "Sin dato",
};

export function RiskBadge({ level }: { level: string }) {
  const key = (level || "UNKNOWN").toUpperCase();
  return (
    <span className={`badge badge-${key.toLowerCase()}`}>
      <span className="badge-dot" />
      {RISK_LABEL[key] || key}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const value = (status || "").toUpperCase();
  const map: Record<string, string> = {
    COMPLETED: "ok",
    READY: "ok",
    SUCCESS: "ok",
    PRODUCTION: "ok",
    ACTIVE: "ok",
    OPEN: "danger",
    FAILED: "danger",
    ERROR: "danger",
    DENIED: "danger",
    CRITICAL: "danger",
    RUNNING: "info",
    QUEUED: "neutral",
    PENDING: "neutral",
    SKIPPED: "warn",
    ACKNOWLEDGED: "warn",
    CLOSED: "neutral",
    CANDIDATE: "info",
    ARCHIVED: "neutral",
  };
  return <span className={`badge badge-${map[value] || "neutral"}`}>{status}</span>;
}

export function Loading({ label = "Cargando información…" }: { label?: string }) {
  return (
    <div className="loading-block">
      <span className="spinner" />
      {label}
    </div>
  );
}

export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      {description}
    </div>
  );
}

export function ErrorAlert({ message }: { message: string }) {
  return <div className="alert alert-error">{message}</div>;
}

export function Progress({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="progress-track" title={`${pct}%`}>
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: T; label: string }[];
  active: T;
  onChange: (key: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          role="tab"
          aria-selected={active === tab.key}
          className={`tab ${active === tab.key ? "active" : ""}`}
          onClick={() => onChange(tab.key)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function Modal({
  title,
  children,
  onClose,
  footer,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  footer?: ReactNode;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className="modal" onClick={(event) => event.stopPropagation()} role="dialog" aria-label={title}>
        <header className="modal-header">
          <h2 style={{ fontSize: 16 }}>{title}</h2>
          <button className="btn btn-ghost btn-sm" style={{ marginLeft: "auto" }} onClick={onClose} aria-label="Cerrar">
            <Icon name="close" />
          </button>
        </header>
        <div className="modal-body">{children}</div>
        {footer ? <footer className="modal-footer">{footer}</footer> : null}
      </div>
    </div>
  );
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  empty = "Sin registros para mostrar.",
}: {
  columns: { key: string; header: string; numeric?: boolean; render?: (row: T) => ReactNode }[];
  rows: T[];
  rowKey: (row: T, index: number) => string | number;
  empty?: string;
}) {
  if (!rows.length) return <EmptyState title="Sin datos" description={empty} />;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.numeric ? "numeric" : ""}>
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={rowKey(row, index)}>
              {columns.map((column) => (
                <td key={column.key} className={column.numeric ? "numeric" : ""}>
                  {column.render ? column.render(row) : String((row as Record<string, unknown>)[column.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("es-PE", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatInteger(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return Math.round(value).toLocaleString("es-PE");
}

export function formatDate(value: string | null | undefined, withTime = false): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("es-PE", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  });
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export function toQuery(params: Record<string, string | number | undefined | null>): string {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    search.set(key, String(value));
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}
