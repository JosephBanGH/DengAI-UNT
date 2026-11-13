import { useMemo, useState, type RefObject } from "react";

const PALETTE = ["#0b5d6e", "#17a2b8", "#0f766e", "#f59e0b", "#b3261e", "#6366f1"];

interface Point {
  x: string;
  y: number;
}

export function LineChart({
  series,
  height = 220,
  yLabel,
  showArea = true,
  labelEvery = 8,
  svgRef,
}: {
  series: { name: string; color?: string; points: Point[] }[];
  height?: number;
  yLabel?: string;
  showArea?: boolean;
  labelEvery?: number;
  svgRef?: RefObject<SVGSVGElement>;
}) {
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const width = 720;
  const padding = { top: 12, right: 14, bottom: 26, left: 44 };
  const primary = series[0]?.points ?? [];

  const { maxY, ticks } = useMemo(() => {
    const values = series.flatMap((item) => item.points.map((point) => point.y));
    const max = Math.max(1, ...values);
    const rounded = Math.ceil((max * 1.12) / 5) * 5;
    return { maxY: rounded, ticks: [0, rounded / 4, rounded / 2, (3 * rounded) / 4, rounded] };
  }, [series]);

  if (!primary.length) {
    return <div className="empty-state">Sin serie temporal disponible.</div>;
  }

  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;
  const stepX = primary.length > 1 ? innerW / (primary.length - 1) : innerW;
  const xFor = (index: number) => padding.left + index * stepX;
  const yFor = (value: number) => padding.top + innerH - (value / maxY) * innerH;

  const pathFor = (points: Point[]) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(index).toFixed(2)},${yFor(point.y).toFixed(2)}`).join(" ");

  const areaFor = (points: Point[]) =>
    `${pathFor(points)} L${xFor(points.length - 1).toFixed(2)},${(padding.top + innerH).toFixed(2)} L${padding.left},${(padding.top + innerH).toFixed(2)} Z`;

  return (
    <div style={{ position: "relative" }}>
      <svg ref={svgRef} viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto" }} role="img" aria-label="Serie temporal">
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={yFor(tick)} y2={yFor(tick)} />
            <text className="chart-axis" x={padding.left - 8} y={yFor(tick) + 3} textAnchor="end">
              {Math.round(tick)}
            </text>
          </g>
        ))}
        {series.map((item, seriesIndex) => {
          const color = item.color || PALETTE[seriesIndex % PALETTE.length];
          return (
            <g key={item.name}>
              {showArea && seriesIndex === 0 ? (
                <path d={areaFor(item.points)} fill={color} opacity={0.1} />
              ) : null}
              <path d={pathFor(item.points)} fill="none" stroke={color} strokeWidth={2.2} strokeLinejoin="round" />
            </g>
          );
        })}
        {primary.map((point, index) =>
          index % labelEvery === 0 || index === primary.length - 1 ? (
            <text key={point.x} className="chart-axis" x={xFor(index)} y={height - 8} textAnchor="middle">
              {point.x.length > 7 ? point.x.slice(5) : point.x}
            </text>
          ) : null,
        )}
        {hover ? (
          <line
            className="chart-grid"
            x1={xFor(hover.index)}
            x2={xFor(hover.index)}
            y1={padding.top}
            y2={padding.top + innerH}
            strokeDasharray="3 3"
          />
        ) : null}
        {primary.map((point, index) => (
          <rect
            key={`hit-${point.x}`}
            x={xFor(index) - stepX / 2}
            y={padding.top}
            width={stepX}
            height={innerH}
            fill="transparent"
            onMouseEnter={() => setHover({ index, x: xFor(index), y: yFor(point.y) })}
            onMouseLeave={() => setHover(null)}
          />
        ))}
      </svg>
      {hover ? (
        <div className="map-tooltip" style={{ left: `${(hover.x / width) * 100}%`, top: `${(hover.y / height) * 100}%` }}>
          <strong>{primary[hover.index]?.x}</strong>
          {series.map((item) => (
            <div key={item.name}>
              {item.name}: {item.points[hover.index]?.y ?? "—"}
            </div>
          ))}
        </div>
      ) : null}
      {yLabel ? <div className="hint" style={{ marginTop: 4 }}>{yLabel}</div> : null}
    </div>
  );
}

export function BarList({
  items,
  valueFormatter,
  max,
}: {
  items: { label: string; value: number; hint?: string; color?: string }[];
  valueFormatter?: (value: number) => string;
  max?: number;
}) {
  const top = max ?? Math.max(1, ...items.map((item) => item.value));
  if (!items.length) return <div className="empty-state">Sin datos para graficar.</div>;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
      {items.map((item, index) => (
        <div key={`${item.label}-${index}`}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 3 }}>
            <span className="truncate" style={{ maxWidth: "70%" }} title={item.label}>
              {item.label}
            </span>
            <strong>{valueFormatter ? valueFormatter(item.value) : item.value}</strong>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${Math.max(2, (item.value / top) * 100)}%`,
                background: item.color || PALETTE[index % PALETTE.length],
              }}
            />
          </div>
          {item.hint ? <div className="hint" style={{ marginTop: 2 }}>{item.hint}</div> : null}
        </div>
      ))}
    </div>
  );
}

export function Donut({
  items,
  size = 168,
  svgRef,
}: {
  items: { label: string; value: number; color?: string }[];
  size?: number;
  svgRef?: RefObject<SVGSVGElement>;
}) {
  const total = items.reduce((sum, item) => sum + item.value, 0);
  if (!total) return <div className="empty-state">Sin distribución disponible.</div>;
  const radius = size / 2 - 12;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
      <svg ref={svgRef} width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Distribución">
        <g transform={`translate(${size / 2},${size / 2}) rotate(-90)`}>
          {items.map((item, index) => {
            const fraction = item.value / total;
            const dash = fraction * circumference;
            const circle = (
              <circle
                key={item.label}
                r={radius}
                fill="none"
                stroke={item.color || PALETTE[index % PALETTE.length]}
                strokeWidth={16}
                strokeDasharray={`${dash} ${circumference - dash}`}
                strokeDashoffset={-offset}
                strokeLinecap="butt"
              />
            );
            offset += dash;
            return circle;
          })}
        </g>
        <text x={size / 2} y={size / 2 - 2} textAnchor="middle" fontSize="19" fontWeight="700" fill="#10243b">
          {total}
        </text>
        <text x={size / 2} y={size / 2 + 16} textAnchor="middle" fontSize="10" fill="#8494a8">
          TOTAL
        </text>
      </svg>
      <div className="legend" style={{ flexDirection: "column", alignItems: "flex-start", gap: 6 }}>
        {items.map((item, index) => (
          <span key={item.label} className="legend-item">
            <span className="legend-swatch" style={{ background: item.color || PALETTE[index % PALETTE.length] }} />
            {item.label}: <strong style={{ marginLeft: 4 }}>{item.value}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

export function Sparkline({ points, color = "#0b5d6e" }: { points: number[]; color?: string }) {
  if (!points.length) return null;
  const width = 120;
  const height = 30;
  const max = Math.max(1, ...points);
  const step = points.length > 1 ? width / (points.length - 1) : width;
  const path = points
    .map((value, index) => `${index === 0 ? "M" : "L"}${(index * step).toFixed(1)},${(height - (value / max) * height).toFixed(1)}`)
    .join(" ");
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
      <path d={path} fill="none" stroke={color} strokeWidth={1.8} />
    </svg>
  );
}
