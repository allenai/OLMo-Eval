import { scaleLinear } from "d3-scale";
import { line as d3line, area as d3area } from "d3-shape";
import { type ReactNode, useRef, useState } from "react";
import { ChartTooltip, niceTicks, TipRow, useSize, useTooltip } from "./core";
import c from "./charts.module.css";

export interface MPoint {
  x: number;
  y: number | null;
  lo?: number | null;
  hi?: number | null;
  id?: string;
}

export interface MSeries {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
  points: MPoint[];
  width?: number;
}

export interface MRef {
  key: string;
  label: string;
  value: number | null;
  color?: string;
}

export interface MPanel {
  key: string;
  title: string;
  subtitle?: string;
  series: MSeries[];
  refs?: MRef[];
}

export function SmallMultiples({
  panels,
  formatX = (v) => String(v),
  formatY = (v) => v.toFixed(1),
  minPanelWidth = 230,
  panelHeight = 150,
  onPointClick,
  onBrush,
  brush,
  emptyText = "No results for this task",
  xLabel,
}: {
  panels: MPanel[];
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  minPanelWidth?: number;
  panelHeight?: number;
  onPointClick?: (series: MSeries, point: MPoint) => void;
  onBrush?: (range: [number, number] | null) => void;
  brush?: [number, number] | null;
  emptyText?: string;
  xLabel?: string;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const allX = panels.flatMap((p) => p.series.flatMap((s) => s.points.map((pt) => pt.x)));
  const xDomain: [number, number] = allX.length ? [Math.min(...allX), Math.max(...allX)] : [0, 1];
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(auto-fill, minmax(${minPanelWidth}px, 1fr))`, gap: 12 }}>
      {panels.map((p) => (
        <MultiplePanel
          key={p.key}
          panel={p}
          xDomain={xDomain}
          height={panelHeight}
          hoverX={hoverX}
          setHoverX={setHoverX}
          formatX={formatX}
          formatY={formatY}
          onPointClick={onPointClick}
          onBrush={onBrush}
          brush={brush ?? null}
          emptyText={emptyText}
          xLabel={xLabel}
        />
      ))}
    </div>
  );
}

function MultiplePanel({
  panel,
  xDomain,
  height,
  hoverX,
  setHoverX,
  formatX,
  formatY,
  onPointClick,
  onBrush,
  brush,
  emptyText,
  xLabel,
}: {
  panel: MPanel;
  xDomain: [number, number];
  height: number;
  hoverX: number | null;
  setHoverX: (x: number | null) => void;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  onPointClick?: (series: MSeries, point: MPoint) => void;
  onBrush?: (range: [number, number] | null) => void;
  brush: [number, number] | null;
  emptyText: string;
  xLabel?: string;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const drag = useRef<number | null>(null);
  const [dragRange, setDragRange] = useState<[number, number] | null>(null);
  const m = { top: 6, right: 10, bottom: 20, left: 38 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const ys = panel.series
    .flatMap((s) => s.points.flatMap((p) => [p.y, p.lo, p.hi]))
    .concat((panel.refs ?? []).map((r) => r.value))
    .filter((v): v is number => v != null && Number.isFinite(v));
  const hasData = panel.series.some((s) => s.points.some((p) => p.y != null));
  const yLo = ys.length ? Math.min(...ys) : 0;
  const yHi = ys.length ? Math.max(...ys) : 1;
  const pad = (yHi - yLo) * 0.08 || Math.abs(yHi) * 0.05 || 0.01;
  const x = scaleLinear().domain(xDomain[0] === xDomain[1] ? [xDomain[0] - 1, xDomain[1] + 1] : xDomain).range([0, innerW]);
  const y = scaleLinear().domain([yLo - pad, yHi + pad]).range([innerH, 0]);
  const yTicks = y.ticks(3);
  const xTicks = niceTicks(xDomain[0], xDomain[1], Math.max(4, Math.floor(innerW / 60)));
  const nearestX = (px: number) => {
    const xs = Array.from(new Set(panel.series.flatMap((s) => s.points.map((p) => p.x))));
    if (!xs.length) return null;
    const target = x.invert(px);
    return xs.reduce((best, v) => (Math.abs(v - target) < Math.abs(best - target) ? v : best), xs[0]);
  };
  const lineGen = d3line<MPoint>()
    .defined((p) => p.y != null)
    .x((p) => x(p.x))
    .y((p) => y(p.y as number));
  const areaGen = d3area<MPoint>()
    .defined((p) => p.lo != null && p.hi != null)
    .x((p) => x(p.x))
    .y0((p) => y(p.lo as number))
    .y1((p) => y(p.hi as number));

  // Reference labels sit just above their line; push apart labels that would overlap.
  const refLabelY = new Map<string, number>();
  let prevLabel = -Infinity;
  for (const r of [...(panel.refs ?? [])].filter((r) => r.value != null).sort((a, b) => y(a.value as number) - y(b.value as number))) {
    const want = Math.max(y(r.value as number) - 3, prevLabel + 11);
    const placed = Math.min(innerH - 2, want);
    refLabelY.set(r.key, placed);
    prevLabel = placed;
  }

  const tooltipFor = (xv: number): ReactNode => (
    <>
      <div className={c.tipTitle}>
        {panel.title} · {xLabel ? `${xLabel} ` : ""}
        {formatX(xv)}
      </div>
      {panel.series.map((s) => {
        const pt = s.points.find((p) => p.x === xv);
        return <TipRow key={s.key} color={s.color} dashed={s.dashed} label={s.label} value={pt?.y != null ? formatY(pt.y) : "—"} />;
      })}
      {(panel.refs ?? []).map((r) => (
        <TipRow key={r.key} color={r.color ?? "var(--muted)"} dashed label={r.label} value={r.value != null ? formatY(r.value) : "—"} />
      ))}
      {onPointClick && <div className={c.tipHint}>Click a point to open its run</div>}
    </>
  );

  const shownBrush = dragRange ?? brush;
  return (
    <div style={{ border: "1px solid var(--border)", borderRadius: 6, padding: "8px 8px 4px", background: "var(--surface)", minWidth: 0 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginBottom: 2, minWidth: 0 }}>
        <span style={{ fontSize: 12, fontWeight: 800, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={panel.title}>
          {panel.title}
        </span>
        {panel.subtitle && <span className="t-caption nowrap">{panel.subtitle}</span>}
        {hasData && (
          <span className="t-caption mono" style={{ marginLeft: "auto" }}>
            {formatY(yLo)}–{formatY(yHi)}
          </span>
        )}
      </div>
      <div ref={ref} style={{ position: "relative" }}>
        {!hasData ? (
          <div className={c.empty} style={{ height }}>
            {emptyText}
          </div>
        ) : (
          width > 0 && (
            <svg
              width={width}
              height={height}
              style={{ display: "block", userSelect: "none" }}
              onMouseLeave={() => {
                setHoverX(null);
                tip.hide();
                if (drag.current != null) {
                  drag.current = null;
                  setDragRange(null);
                }
              }}
            >
              <g transform={`translate(${m.left},${m.top})`}>
                {yTicks.map((t) => (
                  <g key={t}>
                    <line x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                    <text x={-5} y={y(t)} dy="0.32em" textAnchor="end" className={c.axisMono}>
                      {formatY(t)}
                    </text>
                  </g>
                ))}
                {xTicks.map((t) => (
                  <text key={t} x={x(t)} y={innerH + 13} textAnchor="middle" className={c.axisMono}>
                    {formatX(t)}
                  </text>
                ))}
                {shownBrush && (
                  <rect className={c.brush} x={x(Math.min(...shownBrush))} y={0} width={Math.abs(x(shownBrush[1]) - x(shownBrush[0]))} height={innerH} />
                )}
                {(panel.refs ?? []).map((r) =>
                  r.value == null ? null : (
                    <g key={r.key}>
                      <line x1={0} x2={innerW} y1={y(r.value)} y2={y(r.value)} stroke={r.color ?? "var(--muted)"} strokeDasharray="4 3" strokeWidth={1.25} />
                      <text x={innerW} y={refLabelY.get(r.key) ?? y(r.value) - 3} textAnchor="end" className={c.axisText} style={{ fontSize: 9.5 }}>
                        {r.label.length > 22 ? `${r.label.slice(0, 21)}…` : r.label}
                      </text>
                    </g>
                  ),
                )}
                {panel.series.map((s) => (
                  <g key={s.key}>
                    {s.points.some((p) => p.lo != null) && <path d={areaGen(s.points) ?? ""} fill={s.color} opacity={0.15} />}
                    <path d={lineGen(s.points) ?? ""} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dashed ? "5 3" : undefined} strokeLinejoin="round" />
                    {s.points.map((p, i) =>
                      p.y == null ? null : (
                        <circle
                          key={i}
                          cx={x(p.x)}
                          cy={y(p.y)}
                          r={hoverX === p.x ? 4.5 : 3}
                          fill={s.dashed ? "var(--surface)" : s.color}
                          stroke={s.dashed ? s.color : "var(--surface)"}
                          strokeWidth={s.dashed ? 1.5 : 1}
                        />
                      ),
                    )}
                  </g>
                ))}
                {hoverX != null && <line x1={x(hoverX)} x2={x(hoverX)} y1={0} y2={innerH} stroke="var(--fg-2)" strokeOpacity={0.35} />}
                <rect
                  x={0}
                  y={0}
                  width={innerW}
                  height={innerH}
                  fill="transparent"
                  style={{ cursor: onPointClick ? "pointer" : "crosshair" }}
                  onMouseDown={(e) => {
                    if (!onBrush) return;
                    const px = e.nativeEvent.offsetX - m.left;
                    drag.current = x.invert(px);
                  }}
                  onMouseMove={(e) => {
                    const px = e.nativeEvent.offsetX - m.left;
                    if (drag.current != null) {
                      setDragRange([drag.current, x.invert(px)]);
                      return;
                    }
                    const nx = nearestX(px);
                    setHoverX(nx);
                    if (nx != null) tip.show(e, tooltipFor(nx));
                  }}
                  onMouseUp={(e) => {
                    const px = e.nativeEvent.offsetX - m.left;
                    const start = drag.current;
                    drag.current = null;
                    setDragRange(null);
                    if (start != null && Math.abs(x(start) - px) > 6) {
                      const end = x.invert(px);
                      onBrush?.([Math.min(start, end), Math.max(start, end)]);
                      return;
                    }
                    const nx = nearestX(px);
                    if (nx == null || !onPointClick) return;
                    const pointerY = e.nativeEvent.offsetY - m.top;
                    let best: { s: MSeries; p: MPoint; d: number } | null = null;
                    for (const s of panel.series) {
                      const p = s.points.find((pt) => pt.x === nx && pt.y != null);
                      if (!p) continue;
                      const d = Math.abs(y(p.y as number) - pointerY);
                      if (!best || d < best.d) best = { s, p, d };
                    }
                    if (best) onPointClick(best.s, best.p);
                  }}
                />
              </g>
            </svg>
          )
        )}
      </div>
      <ChartTooltip state={tip.state} />
    </div>
  );
}
