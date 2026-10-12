import { scaleLinear } from "d3-scale";
import { line as d3line } from "d3-shape";
import { useRef, useState } from "react";
import { formatDuration, sig3 } from "@/lib/format";
import { ChartTooltip, TipRow, useSize, useTooltip } from "./core";
import c from "./charts.module.css";

export interface TimelineSeries {
  key: string;
  label: string;
  unit: string | null;
  points: [number, number][];
}

export interface TimelineBand {
  label: string;
  start: number;
  end: number;
}

/**
 * Small multiples sharing an x-axis (seconds since run start). Task boundaries are alternating
 * bands; drag across any panel to zoom all of them, double-click to reset.
 */
export function Timeline({ series, bands, panelHeight = 92 }: { series: TimelineSeries[]; bands: TimelineBand[]; panelHeight?: number }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const allX = series.flatMap((s) => s.points.map((p) => p[0]));
  const full: [number, number] = allX.length ? [Math.min(...allX), Math.max(...allX)] : [0, 1];
  const [domain, setDomain] = useState<[number, number] | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const drag = useRef<number | null>(null);
  const [dragRange, setDragRange] = useState<[number, number] | null>(null);
  const m = { left: 132, right: 14 };
  const innerW = Math.max(10, width - m.left - m.right);
  const x = scaleLinear().domain(domain ?? full).range([0, innerW]);
  const ticks = x.ticks(Math.max(2, Math.floor(innerW / 90)));
  const visibleBands = bands.filter((b) => b.end >= x.domain()[0] && b.start <= x.domain()[1]);

  const toX = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    return x.invert(e.clientX - rect.left);
  };

  return (
    <div ref={ref} style={{ userSelect: "none" }} onDoubleClick={() => setDomain(null)}>
      {width > 0 && (
        <>
          <svg width={width} height={22} style={{ display: "block" }}>
            <g transform={`translate(${m.left},0)`}>
              {visibleBands.map((b, i) => {
                const x0 = Math.max(0, x(b.start));
                const x1 = Math.min(innerW, x(b.end));
                return x1 - x0 > 36 ? (
                  <text key={i} x={(x0 + x1) / 2} y={14} textAnchor="middle" className={c.axisText} style={{ fontSize: 9.5 }}>
                    {b.label.length * 5.5 > x1 - x0 ? `${b.label.slice(0, Math.max(3, Math.floor((x1 - x0) / 6)))}…` : b.label}
                  </text>
                ) : null;
              })}
            </g>
          </svg>
          {series.map((s) => {
            const pts = s.points.filter((p) => p[0] >= x.domain()[0] && p[0] <= x.domain()[1]);
            const ys = pts.map((p) => p[1]);
            const lo = Math.min(0, ...ys);
            const hi = Math.max(1e-9, ...ys);
            const y = scaleLinear().domain([lo, hi]).range([panelHeight - 6, 6]).nice();
            const path = d3line<[number, number]>()
              .x((p) => x(p[0]))
              .y((p) => y(p[1]))(pts);
            const hoverPoint = hoverX != null ? pts.reduce<[number, number] | null>((best, p) => (!best || Math.abs(p[0] - hoverX) < Math.abs(best[0] - hoverX) ? p : best), null) : null;
            return (
              <svg key={s.key} width={width} height={panelHeight} style={{ display: "block", borderTop: "1px solid var(--grid)" }}>
                <text x={0} y={18} className={c.label} style={{ fontSize: 11.5, fontWeight: 800, fill: "var(--fg-2)" }}>
                  {s.label}
                </text>
                <text x={0} y={33} className={c.axisText}>
                  {s.unit ?? ""}
                </text>
                <text x={0} y={panelHeight - 8} className={c.axisMono}>
                  max {sig3(hi)}
                </text>
                <g transform={`translate(${m.left},0)`}>
                  {visibleBands.map((b, i) =>
                    i % 2 === 0 ? (
                      <rect key={i} x={Math.max(0, x(b.start))} y={0} width={Math.max(0, Math.min(innerW, x(b.end)) - Math.max(0, x(b.start)))} height={panelHeight} fill="var(--row)" />
                    ) : null,
                  )}
                  {y.ticks(2).map((t) => (
                    <line key={t} x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                  ))}
                  <path d={path ?? ""} fill="none" stroke="var(--teal-line)" strokeWidth={1.75} strokeLinejoin="round" />
                  {dragRange && <rect className={c.brush} x={x(Math.min(...dragRange))} y={0} width={Math.abs(x(dragRange[1]) - x(dragRange[0]))} height={panelHeight} />}
                  {hoverX != null && <line x1={x(hoverX)} x2={x(hoverX)} y1={0} y2={panelHeight} stroke="var(--fg-2)" strokeOpacity={0.35} />}
                  {hoverPoint && <circle cx={x(hoverPoint[0])} cy={y(hoverPoint[1])} r={3.5} fill="var(--accent)" stroke="var(--surface)" />}
                  <rect
                    x={0}
                    y={0}
                    width={innerW}
                    height={panelHeight}
                    fill="transparent"
                    style={{ cursor: "crosshair" }}
                    onMouseDown={(e) => (drag.current = toX(e))}
                    onMouseMove={(e) => {
                      const xv = toX(e);
                      if (drag.current != null) setDragRange([drag.current, xv]);
                      setHoverX(xv);
                      const band = bands.find((b) => xv >= b.start && xv <= b.end);
                      tip.show(
                        e,
                        <>
                          <div className={c.tipTitle}>{formatDuration(xv)} since start</div>
                          {band && <TipRow label="task" value={band.label} />}
                          {series.map((ss) => {
                            const p = ss.points.reduce<[number, number] | null>((best, q) => (!best || Math.abs(q[0] - xv) < Math.abs(best[0] - xv) ? q : best), null);
                            return <TipRow key={ss.key} label={ss.label} value={p ? `${sig3(p[1])}${ss.unit ? ` ${ss.unit}` : ""}` : "—"} />;
                          })}
                          <div className={c.tipHint}>Drag to zoom · double-click to reset</div>
                        </>,
                      );
                    }}
                    onMouseUp={(e) => {
                      const start = drag.current;
                      drag.current = null;
                      setDragRange(null);
                      const end = toX(e);
                      if (start != null && Math.abs(x(start) - x(end)) > 8) setDomain([Math.min(start, end), Math.max(start, end)]);
                    }}
                    onMouseLeave={() => {
                      setHoverX(null);
                      tip.hide();
                      drag.current = null;
                      setDragRange(null);
                    }}
                  />
                </g>
              </svg>
            );
          })}
          <svg width={width} height={20} style={{ display: "block", borderTop: "1px solid var(--border)" }}>
            <g transform={`translate(${m.left},0)`}>
              {ticks.map((t) => (
                <text key={t} x={x(t)} y={14} textAnchor="middle" className={c.axisMono}>
                  {formatDuration(t)}
                </text>
              ))}
            </g>
          </svg>
        </>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}
