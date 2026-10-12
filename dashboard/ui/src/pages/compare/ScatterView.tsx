import { scaleLinear, scaleSqrt } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { ChartPanel, ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import { Checkbox, EmptyState, Panel, Select, Skeleton } from "@/components/primitives";
import { formatCount, formatDelta, formatScore } from "@/lib/format";
import c from "./compare.module.css";
import cc from "@/charts/charts.module.css";
import type { CompareCtx } from "./types";

interface Pt {
  task: string;
  x: number;
  y: number;
  sx: number | null;
  sy: number | null;
  n: number;
  sig: "b" | "a" | "ns";
  delta: number;
}

export function ScatterView(ctx: CompareCtx) {
  const { matrix, keys, baseline, search, setSearch } = ctx;
  const a = search.a && keys.includes(search.a) ? search.a : (baseline ?? keys[0]);
  const b = search.b && keys.includes(search.b) && search.b !== a ? search.b : keys.find((k) => k !== a)!;
  const sizeByN = search.size === "n";
  const [showCI, setShowCI] = useState(false);
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const [brush, setBrush] = useState<[number, number, number, number] | null>(null);
  const dragStart = useRef<[number, number] | null>(null);
  const [dragRect, setDragRect] = useState<[number, number, number, number] | null>(null);

  const { points, omitted } = useMemo(() => {
    const data = matrix.data;
    if (!data) return { points: [] as Pt[], omitted: 0 };
    const ia = data.subjects.findIndex((s) => s.key === a);
    const ib = data.subjects.findIndex((s) => s.key === b);
    let omitted = 0;
    const pts: Pt[] = [];
    for (const row of data.rows) {
      if (row.kind !== "task") continue;
      const ca = row.cells[ia];
      const cb = row.cells[ib];
      if (!ca || !cb || ca.score == null || cb.score == null) continue;
      if ((row.meta?.display_format ?? "percent") !== "percent") {
        omitted += 1;
        continue;
      }
      const hib = row.meta?.higher_is_better !== false;
      let sig: Pt["sig"] = "ns";
      const stats = a === baseline ? cb.delta : b === baseline ? ca.delta : null;
      if (stats) {
        if (stats.significant) {
          const bBetter = a === baseline ? stats.improved : !stats.improved;
          sig = bBetter ? "b" : "a";
        }
      } else if (ca.stderr != null && cb.stderr != null) {
        const se = Math.sqrt(ca.stderr ** 2 + cb.stderr ** 2);
        const d = cb.score - ca.score;
        if (Math.abs(d) > 1.96 * se) sig = (d > 0) === hib ? "b" : "a";
      }
      pts.push({ task: row.name, x: ca.score * 100, y: cb.score * 100, sx: ca.stderr != null ? ca.stderr * 100 : null, sy: cb.stderr != null ? cb.stderr * 100 : null, n: cb.n ?? 0, sig, delta: cb.score - ca.score });
    }
    return { points: pts, omitted };
  }, [matrix.data, a, b, baseline]);

  if (matrix.isLoading) return <Skeleton height={480} />;
  const size = Math.min(Math.max(280, width - 8), 620);
  const m = { top: 12, right: 12, bottom: 40, left: 46 };
  const inner = size - m.left - m.right;
  const all = points.flatMap((p) => [p.x, p.y]);
  const lo = Math.max(0, Math.floor((Math.min(...all, 100) - 3) / 5) * 5);
  const hi = Math.min(100, Math.ceil((Math.max(...all, 0) + 3) / 5) * 5);
  const x = scaleLinear().domain([lo, hi]).range([0, inner]);
  const y = scaleLinear().domain([lo, hi]).range([inner, 0]);
  const r = scaleSqrt().domain([0, Math.max(1, ...points.map((p) => p.n))]).range([3, 11]);
  const color = (p: Pt) => (p.sig === "b" ? "var(--div-p2)" : p.sig === "a" ? "var(--div-n2)" : "var(--muted)");
  const inBrush = brush ? points.filter((p) => p.x >= brush[0] && p.x <= brush[2] && p.y >= brush[1] && p.y <= brush[3]) : [];
  const ticks = x.ticks(5);
  const subjectOptions = keys.map((k) => ({ value: k, label: ctx.labelOf(k) }));
  const local = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    return [x.invert(e.clientX - rect.left), y.invert(e.clientY - rect.top)] as [number, number];
  };

  return (
    <div className={c.side}>
      <ChartPanel
        title="Scatter: A vs B"
        caption="Each point is a task. Above the diagonal, B scores higher. Teal: B significantly better, ochre: A significantly better, gray: not resolved. Drag to select tasks."
        csv={() => ({ header: ["task", "a", "b", "delta"], rows: points.map((p) => [p.task, p.x, p.y, p.y - p.x]) })}
        legend={
          <div className={c.toolbar} style={{ marginBottom: 0 }}>
            <Select size="sm" label="A (x)" value={a} onChange={(v) => setSearch({ a: v }, { replace: true })} options={subjectOptions} width={220} />
            <Select size="sm" label="B (y)" value={b} onChange={(v) => setSearch({ b: v }, { replace: true })} options={subjectOptions.filter((o) => o.value !== a)} width={220} />
            <Checkbox checked={sizeByN} onChange={(on) => setSearch({ size: on ? "n" : undefined }, { replace: true })} label="Size by n" />
            <Checkbox checked={showCI} onChange={setShowCI} label="Stderr crosshairs" />
            <Legend
              items={[
                { label: "B better", color: "var(--div-p2)" },
                { label: "A better", color: "var(--div-n2)" },
                { label: "n.s.", color: "var(--muted)" },
              ]}
            />
          </div>
        }
      >
        <div ref={ref} style={{ display: "flex", justifyContent: "center" }}>
          {points.length === 0 ? (
            <EmptyState title="No shared percent-scale tasks" compact>
              A and B have no tasks in common in this scope.
            </EmptyState>
          ) : (
            width > 0 && (
              <svg width={size} height={size} onMouseLeave={tip.hide}>
                <g transform={`translate(${m.left},${m.top})`}>
                  {ticks.map((t) => (
                    <g key={t}>
                      <line x1={x(t)} x2={x(t)} y1={0} y2={inner} stroke="var(--grid)" />
                      <line x1={0} x2={inner} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                      <text x={x(t)} y={inner + 14} textAnchor="middle" className={cc.axisMono}>{t}</text>
                      <text x={-6} y={y(t)} dy="0.32em" textAnchor="end" className={cc.axisMono}>{t}</text>
                    </g>
                  ))}
                  <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} stroke="var(--muted)" strokeDasharray="4 4" />
                  <text x={inner / 2} y={inner + 32} textAnchor="middle" className={cc.axisText}>{ctx.labelOf(a)} (A)</text>
                  <text transform={`translate(${-34},${inner / 2}) rotate(-90)`} textAnchor="middle" className={cc.axisText}>{ctx.labelOf(b)} (B)</text>
                  {brush && <rect className={cc.brush} x={x(brush[0])} y={y(brush[3])} width={x(brush[2]) - x(brush[0])} height={y(brush[1]) - y(brush[3])} />}
                  {dragRect && <rect className={cc.brush} x={x(dragRect[0])} y={y(dragRect[3])} width={x(dragRect[2]) - x(dragRect[0])} height={y(dragRect[1]) - y(dragRect[3])} />}
                  <rect
                    x={0}
                    y={0}
                    width={inner}
                    height={inner}
                    fill="transparent"
                    style={{ cursor: "crosshair" }}
                    onMouseDown={(e) => (dragStart.current = local(e))}
                    onMouseMove={(e) => {
                      if (!dragStart.current) return;
                      const [px, py] = local(e);
                      const [sx, sy] = dragStart.current;
                      setDragRect([Math.min(sx, px), Math.min(sy, py), Math.max(sx, px), Math.max(sy, py)]);
                    }}
                    onMouseUp={() => {
                      dragStart.current = null;
                      if (dragRect && x(dragRect[2]) - x(dragRect[0]) > 6) setBrush(dragRect);
                      else setBrush(null);
                      setDragRect(null);
                    }}
                  />
                  {points.map((p) => {
                    const highlighted = !brush || inBrush.includes(p);
                    return (
                      <g key={p.task} opacity={highlighted ? 1 : 0.25}>
                        {showCI && p.sx != null && <line x1={x(p.x - p.sx)} x2={x(p.x + p.sx)} y1={y(p.y)} y2={y(p.y)} stroke={color(p)} strokeOpacity={0.6} />}
                        {showCI && p.sy != null && <line x1={x(p.x)} x2={x(p.x)} y1={y(p.y - p.sy)} y2={y(p.y + p.sy)} stroke={color(p)} strokeOpacity={0.6} />}
                        <circle
                          cx={x(p.x)}
                          cy={y(p.y)}
                          r={sizeByN ? r(p.n) : 5}
                          fill={color(p)}
                          stroke="var(--surface)"
                          strokeWidth={2}
                          style={{ cursor: "pointer" }}
                          onClick={() => setSearch({ view: "disagree", task: p.task, a, b, cell: undefined })}
                          onMouseMove={(e) =>
                            tip.show(
                              e,
                              <>
                                <div style={{ fontWeight: 800, marginBottom: 4 }}>{p.task}</div>
                                <TipRow label={`A ${ctx.labelOf(a)}`} value={p.x.toFixed(1)} />
                                <TipRow label={`B ${ctx.labelOf(b)}`} value={p.y.toFixed(1)} />
                                <TipRow label="B − A" value={formatDelta(p.delta, "percent")} />
                                <TipRow label="n" value={formatCount(p.n)} />
                                <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Click for disagreements on this task</div>
                              </>,
                            )
                          }
                        />
                      </g>
                    );
                  })}
                </g>
              </svg>
            )
          )}
        </div>
        {omitted > 0 && <p className="t-caption" style={{ marginTop: 6 }}>{omitted} raw-scale tasks (for example bits per byte) are left out because they do not share the 0–100 axis.</p>}
        <ChartTooltip state={tip.state} />
      </ChartPanel>
      <Panel title={brush ? `${inBrush.length} selected tasks` : "Largest gaps"} caption={brush ? "Drag elsewhere to change; click empty space to clear." : "Tasks with the largest |B − A|, signed."} pad="none">
        <div className="col" style={{ gap: 0, maxHeight: 560, overflow: "auto" }}>
          {(brush ? inBrush : [...points].sort((p, q) => Math.abs(q.delta) - Math.abs(p.delta)).slice(0, 20)).map((p) => (
            <button
              key={p.task}
              type="button"
              className="row"
              onClick={() => setSearch({ view: "disagree", task: p.task, a, b })}
              style={{ justifyContent: "space-between", padding: "6px 12px", border: 0, borderTop: "1px solid var(--border)", background: "transparent", cursor: "pointer", fontSize: 12.5, textAlign: "left" }}
            >
              <span className="truncate">{p.task}</span>
              <span className="mono" style={{ color: p.sig === "b" ? "var(--better)" : p.sig === "a" ? "var(--worse)" : "var(--muted)" }}>
                {formatDelta(p.delta, "percent")}
              </span>
            </button>
          ))}
          {!points.length && <span className="t-caption" style={{ padding: 12 }}>{formatScore(null)}</span>}
        </div>
      </Panel>
    </div>
  );
}
