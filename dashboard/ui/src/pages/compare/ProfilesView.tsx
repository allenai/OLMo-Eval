import { scaleLinear, scalePoint } from "d3-scale";
import { useMemo, useRef, useState } from "react";
import { ChartPanel, ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import { Button, Checkbox, EmptyState, Segmented, Skeleton } from "@/components/primitives";
import { catColor } from "@/lib/scales";
import cc from "@/charts/charts.module.css";
import c from "./compare.module.css";
import type { CompareCtx } from "./types";

interface Axis {
  key: string;
  label: string;
  values: (number | null)[];
}

const MAX_AXES = 30;

/** Brushes in the URL: "lo~hi~axisKey" entries joined by commas (axis keys contain colons). */
export function parseBrushes(value: string | undefined): Record<string, [number, number]> {
  const out: Record<string, [number, number]> = {};
  for (const part of (value ?? "").split(",")) {
    const [lo, hi, ...rest] = part.split("~");
    const key = rest.join("~");
    if (key && Number.isFinite(Number(lo)) && Number.isFinite(Number(hi)) && lo !== "" && hi !== "") out[key] = [Number(lo), Number(hi)];
  }
  return out;
}

export function brushesParam(brushes: Record<string, [number, number]>): string | undefined {
  const parts = Object.entries(brushes).map(([key, [lo, hi]]) => `${Math.min(lo, hi).toFixed(1)}~${Math.max(lo, hi).toFixed(1)}~${key}`);
  return parts.length ? parts.join(",") : undefined;
}

export function ProfilesView(ctx: CompareCtx) {
  const { matrix, baseline, search, setSearch } = ctx;
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  // Level, scale, order and brushes live in the URL so a copied link reproduces the view.
  const level: "auto" | "suite" | "task" = search.plevel === "suite" || search.plevel === "task" ? search.plevel : "auto";
  const setLevel = (v: "auto" | "suite" | "task") => setSearch({ plevel: v === "auto" ? undefined : v, porder: undefined, pbrush: undefined }, { replace: true });
  const shared = search.pscale === "1";
  const setShared = (on: boolean) => setSearch({ pscale: on ? "1" : undefined }, { replace: true });
  const order = useMemo(() => (search.porder ? search.porder.split(",") : null), [search.porder]);
  const setOrder = (next: string[]) => setSearch({ porder: next.join(",") }, { replace: true });
  const brushes = useMemo(() => parseBrushes(search.pbrush), [search.pbrush]);
  const setBrushes = (update: (prev: Record<string, [number, number]>) => Record<string, [number, number]>) =>
    setSearch({ pbrush: brushesParam(update(brushes)) }, { replace: true });
  const [hover, setHover] = useState<string | null>(null);
  const dragAxis = useRef<string | null>(null);
  const brushDrag = useRef<{ axis: string; start: number } | null>(null);
  const [liveBrush, setLiveBrush] = useState<{ axis: string; range: [number, number] } | null>(null);

  const { axes, available, axisKind } = useMemo(() => {
    const data = matrix.data;
    if (!data) return { axes: [] as Axis[], available: 0, axisKind: "tasks" };
    const pct = data.rows.filter((r) => (r.meta?.display_format ?? "percent") === "percent");
    const suites = pct.filter((r) => r.kind === "suite");
    const tasks = pct.filter((r) => r.kind === "task");
    const useSuites = level === "suite" || (level === "auto" && tasks.length > MAX_AXES && suites.length >= 3);
    const all = useSuites ? suites : tasks;
    const axes = all
      .slice(0, MAX_AXES)
      .map((r) => ({ key: r.key, label: r.name, values: r.cells.map((cl) => (cl.score == null ? null : cl.score * 100)) }));
    return { axes, available: all.length, axisKind: useSuites ? "suites" : "tasks" };
  }, [matrix.data, level]);

  const ordered = useMemo(() => {
    if (!order) return axes;
    const byKey = new Map(axes.map((a) => [a.key, a]));
    const out = order.map((k) => byKey.get(k)).filter((a): a is Axis => !!a);
    for (const a of axes) if (!order.includes(a.key)) out.push(a);
    return out;
  }, [axes, order]);

  if (matrix.isLoading) return <Skeleton height={460} />;
  if (ordered.length < 2) {
    return <EmptyState title="Not enough axes" compact>Profiles need at least two percent-scale tasks or suites in scope.</EmptyState>;
  }
  const subjects = matrix.data!.subjects;
  const height = 420;
  const m = { top: 52, right: 64, bottom: 16, left: 64 };
  const innerW = Math.max(100, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const xp = scalePoint<string>().domain(ordered.map((a) => a.key)).range([0, innerW]);
  const yScales = new Map(
    ordered.map((a) => {
      const vals = a.values.filter((v): v is number => v != null);
      const lo = shared ? 0 : Math.min(...vals);
      const hi = shared ? 100 : Math.max(...vals);
      const pad = shared ? 0 : (hi - lo) * 0.08 || 1;
      return [a.key, scaleLinear().domain([lo - pad, hi + pad]).range([innerH, 0])];
    }),
  );
  const passes = (si: number) =>
    Object.entries(brushes).every(([axisKey, [lo, hi]]) => {
      const axis = ordered.find((a) => a.key === axisKey);
      const v = axis?.values[si];
      return v != null && v >= Math.min(lo, hi) && v <= Math.max(lo, hi);
    });
  const colorOf = (key: string) => (key === baseline ? "var(--teal-line)" : catColor(ctx.slots[key]));
  const step = ordered.length > 1 ? innerW / (ordered.length - 1) : innerW;

  return (
    <ChartPanel
      title="Profiles"
      caption="Parallel coordinates: one axis per task or suite, one line per subject. Drag along an axis to filter; drag an axis label to reorder. Baseline is dashed teal."
      csv={() => ({ header: ["subject", ...ordered.map((a) => a.label)], rows: subjects.map((s, i) => [s.label, ...ordered.map((a) => a.values[i])]) })}
      legend={
        <div className={c.toolbar} style={{ marginBottom: 0 }}>
          <Segmented
            value={level}
            onChange={setLevel}
            label="Axes"
            options={[
              { value: "auto", label: "Auto" },
              { value: "suite", label: "Suites" },
              { value: "task", label: "Tasks" },
            ]}
          />
          <Checkbox checked={shared} onChange={setShared} label="Shared 0–100 scale" />
          {Object.keys(brushes).length > 0 && (
            <Button size="sm" variant="ghost" onClick={() => setBrushes(() => ({}))}>
              Clear {Object.keys(brushes).length} filter{Object.keys(brushes).length > 1 ? "s" : ""}
            </Button>
          )}
          <span className="spacer" />
          <Legend items={subjects.map((s) => ({ label: ctx.labelOf(s.key), color: colorOf(s.key), dashed: s.key === baseline }))} />
        </div>
      }
    >
      <div ref={ref} style={{ userSelect: "none", overflowX: "auto" }}>
        {width > 0 && (
          <svg width={Math.max(width, ordered.length * 60)} height={height} onMouseLeave={() => (tip.hide(), setHover(null))}>
            <g transform={`translate(${m.left},${m.top})`}>
              {subjects.map((s, si) => {
                const ok = passes(si);
                const isHover = hover === s.key;
                const pts = ordered
                  .map((a) => (a.values[si] == null ? null : [xp(a.key)!, yScales.get(a.key)!(a.values[si]!)]))
                  .filter((p): p is [number, number] => !!p);
                const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join("");
                return (
                  <path
                    key={s.key}
                    d={d}
                    fill="none"
                    stroke={isHover ? "var(--accent)" : colorOf(s.key)}
                    strokeWidth={isHover ? 3 : 2}
                    strokeDasharray={s.key === baseline ? "6 4" : undefined}
                    opacity={!ok ? 0.12 : hover && !isHover ? 0.25 : 1}
                    style={{ cursor: "pointer" }}
                    onMouseMove={(e) => {
                      setHover(s.key);
                      tip.show(
                        e,
                        <>
                          <div style={{ fontWeight: 800, marginBottom: 4 }}>{s.label}</div>
                          {ordered.map((a) => (
                            <TipRow key={a.key} label={a.label} value={a.values[si] == null ? "—" : a.values[si]!.toFixed(1)} />
                          ))}
                        </>,
                      );
                    }}
                  />
                );
              })}
              {ordered.map((a) => {
                const y = yScales.get(a.key)!;
                const x = xp(a.key)!;
                const br = liveBrush?.axis === a.key ? liveBrush.range : brushes[a.key];
                return (
                  <g key={a.key} transform={`translate(${x},0)`}>
                    <line y1={0} y2={innerH} stroke="var(--border-strong)" />
                    {y.ticks(4).map((t) => (
                      <text key={t} x={-4} y={y(t)} dy="0.32em" textAnchor="end" className={cc.axisMono} style={{ fontSize: 9 }}>
                        {t}
                      </text>
                    ))}
                    {br && <rect className={cc.brush} x={-7} width={14} y={y(Math.max(...br))} height={Math.abs(y(br[0]) - y(br[1]))} />}
                    <rect
                      x={-9}
                      width={18}
                      y={0}
                      height={innerH}
                      fill="transparent"
                      style={{ cursor: "ns-resize" }}
                      onMouseDown={(e) => {
                        const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                        brushDrag.current = { axis: a.key, start: y.invert(e.clientY - rect.top) };
                      }}
                      onMouseMove={(e) => {
                        if (!brushDrag.current || brushDrag.current.axis !== a.key) return;
                        const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                        setLiveBrush({ axis: a.key, range: [brushDrag.current.start, y.invert(e.clientY - rect.top)] });
                      }}
                      onMouseUp={() => {
                        if (liveBrush && liveBrush.axis === a.key && Math.abs(y(liveBrush.range[0]) - y(liveBrush.range[1])) > 4) {
                          setBrushes((prev) => ({ ...prev, [a.key]: liveBrush.range }));
                        } else {
                          setBrushes((prev) => {
                            const next = { ...prev };
                            delete next[a.key];
                            return next;
                          });
                        }
                        brushDrag.current = null;
                        setLiveBrush(null);
                      }}
                    />
                    <foreignObject x={-step / 2} y={-48} width={step} height={44}>
                      <div
                        draggable
                        onDragStart={() => (dragAxis.current = a.key)}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={() => {
                          const from = dragAxis.current;
                          if (!from || from === a.key) return;
                          const keysNow = ordered.map((o) => o.key);
                          const next = keysNow.filter((k) => k !== from);
                          next.splice(next.indexOf(a.key), 0, from);
                          setOrder(next);
                        }}
                        title={`${a.label} — drag to reorder`}
                        style={{
                          fontSize: 10.5,
                          lineHeight: "12px",
                          textAlign: "center",
                          color: "var(--fg-2)",
                          cursor: "grab",
                          overflow: "hidden",
                          display: "-webkit-box",
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: "vertical",
                          wordBreak: "break-all",
                          padding: "0 2px",
                          height: 26,
                          marginTop: 14,
                        }}
                      >
                        {a.label}
                      </div>
                    </foreignObject>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>
      <p className="t-caption" style={{ marginTop: 6 }}>
        Values are scores in percent. Lines that fail any axis filter fade out.
        {available > axes.length && ` Showing the first ${axes.length} of ${available} ${axisKind}; narrow the scope to see others.`}
      </p>
      <ChartTooltip state={tip.state} />
    </ChartPanel>
  );
}
