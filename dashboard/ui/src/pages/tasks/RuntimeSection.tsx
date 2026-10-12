import type { MetricMeta, RuntimeMode, TaskRuntimeModelRow, TaskRuntimePoint } from "@contract/api-types";
import { scaleLinear, scaleLog } from "d3-scale";
import { useMemo } from "react";
import { useTaskRuntime } from "@/api/hooks/catalog";
import { ChartEmpty, ChartPanel, ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import cc from "@/charts/charts.module.css";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { ErrorPanel, Panel, Skeleton, Tip } from "@/components/primitives";
import { formatCount, formatRuntime, formatScore, formatStep, pluralize } from "@/lib/format";
import { BASIS_HELP, BASIS_LABEL, basisMark, gpuLabel, gpuShort, RUNTIME_MODE_LABEL, runtimeTicks, runtimeValue, wantsLogScale } from "@/lib/runtime";
import { extent, familyColors } from "@/lib/scales";
import { modelLabel } from "@/lib/subjects";
import s from "./runtime.module.css";

type ColorOf = (family: string | null | undefined) => string;

/** Runtime axis: log when the values span more than ~20x, with ticks at human time steps. */
function runtimeAxis(values: number[], range: [number, number], maxTicks: number, fromZero = true) {
  const ext = extent(values) ?? [1, 10];
  const log = wantsLogScale(values);
  if (log) {
    const lo = Math.max(0.05, ext[0] / 1.25);
    const hi = ext[1] * 1.25;
    return { scale: scaleLog().domain([lo, hi]).range(range).clamp(true), ticks: runtimeTicks(lo, hi, maxTicks, true), log };
  }
  if (!fromZero) {
    const pad = (ext[1] - ext[0]) * 0.08 || ext[1] * 0.1 || 1;
    const lo = Math.max(0, ext[0] - pad);
    const hi = ext[1] + pad;
    return { scale: scaleLinear().domain([lo, hi]).range(range), ticks: runtimeTicks(lo, hi, maxTicks), log };
  }
  const hi = ext[1] * 1.08 || 1;
  return { scale: scaleLinear().domain([0, hi]).range(range), ticks: runtimeTicks(0, hi, maxTicks), log };
}

function PointTip({ p, mode, fmt }: { p: TaskRuntimePoint; mode: RuntimeMode; fmt: MetricMeta["display_format"] }) {
  return (
    <>
      <div className={cc.tipTitle}>{modelLabel(p.model)}</div>
      <TipRow label={RUNTIME_MODE_LABEL[mode].toLowerCase()} value={`${basisMark(p.runtime.basis)}${formatRuntime(runtimeValue(p.runtime, mode))}`} />
      <TipRow label="score" value={formatScore(p.score, fmt)} />
      <TipRow label="run date" value={new Date(p.date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })} />
      <TipRow label="GPUs" value={gpuLabel(p.gpu_type, p.gpu_count)} />
      <TipRow label="basis" value={BASIS_LABEL[p.runtime.basis]} />
      <div className={cc.tipHint}>{BASIS_HELP[p.runtime.basis]} Click to open the run.</div>
    </>
  );
}

function Dot({ p, cx, cy, color, onShow, onHide, onOpen }: { p: TaskRuntimePoint; cx: number; cy: number; color: string; onShow: (e: React.MouseEvent) => void; onHide: () => void; onOpen: () => void }) {
  const estimated = p.runtime.basis === "estimated";
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4.5}
      fill={estimated ? "var(--surface)" : color}
      stroke={estimated ? color : "var(--surface)"}
      strokeWidth={estimated ? 1.75 : 1.5}
      style={{ cursor: "pointer" }}
      onMouseMove={onShow}
      onMouseLeave={onHide}
      onClick={onOpen}
    />
  );
}

function useOpenRun(task: string) {
  const navigate = useAppNavigate();
  return (p: TaskRuntimePoint) => navigate(`/runs/${p.run_id}?tab=tasks&task=${encodeURIComponent(task)}`);
}

export function RuntimeVsScore({ task, points, mode, meta, colorOf }: { task: string; points: TaskRuntimePoint[]; mode: RuntimeMode; meta: MetricMeta | null; colorOf: ColorOf }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const open = useOpenRun(task);
  const fmt = meta?.display_format ?? "percent";
  const data = points.filter((p) => runtimeValue(p.runtime, mode) != null && p.score != null);
  const height = 240;
  const m = { top: 8, right: 14, bottom: 34, left: 44 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const xa = runtimeAxis(data.map((p) => runtimeValue(p.runtime, mode)!), [0, innerW], Math.max(3, Math.floor(innerW / 80)), false);
  const sExt = extent(data.map((p) => p.score)) ?? [0, 1];
  const pad = (sExt[1] - sExt[0]) * 0.1 || 0.01;
  const y = scaleLinear().domain([sExt[0] - pad, sExt[1] + pad]).range([innerH, 0]).nice(4);
  if (!data.length) return <ChartEmpty height={height}>No runs with runtime data for these filters.</ChartEmpty>;
  return (
    <div ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Runtime versus score">
          <g transform={`translate(${m.left},${m.top})`}>
            {y.ticks(4).map((t) => (
              <g key={t}>
                <line x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                <text x={-6} y={y(t)} dy="0.32em" textAnchor="end" className={cc.axisMono}>
                  {formatScore(t, fmt)}
                </text>
              </g>
            ))}
            {xa.ticks.map((t) => (
              <g key={t}>
                <line x1={xa.scale(t)} x2={xa.scale(t)} y1={0} y2={innerH} stroke="var(--grid)" />
                <text x={xa.scale(t)} y={innerH + 14} textAnchor="middle" className={cc.axisMono}>
                  {formatRuntime(t)}
                </text>
              </g>
            ))}
            <text x={innerW} y={innerH + 29} textAnchor="end" className={cc.axisText}>
              {RUNTIME_MODE_LABEL[mode].toLowerCase()} time{xa.log ? " (log scale)" : ""} · faster ←
            </text>
            {data.map((p) => (
              <Dot
                key={p.task_result_id}
                p={p}
                cx={xa.scale(runtimeValue(p.runtime, mode)!)}
                cy={y(p.score!)}
                color={colorOf(p.model.family)}
                onShow={(e) => tip.show(e, <PointTip p={p} mode={mode} fmt={fmt} />)}
                onHide={tip.hide}
                onOpen={() => open(p)}
              />
            ))}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

export function RuntimeOverTime({ task, points, mode, meta, colorOf }: { task: string; points: TaskRuntimePoint[]; mode: RuntimeMode; meta: MetricMeta | null; colorOf: ColorOf }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const open = useOpenRun(task);
  const fmt = meta?.display_format ?? "percent";
  const data = points.filter((p) => runtimeValue(p.runtime, mode) != null);
  const height = 240;
  const m = { top: 8, right: 14, bottom: 34, left: 52 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const ya = runtimeAxis(data.map((p) => runtimeValue(p.runtime, mode)!), [innerH, 0], 5);
  const ts = data.map((p) => new Date(p.date).getTime());
  const x = scaleLinear()
    .domain([Math.min(...ts) - 12 * 3600_000, Math.max(...ts) + 12 * 3600_000])
    .range([0, innerW]);
  if (!data.length) return <ChartEmpty height={height}>No runs with runtime data for these filters.</ChartEmpty>;
  return (
    <div ref={ref}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Runtime over time">
          <g transform={`translate(${m.left},${m.top})`}>
            {ya.ticks.map((t) => (
              <g key={t}>
                <line x1={0} x2={innerW} y1={ya.scale(t)} y2={ya.scale(t)} stroke="var(--grid)" />
                <text x={-6} y={ya.scale(t)} dy="0.32em" textAnchor="end" className={cc.axisMono}>
                  {formatRuntime(t)}
                </text>
              </g>
            ))}
            {x.ticks(Math.max(2, Math.floor(innerW / 90))).map((t) => (
              <text key={t} x={x(t)} y={innerH + 14} textAnchor="middle" className={cc.axisMono}>
                {new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
              </text>
            ))}
            <text x={innerW} y={innerH + 29} textAnchor="end" className={cc.axisText}>
              run date{ya.log ? " · runtime on a log scale" : ""}
            </text>
            {data.map((p) => (
              <Dot
                key={p.task_result_id}
                p={p}
                cx={x(new Date(p.date).getTime())}
                cy={ya.scale(runtimeValue(p.runtime, mode)!)}
                color={colorOf(p.model.family)}
                onShow={(e) => tip.show(e, <PointTip p={p} mode={mode} fmt={fmt} />)}
                onHide={tip.hide}
                onOpen={() => open(p)}
              />
            ))}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

function RangeMark({ stats, scale, width, color }: { stats: TaskRuntimeModelRow["inference"]; scale: (v: number) => number; width: number; color: string }) {
  if (stats.median == null) return <span className="faint mono">—</span>;
  const h = 16;
  const mid = h / 2;
  const x = (v: number) => Math.max(3, Math.min(width - 3, scale(v)));
  return (
    <svg width={width} height={h} aria-hidden style={{ display: "block" }}>
      {stats.min != null && stats.max != null && <line x1={x(stats.min)} x2={x(stats.max)} y1={mid} y2={mid} stroke="var(--border-strong)" strokeWidth={1.5} />}
      {stats.p90 != null && <rect x={x(stats.median)} y={mid - 2.5} width={Math.max(0, x(stats.p90) - x(stats.median))} height={5} rx={1.5} fill={color} opacity={0.35} />}
      <circle cx={x(stats.median)} cy={mid} r={4} fill={color} stroke="var(--surface)" strokeWidth={1.5} />
    </svg>
  );
}

function runtimeStats(r: TaskRuntimeModelRow, mode: RuntimeMode) {
  return mode === "inference" ? r.inference : r.with_startup;
}

/** One line per (model, GPUs): median runtime with the p90 and min-max range on a shared axis. */
export function RuntimeByModel({ rows, mode, colorOf, groupByGpu }: { rows: TaskRuntimeModelRow[]; mode: RuntimeMode; colorOf: ColorOf; groupByGpu: boolean }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const statsOf = (r: TaskRuntimeModelRow) => runtimeStats(r, mode);
  const sorted = useMemo(
    () => [...rows].sort((a, b) => (runtimeStats(a, mode).median ?? Infinity) - (runtimeStats(b, mode).median ?? Infinity)),
    [rows, mode],
  );
  const groups = useMemo(() => {
    if (!groupByGpu) return [{ gpu: null as string | null, rows: sorted }];
    const map = new Map<string, TaskRuntimeModelRow[]>();
    for (const r of sorted) map.set(r.gpu_type ?? "Unknown GPU", [...(map.get(r.gpu_type ?? "Unknown GPU") ?? []), r]);
    return [...map.entries()].sort((a, b) => b[1].length - a[1].length).map(([gpu, list]) => ({ gpu, rows: list }));
  }, [sorted, groupByGpu]);
  const showRange = width >= 520;
  const rangeW = showRange ? Math.min(320, Math.max(140, width - 560)) : 0;
  const all = rows.flatMap((r) => [statsOf(r).min, statsOf(r).max, statsOf(r).median]).filter((v): v is number => v != null);
  const axis = runtimeAxis(all, [6, rangeW - 6], Math.max(2, Math.floor(rangeW / 80)), false);
  if (!rows.length) return <ChartEmpty height={120}>No runs with runtime data for these filters.</ChartEmpty>;
  return (
    <div ref={ref} className={s.byModel}>
      <div className={s.byModelHead} style={{ gridTemplateColumns: showRange ? `minmax(0,1fr) 64px ${rangeW}px 76px 44px 72px` : "minmax(0,1fr) 76px 44px 72px" }}>
        <span>Model</span>
        {showRange && <span>GPUs</span>}
        {showRange && (
          <span className={s.axisLabels} style={{ width: rangeW }}>
            {axis.ticks.map((t) => (
              <span key={t} style={{ left: axis.scale(t) }}>
                {formatRuntime(t)}
              </span>
            ))}
          </span>
        )}
        <Tip content="Median across runs. The bar runs from the median to the 90th percentile; the line spans the fastest to the slowest run.">
          <span className={s.num}>Median</span>
        </Tip>
        <span className={s.num}>Runs</span>
        <Tip content="Inference seconds per 1,000 instances (median). Comparable across task variants of different sizes.">
          <span className={s.num}>Per 1k</span>
        </Tip>
      </div>
      <div className={s.byModelBody}>
        {groups.map((g) => (
          <div key={g.gpu ?? "all"}>
            {g.gpu && (
              <div className={s.groupHead}>
                {gpuShort(g.gpu)} <span className="faint">· {pluralize(g.rows.length, "model")}</span>
              </div>
            )}
            {g.rows.map((r) => {
              const st = statsOf(r);
              return (
                <div
                  key={`${r.model.model_id}|${r.gpu_type}|${r.gpu_count}`}
                  className={s.byModelRow}
                  style={{ gridTemplateColumns: showRange ? `minmax(0,1fr) 64px ${rangeW}px 76px 44px 72px` : "minmax(0,1fr) 76px 44px 72px" }}
                >
                  <AppLink href={`/models/${encodeURIComponent(r.model.series)}`} className={s.model} title={r.model.name}>
                    <span className={s.swatch} style={{ background: colorOf(r.model.family) }} />
                    <span className="truncate">{r.model.series_label}</span>
                    {r.model.step != null && <span className={s.step}>@ {formatStep(r.model.step)}</span>}
                  </AppLink>
                  {showRange && <span className="mono muted">{gpuLabel(r.gpu_type, r.gpu_count)}</span>}
                  {showRange && <RangeMark stats={st} scale={axis.scale} width={rangeW} color={colorOf(r.model.family)} />}
                  <Tip content={st.p90 != null ? `p90 ${formatRuntime(st.p90)} · range ${formatRuntime(st.min)} to ${formatRuntime(st.max)}` : null}>
                    <span className={`mono ${s.num}`}>{formatRuntime(st.median)}</span>
                  </Tip>
                  <span className={`mono muted ${s.num}`}>{formatCount(r.runs)}</span>
                  <span className={`mono muted ${s.num}`}>{r.seconds_per_1k_instances.median != null ? formatRuntime(r.seconds_per_1k_instances.median) : "—"}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

export function RuntimeSection({
  task,
  hash,
  mode,
  gpu,
  family,
  group,
  user,
  palette,
}: {
  /** Family colors shared with the rest of the page, so a family keeps its color across charts. */
  palette?: ReturnType<typeof familyColors>;
  task: string;
  hash: string | undefined;
  mode: RuntimeMode;
  gpu: string | undefined;
  family?: string;
  group?: string;
  user?: string;
}) {
  const q = useTaskRuntime(task, { hash, gpu_type: gpu, family, group, user });
  const data = q.data;
  const points = useMemo(() => data?.points ?? [], [data]);
  const { colorOf, legend } = useMemo(() => {
    const own = palette ?? familyColors(points.map((p) => p.model.family));
    const present = new Set(points.map((p) => own.colorOf(p.model.family)));
    return { colorOf: own.colorOf, legend: own.legend.filter((l) => present.has(l.color)) };
  }, [points, palette]);
  const hasEstimates = points.some((p) => p.runtime.basis === "estimated");
  const legendItems = [...legend, ...(hasEstimates ? [{ label: "estimated", color: "var(--fg-2)", hollow: true }] : [])];
  const caption =
    mode === "inference"
      ? "Each task's own inference time. Runs share workers across tasks, so this is the task's share, not when it finished."
      : "Inference plus the run's startup (model load, server start). Beaker queue time is not included.";
  if (q.error) return <ErrorPanel error={q.error} onRetry={() => q.refetch()} title="Could not load runtime" />;
  return (
    <section className={s.section} aria-labelledby="runtime-heading">
      <div className={s.sectionHead}>
        <h2 id="runtime-heading" className="t-section">
          Runtime
        </h2>
        <span className="t-caption">{caption}</span>
        {data && data.not_recorded > 0 && (
          <span className={s.note}>
            {pluralize(data.not_recorded, "older run")} {data.not_recorded === 1 ? "has" : "have"} no runtime data and {data.not_recorded === 1 ? "is" : "are"} left out.
          </span>
        )}
      </div>
      <div className="grid-12">
        <div className="span-6">
          <ChartPanel
            id="runtime-vs-score"
            title="Runtime vs score"
            caption="One point per run. Hollow points are estimates."
            legend={<Legend items={legendItems} />}
            refetching={q.isFetching && !q.isLoading}
            csv={() => ({
              header: ["run_id", "model", "date", "gpu_type", "gpu_count", "basis", "inference_seconds", "with_startup_seconds", "score"],
              rows: points.map((p) => [p.run_id, p.model.name, p.date, p.gpu_type, p.gpu_count, p.runtime.basis, p.runtime.inference_seconds, p.runtime.with_startup_seconds, p.score]),
            })}
          >
            {q.isLoading ? <Skeleton height={240} /> : <RuntimeVsScore task={task} points={points} mode={mode} meta={data?.meta ?? null} colorOf={colorOf} />}
          </ChartPanel>
        </div>
        <div className="span-6">
          <ChartPanel id="runtime-over-time" title="Runtime over time" caption="Each run by date. Spread within a model usually comes from GPU type or shared load." legend={<Legend items={legendItems} />} refetching={q.isFetching && !q.isLoading}>
            {q.isLoading ? <Skeleton height={240} /> : <RuntimeOverTime task={task} points={points} mode={mode} meta={data?.meta ?? null} colorOf={colorOf} />}
          </ChartPanel>
        </div>
      </div>
      <Panel
        title="Runtime by model"
        caption={gpu ? `Runs on ${gpuShort(gpu)} only.` : "Grouped by GPU type, fastest first. Times are not comparable across GPU types."}
        refetching={q.isFetching && !q.isLoading}
      >
        {q.isLoading ? <Skeleton height={160} /> : <RuntimeByModel rows={data?.rows ?? []} mode={mode} colorOf={colorOf} groupByGpu={!gpu && (data?.gpu_types.length ?? 0) > 1} />}
      </Panel>
    </section>
  );
}
