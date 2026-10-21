import type { BoxStats, DeltaStats, DisplayFormat, Histogram as HistogramData, MiniHeatmap as MiniHeatmapData } from "@contract/api-types";
import { scaleLinear } from "d3-scale";
import type { ReactNode } from "react";
import { formatCI, formatCount, formatDelta, formatP, formatScore, sig3 } from "@/lib/format";
import { deltaEvidence, divColor, extent, goodness, normalize, seqColor } from "@/lib/scales";
import { ChartEmpty, ChartTooltip, niceTicks, TipRow, useSize, useTooltip } from "./core";
import c from "./charts.module.css";

// ---------------------------------------------------------------- sparkline

export function Sparkline({
  values,
  width = 60,
  height = 16,
  color = "var(--teal-line)",
  fill = false,
  fluid = false,
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  fill?: boolean;
  /** Shrink horizontally to fit a narrow container, up to `width`. */
  fluid?: boolean;
}) {
  const box = fluid
    ? { width: "100%", viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none" as const }
    : { width };
  const sizeStyle = fluid ? { maxWidth: width } : {};
  if (values.length < 2) return <svg {...box} height={height} aria-hidden style={sizeStyle} />;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const x = (i: number) => (i / (values.length - 1)) * (width - 2) + 1;
  const y = (v: number) => (hi === lo ? height / 2 : height - 2 - ((v - lo) / (hi - lo)) * (height - 4));
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <svg {...box} height={height} aria-hidden style={{ overflow: "visible", display: "block", ...sizeStyle }}>
      {fill && <path d={`${d}L${x(values.length - 1)},${height}L${x(0)},${height}Z`} fill={color} opacity={0.12} />}
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      {!fluid && <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r={2} fill={color} />}
    </svg>
  );
}

// ---------------------------------------------------------------- percentile strip

export function percentileOf(scores: number[], value: number, higherIsBetter: boolean | null): number {
  if (!scores.length) return NaN;
  const below = scores.filter((s) => (higherIsBetter === false ? s > value : s < value)).length;
  return below / scores.length;
}

export function PercentileStrip({
  scores,
  value,
  baseline,
  higherIsBetter = true,
  width = 80,
  height = 14,
  format = "percent",
  topLabel,
}: {
  scores: number[];
  value: number | null;
  baseline?: number | null;
  higherIsBetter?: boolean | null;
  width?: number;
  height?: number;
  format?: DisplayFormat;
  topLabel?: string;
}) {
  const tip = useTooltip();
  if (!scores.length || value == null) return <span className="faint mono">—</span>;
  const lo = Math.min(...scores, value, baseline ?? value);
  const hi = Math.max(...scores, value, baseline ?? value);
  const x = scaleLinear().domain(higherIsBetter === false ? [hi, lo] : [lo, hi]).range([4, width - 4]);
  const pct = percentileOf(scores, value, higherIsBetter);
  return (
    <>
      <svg
        width={width}
        height={height}
        style={{ display: "block", cursor: "default" }}
        onMouseMove={(e) =>
          tip.show(
            e,
            <>
              <div className={c.tipTitle}>Better than {Math.round(pct * 100)}% of {formatCount(scores.length)} runs</div>
              <TipRow color="var(--accent)" label="This run" value={formatScore(value, format)} />
              {baseline != null && <TipRow color="var(--teal-line)" dashed label="Baseline" value={formatScore(baseline, format)} />}
              <TipRow label="Range" value={`${formatScore(Math.min(...scores), format)} – ${formatScore(Math.max(...scores), format)}`} />
              {topLabel && <div className={c.tipHint}>Top: {topLabel}</div>}
            </>,
          )
        }
        onMouseLeave={tip.hide}
        role="img"
        aria-label={`Better than ${Math.round(pct * 100)}% of runs`}
      >
        <line x1={2} x2={width - 2} y1={height / 2} y2={height / 2} stroke="var(--border)" />
        {scores.slice(0, 500).map((sc, i) => (
          <line key={i} x1={x(sc)} x2={x(sc)} y1={height / 2 - 3.5} y2={height / 2 + 3.5} stroke="var(--muted)" strokeOpacity={0.45} />
        ))}
        {baseline != null && <circle cx={x(baseline)} cy={height / 2} r={3.2} fill="var(--surface)" stroke="var(--teal-line)" strokeWidth={1.5} />}
        <circle cx={x(value)} cy={height / 2} r={3.4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={1} />
      </svg>
      <ChartTooltip state={tip.state} />
    </>
  );
}

// ---------------------------------------------------------------- CI whisker (inline)

export function CIWhisker({
  value,
  low,
  high,
  domain,
  width = 72,
  height = 14,
  color = "var(--fg-2)",
  zero,
}: {
  value: number | null;
  low: number | null;
  high: number | null;
  domain: [number, number];
  width?: number;
  height?: number;
  color?: string;
  zero?: boolean;
}) {
  if (value == null) return <span className="faint mono">—</span>;
  const x = scaleLinear().domain(domain).range([3, width - 3]).clamp(true);
  const mid = height / 2;
  return (
    <svg width={width} height={height} style={{ display: "block", flex: "none" }} aria-hidden>
      {zero && <line x1={x(0)} x2={x(0)} y1={1} y2={height - 1} stroke="var(--border-strong)" />}
      {low != null && high != null && (
        <>
          <line x1={x(low)} x2={x(high)} y1={mid} y2={mid} stroke={color} strokeWidth={1} />
          <line x1={x(low)} x2={x(low)} y1={mid - 3} y2={mid + 3} stroke={color} />
          <line x1={x(high)} x2={x(high)} y1={mid - 3} y2={mid + 3} stroke={color} />
        </>
      )}
      <circle cx={x(value)} cy={mid} r={3} fill={color} />
    </svg>
  );
}

// ---------------------------------------------------------------- mini heatmap

export function MiniHeatmap({ data }: { data: MiniHeatmapData }) {
  const tip = useTooltip();
  if (!data.rows.length || !data.columns.length) return <div className="t-caption">No suite scores yet</div>;
  const colExt = data.columns.map((_, j) => extent(data.values.map((r) => r[j])));
  return (
    <>
      <div className={c.mini} style={{ gridTemplateColumns: `repeat(${data.columns.length}, minmax(0, 1fr))` }} onMouseLeave={tip.hide}>
        {data.rows.map((row, i) =>
          data.columns.map((col, j) => {
            const v = data.values[i][j];
            return (
              <span
                key={`${i}-${j}`}
                className={c.miniCell}
                style={{ background: v == null ? "var(--row)" : seqColor(normalize(v, colExt[j])) }}
                onMouseMove={(e) =>
                  tip.show(
                    e,
                    <>
                      <div className={c.tipTitle}>{row.label}</div>
                      <TipRow label={col.label} value={formatScore(v)} />
                    </>,
                  )
                }
              />
            );
          }),
        )}
      </div>
      <ChartTooltip state={tip.state} />
    </>
  );
}

// ---------------------------------------------------------------- histogram

export interface HistSeries {
  data: HistogramData;
  color: string;
  label: string;
  /** "fill" bars or "outline" step line. */
  style: "fill" | "outline";
  dashed?: boolean;
}

export function HistogramChart({
  series,
  height = 170,
  markers = [],
  formatX = (v: number) => sig3(v),
  xLabel,
  normalizeSeries = false,
  onBinClick,
}: {
  series: HistSeries[];
  height?: number;
  markers?: { value: number; color: string; label: string; dashed?: boolean }[];
  formatX?: (v: number) => string;
  xLabel?: string;
  normalizeSeries?: boolean;
  onBinClick?: (lo: number, hi: number) => void;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const valid = series.filter((s) => s.data.counts.length);
  const m = { top: 8, right: 12, bottom: xLabel ? 32 : 22, left: 40 };
  const innerW = Math.max(10, width - m.left - m.right);
  const innerH = height - m.top - m.bottom;
  const scaled = valid.map((s) => {
    const total = s.data.counts.reduce((a, b) => a + b, 0) || 1;
    return { ...s, values: s.data.counts.map((n) => (normalizeSeries ? n / total : n)), total };
  });
  const lo = Math.min(...valid.map((s) => s.data.edges[0]), ...markers.map((mk) => mk.value));
  const hi = Math.max(...valid.map((s) => s.data.edges[s.data.edges.length - 1]), ...markers.map((mk) => mk.value));
  const ymax = Math.max(1e-9, ...scaled.flatMap((s) => s.values));
  const x = scaleLinear().domain([lo, hi]).range([0, innerW]);
  const y = scaleLinear().domain([0, ymax]).range([innerH, 0]).nice();
  if (!valid.length) return <ChartEmpty height={height}>No data</ChartEmpty>;
  const base = scaled[0];
  return (
    <div ref={ref} style={{ position: "relative" }}>
      {width > 0 && (
        <svg width={width} height={height} onMouseLeave={tip.hide} role="img" aria-label="Histogram">
          <g transform={`translate(${m.left},${m.top})`}>
            {y.ticks(4).map((t) => (
              <g key={t}>
                <line x1={0} x2={innerW} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
                <text x={-6} y={y(t)} dy="0.32em" textAnchor="end" className={c.axisMono}>
                  {normalizeSeries ? `${Math.round(t * 100)}%` : formatCount(t)}
                </text>
              </g>
            ))}
            {scaled.map((s, si) =>
              s.style === "fill" ? (
                <g key={si}>
                  {s.values.map((v, i) => {
                    const x0 = x(s.data.edges[i]);
                    const x1 = x(s.data.edges[i + 1]);
                    const w = Math.max(1, x1 - x0 - 1.5);
                    return v > 0 ? (
                      <rect key={i} x={x0 + 0.75} y={y(v)} width={w} height={innerH - y(v)} fill={s.color} rx={Math.min(2, w / 2)} opacity={0.85} />
                    ) : null;
                  })}
                </g>
              ) : (
                <path
                  key={si}
                  d={s.values
                    .map((v, i) => `${i ? "L" : "M"}${x(s.data.edges[i])},${y(v)}H${x(s.data.edges[i + 1])}`)
                    .join("")}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={1.75}
                  strokeDasharray={s.dashed ? "4 3" : undefined}
                />
              ),
            )}
            {markers.map((mk, i) => (
              <g key={i}>
                <line x1={x(mk.value)} x2={x(mk.value)} y1={-4} y2={innerH} stroke={mk.color} strokeWidth={1.5} strokeDasharray={mk.dashed ? "3 3" : undefined} />
              </g>
            ))}
            <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="var(--border-strong)" />
            {niceTicks(lo, hi, Math.max(2, Math.floor(innerW / 70))).map((t) => (
              <text key={t} x={x(t)} y={innerH + 14} textAnchor="middle" className={c.axisMono}>
                {formatX(t)}
              </text>
            ))}
            {xLabel && (
              <text x={innerW / 2} y={innerH + 28} textAnchor="middle" className={c.axisText}>
                {xLabel}
              </text>
            )}
            {base.values.map((_, i) => (
              <rect
                key={`hit${i}`}
                x={x(base.data.edges[i])}
                y={0}
                width={Math.max(1, x(base.data.edges[i + 1]) - x(base.data.edges[i]))}
                height={innerH}
                fill="transparent"
                style={{ cursor: onBinClick ? "pointer" : "default" }}
                onClick={() => onBinClick?.(base.data.edges[i], base.data.edges[i + 1])}
                onMouseMove={(e) =>
                  tip.show(
                    e,
                    <>
                      <div className={c.tipTitle}>
                        {formatX(base.data.edges[i])} – {formatX(base.data.edges[i + 1])}
                      </div>
                      {scaled.map((s) => (
                        <TipRow key={s.label} color={s.color} dashed={s.style === "outline"} label={s.label} value={normalizeSeries ? `${(s.values[i] * 100).toFixed(1)}%` : formatCount(s.values[i])} />
                      ))}
                      {onBinClick && <div className={c.tipHint}>Click to filter instances</div>}
                    </>,
                  )
                }
              />
            ))}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

// ---------------------------------------------------------------- diverging bars

export interface DivergingItem {
  key: string;
  label: string;
  stats: DeltaStats;
  higherIsBetter: boolean | null;
  format: DisplayFormat;
  thisScore?: number | null;
  baseScore?: number | null;
}

export function DivergingBars({
  items,
  onClick,
  labelWidth = 170,
}: {
  items: DivergingItem[];
  onClick?: (item: DivergingItem) => void;
  labelWidth?: number;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const rowH = 22;
  const valueW = 52;
  const plotW = Math.max(60, width - labelWidth - valueW - 8);
  const maxAbs = Math.max(
    1e-9,
    ...items.map((it) => Math.max(Math.abs(goodness(it.stats.ci_low ?? it.stats.delta, it.higherIsBetter)), Math.abs(goodness(it.stats.ci_high ?? it.stats.delta, it.higherIsBetter)))),
  );
  const x = scaleLinear().domain([-maxAbs, maxAbs]).range([0, plotW]);
  const height = items.length * rowH + 18;
  if (!items.length) return <ChartEmpty>No changes to show</ChartEmpty>;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      {width > 0 && (
        <svg width={width} height={height} onMouseLeave={tip.hide} role="img" aria-label="Biggest changes versus baseline">
          <g transform={`translate(${labelWidth},4)`}>
            <line x1={x(0)} x2={x(0)} y1={0} y2={items.length * rowH} stroke="var(--border-strong)" />
            {items.map((it, i) => {
              const g = goodness(it.stats.delta, it.higherIsBetter);
              const lo = goodness(it.stats.ci_low, it.higherIsBetter);
              const hi = goodness(it.stats.ci_high, it.higherIsBetter);
              const ev = deltaEvidence(it.stats);
              const color = g >= 0 ? divColor(2) : divColor(-2);
              const yMid = i * rowH + rowH / 2;
              const x0 = Math.min(x(0), x(g));
              const w = Math.max(1, Math.abs(x(g) - x(0)));
              return (
                <g
                  key={it.key}
                  style={{ cursor: onClick ? "pointer" : "default" }}
                  onClick={() => onClick?.(it)}
                  onMouseMove={(e) =>
                    tip.show(
                      e,
                      <>
                        <div className={c.tipTitle}>{it.label}</div>
                        {it.thisScore !== undefined && <TipRow label="This" value={formatScore(it.thisScore, it.format)} />}
                        {it.baseScore !== undefined && <TipRow label="Baseline" value={formatScore(it.baseScore, it.format)} />}
                        <TipRow label="Delta" value={formatDelta(it.stats.delta, it.format)} />
                        <TipRow label="95% CI" value={formatCI(it.stats.ci_low, it.stats.ci_high, it.format)} />
                        <TipRow label="p" value={formatP(it.stats.p_value)} />
                        {onClick && <div className={c.tipHint}>Click to open changed instances</div>}
                      </>,
                    )
                  }
                >
                  <rect x={-labelWidth} y={i * rowH} width={width} height={rowH} fill="transparent" />
                  <text x={-10} y={yMid} dy="0.32em" textAnchor="end" className={c.label}>
                    {it.label.length > 26 ? `${it.label.slice(0, 25)}…` : it.label}
                  </text>
                  <rect x={x0} y={yMid - 6} width={w} height={12} rx={3} fill={color} opacity={ev === "significant" ? 1 : 0.35} />
                  {it.stats.ci_low != null && it.stats.ci_high != null && (
                    <g stroke="var(--fg-2)" strokeOpacity={0.7}>
                      <line x1={x(lo)} x2={x(hi)} y1={yMid} y2={yMid} />
                      <line x1={x(lo)} x2={x(lo)} y1={yMid - 4} y2={yMid + 4} />
                      <line x1={x(hi)} x2={x(hi)} y1={yMid - 4} y2={yMid + 4} />
                    </g>
                  )}
                  <text
                    x={plotW + 8}
                    y={yMid}
                    dy="0.32em"
                    className={c.axisMono}
                    style={{ fill: ev === "significant" ? (g > 0 ? "var(--better)" : "var(--worse)") : "var(--muted)", fontSize: 11 }}
                  >
                    {formatDelta(it.stats.delta, it.format)}
                  </text>
                </g>
              );
            })}
            <text x={x(-maxAbs * 0.98)} y={items.length * rowH + 12} className={c.axisText}>
              ← worse
            </text>
            <text x={x(maxAbs * 0.98)} y={items.length * rowH + 12} textAnchor="end" className={c.axisText}>
              better →
            </text>
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

// ---------------------------------------------------------------- horizontal bars with reference tick

export function HBars({
  items,
  format = (v: number) => sig3(v),
  labelWidth = 170,
  unit,
}: {
  items: { key: string; label: string; value: number | null; reference?: number | null; detail?: ReactNode }[];
  format?: (v: number) => string;
  labelWidth?: number;
  unit?: string;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const rowH = 20;
  const valueW = 64;
  const plotW = Math.max(40, width - labelWidth - valueW);
  const max = Math.max(1e-9, ...items.map((i) => Math.max(i.value ?? 0, i.reference ?? 0)));
  const x = scaleLinear().domain([0, max]).range([0, plotW]);
  if (!items.length) return <ChartEmpty>No data</ChartEmpty>;
  return (
    <div ref={ref}>
      {width > 0 && (
        <svg width={width} height={items.length * rowH + 4} onMouseLeave={tip.hide}>
          <g transform={`translate(${labelWidth},2)`}>
            {items.map((it, i) => {
              const y = i * rowH;
              return (
                <g
                  key={it.key}
                  onMouseMove={(e) =>
                    tip.show(
                      e,
                      <>
                        <div className={c.tipTitle}>{it.label}</div>
                        <TipRow label="This run" value={it.value == null ? "—" : `${format(it.value)}${unit ? ` ${unit}` : ""}`} />
                        {it.reference != null && <TipRow color="var(--teal-line)" label="Baseline" value={`${format(it.reference)}${unit ? ` ${unit}` : ""}`} />}
                        {it.detail}
                      </>,
                    )
                  }
                >
                  <rect x={-labelWidth} y={y} width={width} height={rowH} fill="transparent" />
                  <text x={-10} y={y + rowH / 2} dy="0.32em" textAnchor="end" className={c.label}>
                    {it.label.length > 26 ? `${it.label.slice(0, 25)}…` : it.label}
                  </text>
                  {it.value != null && <rect x={0} y={y + 4} width={Math.max(1, x(it.value))} height={rowH - 8} rx={3} fill="var(--seq-3)" />}
                  {it.reference != null && <line x1={x(it.reference)} x2={x(it.reference)} y1={y + 1} y2={y + rowH - 1} stroke="var(--teal-line)" strokeWidth={2} />}
                  <text x={plotW + 6} y={y + rowH / 2} dy="0.32em" className={c.axisMono}>
                    {it.value == null ? "—" : format(it.value)}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

// ---------------------------------------------------------------- box plots

export function BoxPlots({
  items,
  format = (v: number) => sig3(v),
  labelWidth = 170,
  unit,
}: {
  items: { key: string; label: string; box: BoxStats | null; reference?: number | null }[];
  format?: (v: number) => string;
  labelWidth?: number;
  unit?: string;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const rowH = 22;
  const plotW = Math.max(40, width - labelWidth - 12);
  const vals = items.flatMap((i) => (i.box ? [i.box.p5, i.box.p95] : [])).concat(items.map((i) => i.reference ?? NaN).filter(Number.isFinite));
  const lo = Math.min(0, ...vals);
  const hi = Math.max(1e-9, ...vals);
  const x = scaleLinear().domain([lo, hi]).range([0, plotW]).nice();
  const ticks = x.ticks(Math.max(2, Math.floor(plotW / 80)));
  const h = items.length * rowH + 20;
  if (!items.length) return <ChartEmpty>No data</ChartEmpty>;
  return (
    <div ref={ref}>
      {width > 0 && (
        <svg width={width} height={h} onMouseLeave={tip.hide}>
          <g transform={`translate(${labelWidth},2)`}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={0} y2={items.length * rowH} stroke="var(--grid)" />
                <text x={x(t)} y={items.length * rowH + 13} textAnchor="middle" className={c.axisMono}>
                  {format(t)}
                </text>
              </g>
            ))}
            {items.map((it, i) => {
              const y = i * rowH + rowH / 2;
              const b = it.box;
              return (
                <g
                  key={it.key}
                  onMouseMove={(e) =>
                    tip.show(
                      e,
                      <>
                        <div className={c.tipTitle}>{it.label}</div>
                        {b ? (
                          <>
                            <TipRow label="p95" value={format(b.p95)} />
                            <TipRow label="p75" value={format(b.p75)} />
                            <TipRow label="median" value={format(b.p50)} />
                            <TipRow label="p25" value={format(b.p25)} />
                            <TipRow label="p5" value={format(b.p5)} />
                            <TipRow label="mean" value={format(b.mean)} />
                            <TipRow label="n" value={formatCount(b.n)} />
                          </>
                        ) : (
                          "No data"
                        )}
                        {it.reference != null && <TipRow color="var(--teal-line)" label="Baseline median" value={format(it.reference)} />}
                        {unit && <div className={c.tipHint}>{unit}</div>}
                      </>,
                    )
                  }
                >
                  <rect x={-labelWidth} y={y - rowH / 2} width={width} height={rowH} fill="transparent" />
                  <text x={-10} y={y} dy="0.32em" textAnchor="end" className={c.label}>
                    {it.label.length > 26 ? `${it.label.slice(0, 25)}…` : it.label}
                  </text>
                  {b && (
                    <>
                      <line x1={x(b.p5)} x2={x(b.p95)} y1={y} y2={y} stroke="var(--muted)" />
                      <line x1={x(b.p5)} x2={x(b.p5)} y1={y - 4} y2={y + 4} stroke="var(--muted)" />
                      <line x1={x(b.p95)} x2={x(b.p95)} y1={y - 4} y2={y + 4} stroke="var(--muted)" />
                      <rect x={x(b.p25)} y={y - 6} width={Math.max(1, x(b.p75) - x(b.p25))} height={12} rx={2} fill="var(--seq-2)" stroke="var(--seq-4)" />
                      <line x1={x(b.p50)} x2={x(b.p50)} y1={y - 6} y2={y + 6} stroke="var(--fg)" strokeWidth={2} />
                    </>
                  )}
                  {it.reference != null && <line x1={x(it.reference)} x2={x(it.reference)} y1={y - 8} y2={y + 8} stroke="var(--teal-line)" strokeWidth={2} strokeDasharray="2 2" />}
                </g>
              );
            })}
          </g>
        </svg>
      )}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

// ---------------------------------------------------------------- 2x2 contingency

export function ContingencyTable({
  both_right,
  only_a,
  only_b,
  both_wrong,
  aLabel = "This run",
  bLabel = "Baseline",
  onCell,
}: {
  both_right: number;
  only_a: number;
  only_b: number;
  both_wrong: number;
  aLabel?: string;
  bLabel?: string;
  onCell?: (cell: "both_right" | "only_a" | "only_b" | "both_wrong") => void;
}) {
  const n = both_right + only_a + only_b + both_wrong || 1;
  const cell = (key: "both_right" | "only_a" | "only_b" | "both_wrong", value: number, tone?: "gain" | "loss") => (
    <button
      type="button"
      onClick={() => onCell?.(key)}
      style={{
        border: "1px solid var(--border)",
        borderRadius: 4,
        padding: "8px 10px",
        textAlign: "right",
        cursor: onCell ? "pointer" : "default",
        background: tone === "gain" ? "color-mix(in srgb, var(--div-p1) 70%, var(--surface))" : tone === "loss" ? "color-mix(in srgb, var(--div-n1) 70%, var(--surface))" : "var(--surface)",
        color: "var(--fg)",
        fontFamily: "var(--font-mono)",
        fontSize: 13,
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: 2,
      }}
      title={onCell ? "Open these instances" : undefined}
    >
      <span style={{ fontWeight: 500 }}>{formatCount(value)}</span>
      <span style={{ fontSize: 10.5, color: tone ? "var(--fg-2)" : "var(--muted)" }}>{((value / n) * 100).toFixed(1)}%</span>
    </button>
  );
  const net = only_a - only_b;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 4, alignItems: "stretch", fontSize: 11.5 }}>
        <span />
        <span className="t-caption" style={{ textAlign: "right" }}>{bLabel} right</span>
        <span className="t-caption" style={{ textAlign: "right" }}>{bLabel} wrong</span>
        <span className="t-caption" style={{ alignSelf: "center", paddingRight: 6 }}>{aLabel} right</span>
        {cell("both_right", both_right)}
        {cell("only_a", only_a, "gain")}
        <span className="t-caption" style={{ alignSelf: "center", paddingRight: 6 }}>{aLabel} wrong</span>
        {cell("only_b", only_b, "loss")}
        {cell("both_wrong", both_wrong)}
      </div>
      <div style={{ marginTop: 8, fontSize: 12.5 }}>
        Net{" "}
        <strong className="mono" style={{ color: net > 0 ? "var(--better)" : net < 0 ? "var(--worse)" : undefined }}>
          {net > 0 ? "+" : net < 0 ? "−" : ""}
          {formatCount(Math.abs(net))}
        </strong>{" "}
        instances <span className="muted">({formatCount(only_a)} gained, {formatCount(only_b)} lost)</span>
      </div>
    </div>
  );
}
