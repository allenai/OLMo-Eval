import type { RunDetail, TaskResultRow } from "@contract/api-types";
import { scaleLinear } from "d3-scale";
import { useMemo, useState } from "react";
import { ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import cc from "@/charts/charts.module.css";
import { formatRate, formatRuntime } from "@/lib/format";
import { BASIS_LABEL, basisMark, layoutTimeline, runtimeTicks, type TimelineBar } from "@/lib/runtime";

const ROW = 15;
const BAR = 10;

/**
 * Gantt of one run: startup as its own bar before zero, then each task's wall-clock span while
 * it shared the inference workers, sorted by start.
 */
export function RuntimeTimeline({ run, rows, onOpen, selected }: { run: RunDetail; rows: TaskResultRow[]; onOpen: (task: string) => void; selected?: string }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const tip = useTooltip();
  const [hover, setHover] = useState<string | null>(null);
  const layout = useMemo(
    () => layoutTimeline(rows.map((r) => ({ key: r.task_name, label: r.task_name, runtime: r.runtime, failed: !!r.error })), run.startup_seconds),
    [rows, run.startup_seconds],
  );
  const narrow = width > 0 && width < 640;
  const labelW = narrow ? 112 : 210;
  const m = { top: 22, right: 12, bottom: 6 };
  const innerW = Math.max(10, width - labelW - m.right);
  const height = m.top + layout.bars.length * ROW + m.bottom;
  const x = scaleLinear().domain(layout.domain).range([0, innerW]);
  const ticks = runtimeTicks(layout.domain[0], layout.domain[1], Math.max(3, Math.floor(innerW / 80))).filter((t) => t >= layout.domain[0]);
  const tasks = layout.bars.filter((b) => b.kind === "task").length;
  if (!tasks) {
    return <p className="t-caption">No task spans were recorded for this run{run.startup_seconds == null ? " (it predates runtime recording)" : ""}.</p>;
  }

  const show = (e: React.MouseEvent, b: TimelineBar) => {
    setHover(b.key);
    tip.show(
      e,
      b.kind === "startup" ? (
        <>
          <div className={cc.tipTitle}>Startup</div>
          <TipRow label="duration" value={formatRuntime(b.end - b.start)} />
          <div className={cc.tipHint}>Model load and server start, before any task. Beaker queue time is not included.</div>
        </>
      ) : (
        <>
          <div className={cc.tipTitle}>{b.label}</div>
          <TipRow label="started" value={`${formatRuntime(b.start)} into processing`} />
          <TipRow label="finished" value={`${formatRuntime(b.end)} into processing`} />
          <TipRow label="span" value={formatRuntime(b.end - b.start)} />
          <TipRow label="own runtime" value={b.inference != null ? `${basisMark(b.basis)}${formatRuntime(b.inference)}` : "—"} />
          {b.inference != null && b.end > b.start && <TipRow label="share of span" value={formatRate(b.inference / (b.end - b.start))} />}
          {b.basis && <TipRow label="basis" value={BASIS_LABEL[b.basis]} />}
          <div className={cc.tipHint}>The span is longer than the task's own time because tasks share the workers. Click to open the task.</div>
        </>
      ),
    );
  };

  return (
    <div>
      <div className="row-wrap" style={{ gap: 12, marginBottom: 8 }}>
        <Legend
          items={[
            { label: "startup", color: "var(--seq-2)" },
            { label: "task span", color: "var(--teal-line)" },
            ...(layout.bars.some((b) => b.failed) ? [{ label: "failed", color: "var(--status-failed)" }] : []),
          ]}
        />
        {layout.missing > 0 && <span className="t-caption">{layout.missing} tasks have no recorded span.</span>}
      </div>
      <div ref={ref} style={{ maxHeight: 320, overflowY: "auto", overflowX: "hidden" }} onMouseLeave={() => (setHover(null), tip.hide())}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-label="Run timeline">
            <g transform={`translate(${labelW},0)`}>
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={x(t)} x2={x(t)} y1={m.top - 4} y2={height - m.bottom} stroke="var(--grid)" />
                  <text x={x(t)} y={12} textAnchor="middle" className={cc.axisMono}>
                    {t === 0 ? "0" : t < 0 ? `−${formatRuntime(-t)}` : formatRuntime(t)}
                  </text>
                </g>
              ))}
              {layout.domain[0] < 0 && <line x1={x(0)} x2={x(0)} y1={m.top - 4} y2={height - m.bottom} stroke="var(--border-strong)" strokeDasharray="3 3" />}
            </g>
            {layout.bars.map((b, i) => {
              const y = m.top + i * ROW;
              const active = hover === b.key || selected === b.key;
              const x0 = x(b.start);
              const w = Math.max(2, x(b.end) - x0);
              const fill = b.kind === "startup" ? "var(--seq-2)" : b.failed ? "var(--status-failed)" : "var(--teal-line)";
              return (
                <g
                  key={b.key}
                  style={{ cursor: b.kind === "task" ? "pointer" : "default" }}
                  onMouseMove={(e) => show(e, b)}
                  onClick={() => b.kind === "task" && onOpen(b.key)}
                >
                  <rect x={0} y={y} width={width} height={ROW} fill={active ? "var(--row-hover)" : "transparent"} />
                  <text
                    x={labelW - 8}
                    y={y + ROW / 2}
                    dy="0.32em"
                    textAnchor="end"
                    className={cc.label}
                    style={{ fontSize: 11, fontWeight: b.kind === "startup" ? 800 : active ? 800 : 400, fill: b.kind === "startup" ? "var(--fg-2)" : undefined }}
                  >
                    {b.label.length > (narrow ? 15 : 32) ? `${b.label.slice(0, narrow ? 14 : 31)}…` : b.label}
                  </text>
                  <rect x={labelW + x0} y={y + (ROW - BAR) / 2} width={w} height={BAR} rx={2} fill={fill} opacity={hover && !active ? 0.55 : 1} />
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <ChartTooltip state={tip.state} />
    </div>
  );
}
