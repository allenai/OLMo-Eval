import type { InferenceKpis, InferenceTaskRow } from "@contract/api-types";
import { Gauge } from "lucide-react";
import { useMemo } from "react";
import { useInference } from "@/api/hooks/runs";
import { BoxPlots, HBars } from "@/charts/basic";
import { ChartPanel } from "@/charts/core";
import { Timeline } from "@/charts/Timeline";
import { type Column, DataTable } from "@/components/DataTable";
import { Stat } from "@/components/PageHeader";
import { EmptyState, ErrorPanel, Panel, Skeleton } from "@/components/primitives";
import { formatCompact, formatCount, formatDuration, formatRate, formatSeconds, sig3 } from "@/lib/format";
import s from "./run.module.css";
import type { RunCtx } from "./types";

const SERIES_LABEL: Record<string, string> = {
  output_tokens_per_second: "Output tokens/s",
  vllm_num_requests_running: "Requests running",
  vllm_num_requests_waiting: "Requests waiting",
  vllm_kv_cache_usage_pct: "KV cache usage",
  gpu_utilization_pct: "GPU utilization",
  gpu_memory_used_mb: "GPU memory",
};

function Change({ value, base, lowerIsBetter }: { value: number | null; base: number | null | undefined; lowerIsBetter?: boolean }) {
  if (value == null || base == null || base === 0) return null;
  const pct = (value - base) / Math.abs(base);
  const good = lowerIsBetter ? pct < 0 : pct > 0;
  return (
    <span className="mono" style={{ color: Math.abs(pct) < 0.02 ? "var(--muted)" : good ? "var(--better)" : "var(--worse)" }}>
      {pct > 0 ? "+" : "−"}
      {Math.abs(pct * 100).toFixed(0)}%
    </span>
  );
}

function KpiRow({ k, b }: { k: InferenceKpis; b: InferenceKpis | null }) {
  const tile = (label: string, value: string, raw: number | null, base: number | null | undefined, baseText: string | null, lower?: boolean, note?: string) => (
    <Stat
      label={label}
      value={<span style={{ fontSize: 21 }}>{value}</span>}
      sub={
        note ? (
          <span>{note}</span>
        ) : b ? (
          <>
            <span>base {baseText ?? "—"}</span>
            <Change value={raw} base={base} lowerIsBetter={lower} />
          </>
        ) : undefined
      }
    />
  );
  return (
    <div className={s.kpis}>
      {tile("Wall clock", formatDuration(k.wall_clock_s), k.wall_clock_s, b?.wall_clock_s, b ? formatDuration(b.wall_clock_s) : null, true)}
      {tile("Prompt tokens", formatCompact(k.prompt_tokens), k.prompt_tokens, b?.prompt_tokens, b ? formatCompact(b.prompt_tokens) : null)}
      {tile("Completion tokens", formatCompact(k.completion_tokens), k.completion_tokens, b?.completion_tokens, b ? formatCompact(b.completion_tokens) : null)}
      {tile("Output tokens/s", k.output_tokens_per_second != null ? formatCount(k.output_tokens_per_second) : "—", k.output_tokens_per_second, b?.output_tokens_per_second, b?.output_tokens_per_second != null ? formatCount(b.output_tokens_per_second) : null)}
      {tile("Mean latency", formatSeconds(k.mean_latency_s), k.mean_latency_s, b?.mean_latency_s, b ? formatSeconds(b.mean_latency_s) : null, true)}
      {tile(
        "TTFT p50 / p95",
        k.ttft_p50_s != null ? `${formatSeconds(k.ttft_p50_s)} / ${formatSeconds(k.ttft_p95_s)}` : "not recorded",
        k.ttft_p50_s,
        b?.ttft_p50_s,
        b?.ttft_p50_s != null ? formatSeconds(b.ttft_p50_s) : null,
        true,
        k.ttft_p50_s == null ? "Per-request metrics were off" : undefined,
      )}
      {tile("Failed requests", formatCount(k.failed_requests), k.failed_requests, b?.failed_requests, b ? formatCount(b.failed_requests) : null, true)}
      {tile("GPU util (mean)", k.gpu_utilization_mean_pct != null ? `${k.gpu_utilization_mean_pct.toFixed(0)}%` : "—", k.gpu_utilization_mean_pct, b?.gpu_utilization_mean_pct, b?.gpu_utilization_mean_pct != null ? `${b.gpu_utilization_mean_pct.toFixed(0)}%` : null)}
      {tile("GPU-hours", k.gpu_hours != null ? sig3(k.gpu_hours) : "—", k.gpu_hours, b?.gpu_hours, b?.gpu_hours != null ? sig3(b.gpu_hours) : null, true)}
    </div>
  );
}

export function InferenceTab(ctx: RunCtx) {
  const inf = useInference(ctx.run.run_id, ctx.baseline);
  const d = inf.data;

  const throughput = useMemo(() => {
    const byTask = new Map<string, { tokens: number; wall: number }>();
    for (const b of d?.batches ?? []) {
      if (!b.task_name) continue;
      const cur = byTask.get(b.task_name) ?? { tokens: 0, wall: 0 };
      cur.tokens += b.total_completion_tokens;
      cur.wall += b.wall_clock_time_s;
      byTask.set(b.task_name, cur);
    }
    return [...byTask.entries()]
      .map(([task, v]) => ({ key: task, label: task, value: v.wall ? v.tokens / v.wall : null }))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
  }, [d]);

  const bands = useMemo(() => {
    const out: { label: string; start: number; end: number }[] = [];
    for (const b of d?.batches ?? []) {
      const last = out[out.length - 1];
      const end = b.t + b.wall_clock_time_s;
      if (last && last.label === (b.task_name ?? "")) last.end = end;
      else out.push({ label: b.task_name ?? "", start: b.t, end });
    }
    return out;
  }, [d]);

  const columns: Column<InferenceTaskRow>[] = [
    { id: "task", header: "Task", title: "Task", width: 230, pin: true, sortValue: (r) => r.task_name, cell: (r) => <span className="truncate">{r.task_name}</span>, csv: (r) => r.task_name },
    { id: "req", header: "Requests", title: "Requests", width: 90, align: "right", sortValue: (r) => r.instances, cell: (r) => <span className="mono">{formatCount(r.instances)}</span>, csv: (r) => r.instances },
    { id: "pt", header: "Prompt tok", title: "Prompt tokens", width: 104, align: "right", sortValue: (r) => r.prompt_tokens, cell: (r) => <span className="mono">{formatCompact(r.prompt_tokens)}</span>, csv: (r) => r.prompt_tokens },
    { id: "ct", header: "Compl. tok", title: "Completion tokens", width: 104, align: "right", sortValue: (r) => r.completion_tokens, cell: (r) => <span className="mono">{formatCompact(r.completion_tokens)}</span>, csv: (r) => r.completion_tokens },
    { id: "mean", header: "Mean len", title: "Mean completion tokens", width: 86, align: "right", sortValue: (r) => r.mean_completion_tokens, cell: (r) => <span className="mono">{r.mean_completion_tokens != null ? Math.round(r.mean_completion_tokens) : "—"}</span>, csv: (r) => r.mean_completion_tokens },
    { id: "trunc", header: "Trunc.", title: "Truncation", width: 72, align: "right", sortValue: (r) => r.truncation_rate, cell: (r) => <span className="mono">{formatRate(r.truncation_rate)}</span>, csv: (r) => r.truncation_rate },
    { id: "wall", header: "Wall clock", title: "Wall clock", width: 96, align: "right", sortValue: (r) => r.duration_seconds, cell: (r) => <span className="mono">{formatDuration(r.duration_seconds)}</span>, csv: (r) => r.duration_seconds },
    {
      id: "tps",
      header: "Tok/s",
      title: "Output tokens per second",
      width: 80,
      align: "right",
      sortValue: (r) => (r.completion_tokens && r.duration_seconds ? r.completion_tokens / r.duration_seconds : null),
      cell: (r) => <span className="mono">{r.completion_tokens && r.duration_seconds ? formatCount(r.completion_tokens / r.duration_seconds) : "—"}</span>,
      csv: (r) => (r.completion_tokens && r.duration_seconds ? r.completion_tokens / r.duration_seconds : null),
    },
    {
      id: "gpuh",
      header: "GPU-h",
      title: "GPU-hours",
      width: 76,
      align: "right",
      sortValue: (r) => r.duration_seconds,
      cell: (r) => <span className="mono">{r.duration_seconds != null ? sig3((r.duration_seconds / 3600) * (ctx.run.environment.gpu_count ?? 1)) : "—"}</span>,
      csv: (r) => (r.duration_seconds != null ? (r.duration_seconds / 3600) * (ctx.run.environment.gpu_count ?? 1) : null),
    },
  ];

  if (inf.isLoading) {
    return (
      <div className="col" style={{ gap: 12 }}>
        <Skeleton height={80} />
        <Skeleton height={300} />
      </div>
    );
  }
  if (inf.error) return <ErrorPanel error={inf.error} onRetry={() => inf.refetch()} />;
  if (!d || !d.available) {
    return (
      <Panel>
        <EmptyState icon={<Gauge />} title="No inference metrics">
          This run did not record inference metrics (API-only providers or metrics disabled).
        </EmptyState>
      </Panel>
    );
  }
  const boxItems = d.batch_latency_by_task.length
    ? d.batch_latency_by_task.map((x) => ({ key: x.task_name, label: x.task_name, box: x.box }))
    : [{ key: "all", label: "all batches", box: d.batch_latency }];
  const lengthItems = d.per_task.filter((t) => t.completion_tokens_box).map((t) => ({ key: t.task_name, label: t.task_name, box: t.completion_tokens_box }));
  return (
    <div className="col" style={{ gap: 14 }}>
      <KpiRow k={d.kpis} b={d.baseline_kpis} />
      <ChartPanel
        title="Timeline"
        caption="Seconds since the run started. Shaded bands mark tasks. Drag to zoom all panels; double-click to reset."
        csv={() => ({ header: ["series", "t_seconds", "value"], rows: d.series.flatMap((se) => se.points.map((p) => [se.name, p[0], p[1]])) })}
      >
        <Timeline
          series={d.series.map((se) => ({ key: `${se.name}:${se.device ?? ""}`, label: SERIES_LABEL[se.name] ?? se.name, unit: se.unit, points: se.points }))}
          bands={bands}
        />
      </ChartPanel>
      <div className="grid-12">
        <div className="span-6">
          <ChartPanel title="Throughput by task" caption="Output tokens per second, summed over each task's batches." csv={() => ({ header: ["task", "tokens_per_s"], rows: throughput.map((t) => [t.label, t.value]) })}>
            <HBars items={throughput.filter((t) => (t.value ?? 0) > 1)} format={(v) => formatCount(v)} unit="tok/s" />
          </ChartPanel>
        </div>
        <div className="span-6">
          <ChartPanel
            title={d.batch_latency_by_task.length ? "Batch latency by task" : "Batch latency"}
            caption="Mean latency per batch: box p25–p75, line at median, whiskers p5–p95."
          >
            <BoxPlots items={boxItems} format={(v) => formatSeconds(v)} />
            {d.request_latency?.ttft_s && (
              <div style={{ marginTop: 12 }}>
                <div className="t-overline" style={{ marginBottom: 4 }}>
                  Per request
                </div>
                <BoxPlots
                  items={[
                    { key: "e2e", label: "end-to-end", box: d.request_latency.end_to_end_s ? { ...d.request_latency.end_to_end_s } : null },
                    { key: "ttft", label: "time to first token", box: d.request_latency.ttft_s },
                    { key: "tpot", label: "time per output token", box: d.request_latency.tpot_s },
                  ]}
                  format={(v) => formatSeconds(v)}
                />
              </div>
            )}
          </ChartPanel>
        </div>
        <div className="span-12">
          <ChartPanel title="Completion length by task" caption="Completion tokens per instance: box p25–p75, median line, whiskers p5–p95.">
            <BoxPlots items={lengthItems} format={(v) => formatCount(v)} unit="tokens" />
          </ChartPanel>
        </div>
      </div>
      <Panel title="Cost and efficiency by task" pad="none">
        <DataTable tableId="inference-tasks" columns={columns} rows={d.per_task} getRowId={(r) => r.task_name} clientSort keyboard={false} maxHeight={480} exportName={`${ctx.run.run_id}-inference`} />
      </Panel>
    </div>
  );
}
