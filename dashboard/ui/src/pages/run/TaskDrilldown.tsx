import type { RunDetail, SubjectKey, TaskResultRow } from "@contract/api-types";
import { Download, ExternalLink } from "lucide-react";
import { useState } from "react";
import { signArtifact, useHistograms, useRunConfigs } from "@/api/hooks/runs";
import { ContingencyTable, HistogramChart } from "@/charts/basic";
import { ChartPanel, Legend } from "@/charts/core";
import { DeltaValue, MetricName, ScoreValue } from "@/components/cells";
import { JsonDiff, JsonTree } from "@/components/Json";
import { Badge, Button, Checkbox, CopyText, EmptyState, ErrorPanel, Panel, Skeleton, Tip } from "@/components/primitives";
import { toast } from "@/components/toast";
import { formatCI, formatCount, formatDelta, formatDuration, formatP, formatRate, formatRuntime, formatScore, metricLabel, sig3 } from "@/lib/format";
import { BASIS_LABEL, basisMark } from "@/lib/runtime";
import { rowMeta, scoreFormat } from "./types";

export async function openSigned(gsUri: string): Promise<void> {
  try {
    const signed = await signArtifact(gsUri);
    window.open(signed.url, "_blank", "noopener");
  } catch {
    toast("Could not create a download link", { tone: "error" });
  }
}

const FINISH_COLORS: Record<string, string> = { stop: "var(--seq-4)", length: "var(--status-partial)", error: "var(--status-failed)" };

export function FinishReasons({ counts, onClick }: { counts: Record<string, number>; onClick?: (reason: string) => void }) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (!total) return <span className="t-caption">Not a generation task.</span>;
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return (
    <div className="col" style={{ gap: 6 }}>
      <div style={{ display: "flex", height: 10, borderRadius: 3, overflow: "hidden", gap: 2 }}>
        {entries.map(([reason, n]) => (
          <Tip key={reason} content={`${reason}: ${formatCount(n)} (${formatRate(n / total, 1)})`}>
            <span
              onClick={() => onClick?.(reason)}
              style={{ flex: n, background: FINISH_COLORS[reason] ?? "var(--cat-5)", cursor: onClick ? "pointer" : "default", minWidth: 2 }}
            />
          </Tip>
        ))}
      </div>
      <div className="row-wrap" style={{ gap: 12, fontSize: 12 }}>
        {entries.map(([reason, n]) => (
          <button
            key={reason}
            type="button"
            onClick={() => onClick?.(reason)}
            className="row"
            style={{ gap: 5, border: 0, background: "transparent", padding: 0, cursor: onClick ? "pointer" : "default" }}
          >
            <span style={{ width: 8, height: 8, borderRadius: 2, background: FINISH_COLORS[reason] ?? "var(--cat-5)" }} />
            <span>{reason}</span>
            <span className="mono muted">{formatRate(n / total, n / total < 0.1 ? 1 : 0)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function TaskDrilldown({
  run,
  row,
  baseline,
  baselineRunId,
  onOpenInstances,
}: {
  run: RunDetail;
  row: TaskResultRow;
  baseline?: SubjectKey;
  baselineRunId?: string | null;
  onOpenInstances: (patch: { cell?: string; fin?: string; score?: string }) => void;
}) {
  const hist = useHistograms(run.run_id, row.error ? null : row.task_result_id, baseline);
  const [showDiff, setShowDiff] = useState(false);
  const configs = useRunConfigs(run.run_id);
  const baseConfigs = useRunConfigs(showDiff ? baselineRunId : null);
  const m = rowMeta(row);
  const fmt = scoreFormat(row);
  const b = row.baseline;
  const config = configs.data?.task_configs.find((c) => c.task_result_id === row.task_result_id)?.config;
  const baseConfig = baseConfigs.data?.task_configs.find((c) => c.task_name === row.task_name)?.config;
  const metricKeys = Object.keys(row.metrics).sort((a, b2) => (a === row.primary_metric ? -1 : b2 === row.primary_metric ? 1 : a.localeCompare(b2)));
  const h = hist.data;

  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="col" style={{ gap: 6 }}>
        <div className="row-wrap" style={{ gap: 8 }}>
          <span className="t-caption">variant</span>
          <CopyText text={row.task_hash} display={<span className="mono">{row.task_hash.slice(0, 10)}</span>} />
          <span className="t-caption">primary</span>
          <MetricName metric={row.primary_metric} />
          {row.suites.map((su) => (
            <Badge key={su} tone="muted">
              {su}
            </Badge>
          ))}
          {b?.delta.hash_mismatch && <Badge tone="warn">config differs from baseline</Badge>}
        </div>
        {row.error ? (
          <ErrorPanel error={new Error(row.error)} title="This task failed" />
        ) : (
          <div className="row-wrap" style={{ gap: 18, alignItems: "baseline" }}>
            <span style={{ fontSize: 26, fontFamily: "var(--font-mono)" }}>
              {formatScore(row.score, fmt)}
              <span className="muted" style={{ fontSize: 13, marginLeft: 6 }}>
                ±{row.stderr != null ? (fmt === "percent" ? (row.stderr * 100).toFixed(1) : sig3(row.stderr)) : "—"}
              </span>
            </span>
            <span className="t-caption">n={formatCount(row.n)}</span>
            {b && (
              <>
                <span className="t-caption">
                  base <span className="mono" style={{ color: "var(--fg)" }}>{formatScore(b.score, fmt)}</span>
                </span>
                <span className="row" style={{ gap: 6 }}>
                  <span className="t-caption">delta</span>
                  <DeltaValue stats={b.delta} format={fmt} higherIsBetter={m.higherIsBetter} />
                </span>
                <span className="t-caption">
                  CI <span className="mono" style={{ color: "var(--fg)" }}>{formatCI(b.delta.ci_low, b.delta.ci_high, fmt)}</span>
                </span>
                <span className="t-caption mono">{formatP(b.delta.p_value)}</span>
              </>
            )}
            <span className="spacer" />
            <Button size="sm" variant="primary" onClick={() => onOpenInstances({})}>
              Open instances →
            </Button>
          </div>
        )}
      </div>

      {!row.error && (
        <div className="grid-12" style={{ gap: 12 }}>
          <div className="span-5">
            <Panel title="Vs baseline" caption={b ? "Click a cell to open those instances." : undefined}>
              {!b ? (
                <EmptyState title="No baseline" compact>
                  Set a baseline to see gained and lost instances.
                </EmptyState>
              ) : b.contingency ? (
                <ContingencyTable {...b.contingency} onCell={(cell) => onOpenInstances({ cell })} />
              ) : h?.delta ? (
                <HistogramChart
                  series={[{ data: h.delta, color: "var(--seq-4)", label: "instances", style: "fill" }]}
                  markers={[{ value: 0, color: "var(--muted)", label: "no change", dashed: true }]}
                  xLabel="per-instance delta (this − baseline)"
                  height={150}
                />
              ) : (
                <span className="t-caption">Instances do not pair with the baseline.</span>
              )}
            </Panel>
          </div>
          <div className="span-7">
            <Panel title="All metrics" pad="none">
              <div style={{ overflow: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                  <thead>
                    <tr style={{ background: "var(--row)", textAlign: "right", fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--muted)" }}>
                      <th style={{ textAlign: "left", padding: "6px 12px" }}>metric:scorer</th>
                      <th style={{ padding: "6px 8px" }}>value</th>
                      <th style={{ padding: "6px 8px" }}>base</th>
                      <th style={{ padding: "6px 12px" }}>delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricKeys.map((k) => {
                      const meta = row.metric_meta[k];
                      const f = meta?.display_format ?? "percent";
                      const v = row.metrics[k];
                      const bv = b?.metrics[k];
                      const d = v != null && bv != null ? v - bv : null;
                      const good = d != null && meta?.higher_is_better != null ? (meta.higher_is_better ? d > 0 : d < 0) : null;
                      return (
                        <tr key={k} style={{ borderTop: "1px solid var(--border)" }}>
                          <td style={{ padding: "5px 12px" }}>
                            <span className="mono" style={{ fontSize: 11.5 }} title={k}>
                              {metricLabel(k)}
                            </span>
                            {k === row.primary_metric && (
                              <Badge tone="teal">
                                primary
                              </Badge>
                            )}
                            {meta?.higher_is_better === false && <span className="t-caption"> lower is better</span>}
                          </td>
                          <td className="num" style={{ padding: "5px 8px" }}>
                            {formatScore(v, k === row.primary_metric ? fmt : f)}
                          </td>
                          <td className="num muted" style={{ padding: "5px 8px" }}>
                            {bv != null ? formatScore(bv, k === row.primary_metric ? fmt : f) : "—"}
                          </td>
                          <td className="num" style={{ padding: "5px 12px", color: good == null ? undefined : good ? "var(--better)" : "var(--worse)" }}>
                            {d != null ? formatDelta(d, k === row.primary_metric ? fmt : f) : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
          <div className="span-6">
            <ChartPanel
              title="Score distribution"
              legend={h?.baseline_score ? <Legend items={[{ label: "this run", color: "var(--seq-4)" }, { label: "baseline", color: "var(--teal-line)", dashed: true }]} /> : undefined}
              caption={h?.kind === "binary" ? "Binary metric: every instance scores 0 or 1." : "Per-instance primary scores; dashed outline is the baseline."}
            >
              {hist.isLoading ? (
                <Skeleton height={160} />
              ) : h?.score ? (
                <HistogramChart
                  series={[
                    { data: h.score, color: "var(--seq-4)", label: "this run", style: "fill" },
                    ...(h.baseline_score ? [{ data: h.baseline_score, color: "var(--teal-line)", label: "baseline", style: "outline" as const, dashed: true }] : []),
                  ]}
                  onBinClick={(lo, hi) => onOpenInstances({ score: `${lo.toFixed(3)}-${hi.toFixed(3)}` })}
                />
              ) : (
                <div className="row" style={{ gap: 18, padding: "18px 0" }}>
                  <div className="col" style={{ gap: 0 }}>
                    <span className="mono" style={{ fontSize: 22 }}>{formatScore(row.score, fmt)}</span>
                    <span className="t-caption">correct</span>
                  </div>
                  <div style={{ flex: 1, height: 12, borderRadius: 3, background: "var(--row)", overflow: "hidden" }}>
                    <div style={{ width: `${Math.max(0, Math.min(1, row.score ?? 0)) * 100}%`, height: "100%", background: "var(--seq-4)" }} />
                  </div>
                </div>
              )}
            </ChartPanel>
          </div>
          <div className="span-6">
            <ChartPanel
              title="Output length by correctness"
              caption="Completion tokens; the dashed line marks max_tokens."
              legend={<Legend items={[{ label: "correct", color: "var(--seq-4)" }, { label: "incorrect", color: "var(--worse)" }, { label: "max_tokens", color: "var(--muted)", dashed: true }]} />}
            >
              {hist.isLoading ? (
                <Skeleton height={160} />
              ) : h?.completion_tokens ? (
                <HistogramChart
                  series={[
                    { data: h.completion_tokens.correct, color: "var(--seq-4)", label: "correct", style: "outline" },
                    { data: h.completion_tokens.incorrect, color: "var(--worse)", label: "incorrect", style: "outline" },
                  ]}
                  markers={h.max_tokens ? [{ value: h.max_tokens, color: "var(--muted)", label: "max_tokens", dashed: true }] : []}
                  formatX={(v) => formatCount(v)}
                  xLabel="completion tokens"
                />
              ) : (
                <EmptyState title="No output lengths" compact>
                  Loglikelihood tasks do not generate text.
                </EmptyState>
              )}
            </ChartPanel>
          </div>
          <div className="span-12">
            <Panel title="Finish reasons" caption="Click a reason to filter the instance list.">
              <FinishReasons counts={row.finish_reason_counts} onClick={(fin) => onOpenInstances({ fin })} />
            </Panel>
          </div>
        </div>
      )}
      <div className="grid-12" style={{ gap: 12 }}>
        <div className="span-8">
          <Panel
            title="Task config"
            actions={baselineRunId ? <Checkbox checked={showDiff} onChange={setShowDiff} label="Diff vs baseline" /> : undefined}
          >
            {!config ? (
              <Skeleton height={120} />
            ) : showDiff ? (
              baseConfig ? <JsonDiff a={baseConfig} b={config} /> : <Skeleton height={80} />
            ) : (
              <JsonTree value={config} filename={`${row.task_name}-config.json`} maxHeight={360} />
            )}
          </Panel>
        </div>
        <div className="span-4">
          <Panel title="Artifacts">
            <div className="col" style={{ gap: 8, fontSize: 12.5 }}>
              {[
                ["Predictions", row.predictions_uri],
                ["Requests", row.requests_uri],
              ].map(([label, uri]) => (
                <div key={label} className="col" style={{ gap: 2 }}>
                  <span className="t-caption">{label}</span>
                  {uri ? (
                    <div className="row" style={{ gap: 6 }}>
                      <CopyText text={uri} display={<span className="mono" style={{ fontSize: 11 }}>{uri.split("/").pop()}</span>} />
                      <Button size="sm" variant="ghost" icon={<Download />} onClick={() => openSigned(uri)} aria-label={`Download ${label}`} />
                    </div>
                  ) : (
                    <span className="faint">not uploaded</span>
                  )}
                </div>
              ))}
              <div className="t-caption">
                {row.runtime.inference_seconds != null && (
                  <>
                    Runtime {basisMark(row.runtime.basis)}
                    {formatRuntime(row.runtime.inference_seconds)} ({BASIS_LABEL[row.runtime.basis].toLowerCase()}).{" "}
                  </>
                )}
                {row.duration_seconds != null && <>Done {formatDuration(row.duration_seconds)} after processing started. </>}
                {row.num_fewshot != null && <>{row.num_fewshot}-shot. </>}
                {row.limit != null && <>Limit {formatCount(row.limit)}. </>}
                Split {row.split ?? "—"}.
              </div>
              {run.links.gcs_console && (
                <a className="t-caption link" href={run.links.gcs_console} target="_blank" rel="noopener noreferrer">
                  Browse run folder in Cloud Console <ExternalLink size={11} />
                </a>
              )}
            </div>
          </Panel>
        </div>
      </div>
      {!row.error && <ScoreSummary row={row} />}
    </div>
  );
}

function ScoreSummary({ row }: { row: TaskResultRow }) {
  return (
    <p className="t-caption">
      {formatCount(row.instances_stored)} instances stored · {formatCount(row.instances_processed ?? row.n)} processed
      {row.instances_failed ? ` · ${formatCount(row.instances_failed)} failed` : ""}
      {row.completion_tokens_total != null && ` · ${formatCount(row.completion_tokens_total)} completion tokens`}
      {row.mean_completion_tokens != null && ` · mean output ${Math.round(row.mean_completion_tokens)} tokens`}
      {row.score_is_mean ? "" : " · corpus score is not a per-instance mean"}
      {" · "}
      <ScoreValue score={row.score} stderr={row.stderr} format={scoreFormat(row)} />
    </p>
  );
}
