import type { DeltaStats, DisplayFormat, ModelRef } from "@contract/api-types";
import type { ReactNode } from "react";
import { formatCI, formatCount, formatDelta, formatP, formatScore, formatStderr, formatStep, metricLabel } from "@/lib/format";
import { deltaEvidence, goodness } from "@/lib/scales";
import { BaseTag, ModelDot, Tip } from "./primitives";
import c from "./cells.module.css";

export function ScoreValue({
  score,
  stderr,
  format = "percent",
  showStderr = true,
  muted,
  strong,
}: {
  score: number | null | undefined;
  stderr?: number | null;
  format?: DisplayFormat;
  showStderr?: boolean;
  muted?: boolean;
  strong?: boolean;
}) {
  if (score == null) return <span className="faint mono">—</span>;
  return (
    <span className={c.score} style={muted ? { color: "var(--muted)" } : undefined}>
      <span className={strong ? c.strong : undefined}>{formatScore(score, format)}</span>
      {showStderr && stderr != null && <span className={c.stderr}>{formatStderr(stderr, format)}</span>}
    </span>
  );
}

export const METHOD_LABEL: Record<string, string> = {
  paired_bootstrap: "paired bootstrap",
  unpaired: "unpaired, stderr-based",
  insufficient: "insufficient data",
  none: "no CI",
};

export function DeltaTooltip({
  stats,
  format,
  thisScore,
  baseScore,
  label,
}: {
  stats: DeltaStats;
  format: DisplayFormat;
  thisScore?: number | null;
  baseScore?: number | null;
  label?: ReactNode;
}) {
  return (
    <div className={c.tip}>
      {label && <div className={c.tipTitle}>{label}</div>}
      <dl>
        {thisScore !== undefined && (
          <>
            <dt>This</dt>
            <dd>{formatScore(thisScore, format)}</dd>
          </>
        )}
        {baseScore !== undefined && (
          <>
            <dt>Baseline</dt>
            <dd>{formatScore(baseScore, format)}</dd>
          </>
        )}
        <dt>Delta</dt>
        <dd>
          {formatDelta(stats.delta, format)}
          {format === "percent" && stats.delta != null ? " pp" : ""}
        </dd>
        <dt>{Math.round((1 - stats.alpha) * 100)}% CI</dt>
        <dd>{formatCI(stats.ci_low, stats.ci_high, format)}</dd>
        <dt>p</dt>
        <dd>{formatP(stats.p_value)}</dd>
        <dt>Shared</dt>
        <dd>{stats.n_shared ? formatCount(stats.n_shared) : "—"}</dd>
        <dt>Method</dt>
        <dd>
          {METHOD_LABEL[stats.method]}
          {stats.n_boot ? `, ${formatCount(stats.n_boot)} resamples` : ""}
        </dd>
      </dl>
      {stats.hash_mismatch && <div className={c.tipNote}>Task configs differ between the two sides.</div>}
      {stats.note && <div className={c.tipNote}>{stats.note}</div>}
    </div>
  );
}

/** Signed delta with significance encoding: dot when significant, muted when not resolved. */
export function DeltaValue({
  stats,
  format = "percent",
  higherIsBetter = true,
  showDot = true,
  tooltip,
  ci,
}: {
  stats: DeltaStats | null | undefined;
  format?: DisplayFormat;
  higherIsBetter?: boolean | null;
  showDot?: boolean;
  tooltip?: ReactNode;
  ci?: boolean;
}) {
  if (!stats || stats.delta == null) return <span className="faint mono">—</span>;
  const ev = deltaEvidence(stats);
  const good = stats.improved === true ? 1 : stats.improved === false ? -1 : goodness(stats.delta, higherIsBetter);
  const color = ev === "significant" ? (good > 0 ? "var(--better)" : good < 0 ? "var(--worse)" : undefined) : "var(--muted)";
  const body = (
    <span className={c.delta} style={{ color }} data-evidence={ev}>
      {formatDelta(stats.delta, format)}
      {showDot && <span className={c.sigDot} data-on={ev === "significant"} role="img" aria-label={ev === "significant" ? "significant" : "not significant"} />}
      {ci && stats.ci_low != null && (
        <span className={c.ci}>{formatCI(stats.ci_low, stats.ci_high, format)}</span>
      )}
    </span>
  );
  return <Tip content={tooltip ?? <DeltaTooltip stats={stats} format={format} />}>{body}</Tip>;
}

export function SubjectLabel({
  model,
  label,
  slot,
  baseline,
  compact,
  showStep = true,
  title,
}: {
  model?: Pick<ModelRef, "series_label" | "step" | "name">;
  label?: string;
  slot?: number | null;
  baseline?: boolean;
  compact?: boolean;
  showStep?: boolean;
  title?: string;
}) {
  const name = label ?? model?.series_label ?? "";
  return (
    <span className={c.subject} title={title ?? model?.name ?? label}>
      <ModelDot slot={slot} baseline={baseline} />
      <span className={c.subjectName}>{name}</span>
      {showStep && !label && model?.step != null && <span className={c.step}>{formatStep(model.step)}</span>}
      {baseline && !compact && <BaseTag />}
    </span>
  );
}

export function MetricName({ metric }: { metric: string | null | undefined }) {
  return (
    <span className="mono muted" style={{ fontSize: 11.5 }} title={metric ?? undefined}>
      {metricLabel(metric)}
    </span>
  );
}

export function SigLegend() {
  return (
    <span className={c.legend}>
      <span className={c.sigDot} data-on="true" /> significant
      <span className={c.legendMuted}>muted = not resolved</span>
    </span>
  );
}
