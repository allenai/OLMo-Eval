import type { PairStats } from "@contract/api-types";
import { useMemo } from "react";
import { usePairwise } from "@/api/hooks/compare";
import { CIWhisker } from "@/charts/basic";
import { ChartPanel, ChartTooltip, TipRow, useTooltip } from "@/charts/core";
import { Banner, ErrorPanel, Input, ModelDot, Skeleton } from "@/components/primitives";
import { formatCI, formatCount, formatDelta, formatP } from "@/lib/format";
import { divColor, divTextColor } from "@/lib/scales";
import { parseNumber } from "@/lib/url";
import { useBaseline } from "@/state/nav";
import c from "./compare.module.css";
import { type CompareCtx, columnLabel } from "./types";

/** Evidence band: 3 for p<0.001, 2 for p<0.01, 1 for p<alpha, 0 otherwise. */
export function evidenceBand(p: number | null, alpha: number): number {
  if (p == null) return 0;
  if (p < 0.001) return 3;
  if (p < 0.01) return 2;
  if (p < alpha) return 1;
  return 0;
}

export function PairwiseView(ctx: CompareCtx) {
  const margin = parseNumber(ctx.search.margin, 0);
  const [, setBaseline] = useBaseline();
  const body = useMemo(
    () => ({ subjects: ctx.keys, group: ctx.group ?? null, scope: ctx.scope, metric: ctx.metric, alpha: ctx.alpha, margin, shared_only: ctx.search.shared === "1" ? true : undefined }),
    [ctx.keys, ctx.group, ctx.scope, ctx.metric, ctx.alpha, margin, ctx.search.shared],
  );
  const pw = usePairwise(body);
  const tip = useTooltip();
  if (pw.isLoading) {
    return (
      <div className="col" style={{ gap: 8 }}>
        <span className="t-caption">Computing paired bootstrap (2,000 resamples)…</span>
        <Skeleton height={360} />
      </div>
    );
  }
  if (pw.error) return <ErrorPanel error={pw.error} onRetry={() => pw.refetch()} />;
  const d = pw.data!;
  const order = d.subjects.map((s) => s.key);
  const pairOf = new Map<string, PairStats>(d.pairs.map((p) => [`${p.row}|${p.col}`, p]));
  const rowSummary = new Map(d.rows.map((r) => [r.subject, r]));
  const fmt = d.display_format;
  // Zoom the win-rate whiskers to the data so narrow CIs stay visible; always include 50%.
  const lows = d.rows.map((r) => r.ci_low ?? r.mean_win_rate);
  const highs = d.rows.map((r) => r.ci_high ?? r.mean_win_rate);
  const wrDomain: [number, number] = [
    Math.max(0, Math.min(0.5, ...lows) - 0.03),
    Math.min(1, Math.max(0.5, ...highs) + 0.03),
  ];
  const template = `200px repeat(${order.length}, 62px) 160px`;
  return (
    <ChartPanel
      title="Pairwise"
      caption="Row minus column on shared instances, pooled across tasks in scope. Color shows the sign; intensity shows the strength of evidence. Ordered by mean win rate."
      refetching={pw.isFetching}
      csv={() => ({
        header: ["row", "col", "n_shared", "wins", "losses", "ties", "win_rate", "delta", "ci_low", "ci_high", "p_sign", "prob_row_better"],
        rows: d.pairs.map((p) => [p.row, p.col, p.n_shared, p.wins, p.losses, p.ties, p.win_rate, p.delta, p.ci_low, p.ci_high, p.p_sign, p.prob_row_better]),
      })}
      legend={
        <div className={c.toolbar} style={{ marginBottom: 0 }}>
          <span className="row" style={{ gap: 6 }}>
            <span className="t-caption">tie margin</span>
            <Input
              style={{ width: 70, height: 26 }}
              defaultValue={String(margin)}
              onBlur={(e) => ctx.setSearch({ margin: Number(e.target.value) ? e.target.value : undefined }, { replace: true })}
              aria-label="Tie margin"
            />
          </span>
          <span className={c.legendRamp}>
            col better
            <span className={c.ramp}>
              {[-3, -2, -1, 0, 1, 2, 3].map((st) => (
                <span key={st} style={{ background: divColor(st) }} />
              ))}
            </span>
            row better · bands p&lt;0.001, &lt;0.01, &lt;{ctx.alpha}, n.s.
          </span>
          <span className="spacer" />
          {d.mde80 != null && <span className="t-caption">MDE80 {formatDelta(d.mde80, fmt).replace("+", "")} pp</span>}
          <span className="t-caption">{d.tasks_used.length} tasks pooled</span>
        </div>
      }
    >
      {d.warnings.map((w) => (
        <div key={w} style={{ marginBottom: 8 }}>
          <Banner>{w}</Banner>
        </div>
      ))}
      <div className={c.pair} onMouseLeave={tip.hide}>
        <div className={c.pairGrid} style={{ gridTemplateColumns: template }}>
          <span />
          {order.map((k) => {
            const lbl = columnLabel(ctx, k);
            return (
              <span key={k} className={c.pairHead} title={ctx.labelOf(k)}>
                <span className="row" style={{ gap: 4, minWidth: 0 }}>
                  <ModelDot slot={ctx.slots[k]} baseline={k === ctx.baseline} />
                  <span className="truncate">{lbl.main}</span>
                </span>
                <span className={c.pairHeadSub}>{lbl.sub}</span>
              </span>
            );
          })}
          <span className={c.pairHead}>mean win rate</span>
          {order.map((row) => {
            const summary = rowSummary.get(row);
            return (
              <div key={row} style={{ display: "contents" }}>
                <button type="button" className={c.pairRowHead} title={`${ctx.labelOf(row)}: click to set as baseline`} onClick={() => setBaseline(row, ctx.labelOf(row))}>
                  <ModelDot slot={ctx.slots[row]} baseline={row === ctx.baseline} />
                  <span className="truncate">{ctx.labelOf(row)}</span>
                </button>
                {order.map((col) => {
                  if (row === col) return <span key={col} className={`${c.pairCell} ${c.pairDiag}`} />;
                  const p = pairOf.get(`${row}|${col}`);
                  if (!p) return <span key={col} className={c.pairCell} />;
                  const band = evidenceBand(p.p_sign ?? p.p_bootstrap, ctx.alpha);
                  const sign = (p.delta ?? 0) > 0 ? 1 : (p.delta ?? 0) < 0 ? -1 : 0;
                  const step = sign * band;
                  return (
                    <button
                      type="button"
                      key={col}
                      className={c.pairCell}
                      aria-label={`${ctx.labelOf(row)} minus ${ctx.labelOf(col)}: ${formatDelta(p.delta, fmt)}, win rate ${(p.win_rate * 100).toFixed(0)}%. Open disagreements.`}
                      onFocus={(e) => {
                        const r = e.currentTarget.getBoundingClientRect();
                        e.currentTarget.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: r.right, clientY: r.bottom }));
                      }}
                      onBlur={tip.hide}
                      style={{ background: band ? divColor(step) : "var(--div-0)", color: band ? divTextColor(step) : "var(--muted)" }}
                      onClick={() => ctx.setSearch({ view: "disagree", a: col, b: row, task: undefined, cell: undefined })}
                      onMouseMove={(e) =>
                        tip.show(
                          e,
                          <>
                            <div style={{ fontWeight: 800, marginBottom: 4 }}>
                              {ctx.labelOf(row)} vs {ctx.labelOf(col)}
                            </div>
                            <TipRow label="Delta (row − col)" value={`${formatDelta(p.delta, fmt)}${fmt === "percent" ? " pp" : ""}`} />
                            <TipRow label="95% CI" value={formatCI(p.ci_low, p.ci_high, fmt)} />
                            <TipRow label="Wins / losses / ties" value={`${formatCount(p.wins)} / ${formatCount(p.losses)} / ${formatCount(p.ties)}`} />
                            <TipRow label="Win rate" value={`${(p.win_rate * 100).toFixed(1)}%`} />
                            <TipRow label="Shared / contested" value={`${formatCount(p.n_shared)} / ${formatCount(p.n_contested)}`} />
                            <TipRow label="Sign test" value={formatP(p.p_sign)} />
                            <TipRow label="P(row better)" value={p.prob_row_better != null ? `${(p.prob_row_better * 100).toFixed(1)}%` : "—"} />
                            <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Click for instance-level disagreements</div>
                          </>,
                        )
                      }
                    >
                      {formatDelta(p.delta, fmt)}
                      <span className={c.pairSub}>{(p.win_rate * 100).toFixed(0)}% wins</span>
                    </button>
                  );
                })}
                <span className="row" style={{ gap: 6, paddingLeft: 8 }}>
                  <span className="mono" style={{ fontSize: 12, width: 40, textAlign: "right" }}>
                    {summary ? `${(summary.mean_win_rate * 100).toFixed(0)}%` : "—"}
                  </span>
                  {summary && <CIWhisker value={summary.mean_win_rate} low={summary.ci_low} high={summary.ci_high} domain={wrDomain} width={84} color="var(--fg)" />}
                </span>
              </div>
            );
          })}
        </div>
      </div>
      <ChartTooltip state={tip.state} />
    </ChartPanel>
  );
}
