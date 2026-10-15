import type { MatrixCell, MatrixRow } from "@contract/api-types";
import { useMemo } from "react";
import { ChartPanel, Legend } from "@/charts/core";
import { Heatmap, type HeatCell, type HeatRow } from "@/charts/Heatmap";
import { DeltaTooltip } from "@/components/cells";
import { Badge, Button, EmptyState, ModelDot, Segmented, Select, Skeleton } from "@/components/primitives";
import { formatCount, formatDelta, formatRate, formatRuntime, formatRuntimeDelta, formatScore, formatStderr } from "@/lib/format";
import { RUNTIME_MODE_LABEL, runtimeMetricMode } from "@/lib/runtime";
import { deltaEvidence, divColor, divStep, extent, goodness, normalize, ranks, seqColor, seqTextColor, divTextColor } from "@/lib/scales";
import { useBaseline } from "@/state/nav";
import c from "./compare.module.css";
import type { CompareCtx } from "./types";

type Mode = "abs" | "delta" | "rank";

/** Runtime delta relative to the baseline's runtime (score minus delta), so rows of any length compare. */
function relativeDelta(cell: MatrixCell): number | null {
  const d = cell.delta?.delta;
  if (d == null || cell.score == null) return null;
  const base = cell.score - d;
  return base > 0 ? d / base : null;
}

function fmtOf(row: MatrixRow) {
  return row.meta?.display_format ?? "percent";
}

export function HeatmapView(ctx: CompareCtx) {
  const { matrix, search, setSearch, baseline } = ctx;
  const [, setBaseline] = useBaseline();
  const wantsDelta = search.mode === "delta";
  const mode: Mode = wantsDelta && baseline ? "delta" : search.mode === "rank" ? "rank" : "abs";
  const globalScale = search.scale === "global";
  const rsort = search.rsort ?? "name";
  const data = matrix.data;
  const keys = useMemo(() => data?.subjects.map((s) => s.key) ?? [], [data]);
  const baseIdx = baseline ? keys.indexOf(baseline) : -1;
  const rtMode = runtimeMetricMode(data?.metric ?? ctx.metric);

  const rows = useMemo(() => {
    if (!data) return [];
    if (rsort === "name") return data.rows;
    const tasks = data.rows.filter((r) => r.kind === "task");
    const score = (r: MatrixRow): number => {
      if (rsort === "spread") {
        const ext = extent(r.cells.map((cl) => cl.score));
        return ext ? -(ext[1] - ext[0]) : Infinity;
      }
      if (rsort.startsWith("delta:")) {
        const idx = keys.indexOf(rsort.slice(6));
        const d = r.cells[idx]?.delta?.delta;
        return d == null ? Infinity : goodness(d, r.meta?.higher_is_better ?? true);
      }
      return 0;
    };
    return [...tasks].sort((a, b) => score(a) - score(b)).map((r) => ({ ...r, depth: 0, parent: null }));
  }, [data, rsort, keys]);

  const globalExt = useMemo(() => extent((data?.rows ?? []).filter((r) => fmtOf(r) === "percent").flatMap((r) => r.cells.map((cl) => cl.score))), [data]);
  const deltaScale = useMemo(() => {
    const abs = (data?.rows ?? []).flatMap((r) => r.cells.map((cl) => (rtMode ? Math.abs(relativeDelta(cl) ?? 0) : Math.abs(cl.delta?.delta ?? 0) / (fmtOf(r) === "percent" ? 1 : 100))));
    const sorted = abs.filter((v) => v > 0).sort((a, b) => a - b);
    return sorted.length ? sorted[Math.floor(sorted.length * 0.9)] : 0.05;
  }, [data, rtMode]);

  if (matrix.isLoading || !data) return <Skeleton height={480} />;

  const heatRows: HeatRow[] = rows.map((row) => {
    const fmt = fmtOf(row);
    const hib = rtMode ? false : (row.meta?.higher_is_better ?? true);
    const show = (v: number | null) => (rtMode ? formatRuntime(v) : formatScore(v, fmt));
    const ext = globalScale && fmt === "percent" ? globalExt : extent(row.cells.map((cl) => cl.score));
    const rk = ranks(row.cells.map((cl) => cl.score), hib);
    const cells: HeatCell[] = row.cells.map((cell: MatrixCell, j) => {
      const subjectLabel = ctx.labelOf(keys[j]);
      const status = cell.status === "partial" ? "ok" : cell.status;
      const tooltip = (
        <div>
          <div style={{ fontWeight: 800, marginBottom: 4 }}>
            {row.name} · {subjectLabel}
          </div>
          {cell.status === "missing" ? (
            <span className="muted">No result for this subject.</span>
          ) : cell.status === "failed" ? (
            <span style={{ color: "var(--danger-fg)" }}>Task failed in this run.</span>
          ) : (
            rtMode ? (
              <>
                <div className="mono">
                  {formatRuntime(cell.score)} <span className="muted">{RUNTIME_MODE_LABEL[rtMode].toLowerCase()}</span>
                  {cell.n != null && <> · n {formatCount(cell.n)}</>}
                </div>
                {cell.score == null && <div className="muted">Runtime not recorded for this result.</div>}
                {cell.children_missing > 0 && <div className="muted">{cell.children_missing} child tasks missing from the sum</div>}
                {cell.delta?.delta != null && (
                  <div className="mono" style={{ color: cell.delta.delta < 0 ? "var(--better)" : cell.delta.delta > 0 ? "var(--worse)" : undefined }}>
                    {formatRuntimeDelta(cell.delta.delta)} vs baseline
                    {relativeDelta(cell) != null && ` (${relativeDelta(cell)! > 0 ? "+" : "−"}${formatRate(Math.abs(relativeDelta(cell)!))})`}
                  </div>
                )}
                <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Lower is better. No significance test: runtime has no per-instance values.</div>
              </>
            ) : (
            <>
              <div className="mono">
                {formatScore(cell.score, fmt)} <span className="muted">{formatStderr(cell.stderr, fmt)}</span> · n {formatCount(cell.n)}
              </div>
              {cell.children_missing > 0 && <div className="muted">{cell.children_missing} child tasks missing</div>}
              {cell.delta && <DeltaTooltip stats={cell.delta} format={fmt} />}
              {row.kind === "task" && <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Click to open disagreements vs baseline</div>}
            </>
            )
          )}
        </div>
      );
      const onClick =
        row.kind === "task" && cell.status === "ok" && !rtMode
          ? () => setSearch({ view: "disagree", task: row.name, a: baseline ?? keys[0], b: keys[j] === (baseline ?? keys[0]) ? keys.find((k) => k !== keys[j]) : keys[j], cell: undefined })
          : undefined;
      if (mode === "delta") {
        if (j === baseIdx) return { text: show(cell.score), status, tooltip, background: "var(--row)", color: "var(--muted)", onClick };
        const d = cell.delta;
        if (rtMode) {
          const rel = relativeDelta(cell);
          const step = rel != null ? divStep(goodness(rel, false), deltaScale) : 0;
          return { text: d?.delta != null ? formatRuntimeDelta(d.delta) : "", status, tooltip, background: d?.delta != null ? divColor(step) : undefined, color: divTextColor(step), onClick };
        }
        const ev = deltaEvidence(d);
        const step = d?.delta != null ? divStep(goodness(d.delta / (fmt === "percent" ? 1 : 100), hib), deltaScale) : 0;
        return {
          text: d?.delta != null ? formatDelta(d.delta, fmt) : "",
          status,
          tooltip,
          background:
            ev === "insufficient"
              ? "repeating-linear-gradient(45deg, var(--border) 0 1px, transparent 1px 6px)"
              : ev === "significant"
                ? divColor(step)
                : `color-mix(in srgb, ${divColor(step)} 35%, var(--surface))`,
          // Unresolved cells are marked by the faded fill and the missing dot; --muted text fails AA on the tint.
          color: ev === "significant" ? divTextColor(step) : ev === "insufficient" ? "var(--muted)" : "var(--fg-2)",
          sig: ev === "significant",
          hashMismatch: cell.delta?.hash_mismatch,
          onClick,
        };
      }
      if (mode === "rank") {
        const r = rk[j];
        const t = r == null ? 0 : 1 - (r - 1) / Math.max(1, keys.length - 1);
        return { text: r == null ? "" : String(r), status, tooltip, background: r == null ? undefined : seqColor(t), color: seqTextColor(t), onClick };
      }
      const t = cell.score == null ? 0 : normalize(cell.score, ext, hib !== false);
      return { text: show(cell.score), status, tooltip, background: cell.score == null ? undefined : seqColor(t), color: seqTextColor(t), onClick };
    });
    return {
      key: row.key,
      label: row.name,
      kind: row.kind,
      depth: row.depth,
      parent: row.parent,
      cells,
      badge: rtMode || (row.kind === "suite" && row.aggregation) ? undefined : row.meta?.higher_is_better === false ? <Badge tone="muted">↓ better</Badge> : undefined,
      onLabelClick: () => setSearch({ scope: `task:${row.name}` }),
    };
  });

  const columns = data.subjects.map((sub) => ({
    key: sub.key,
    title: `${sub.label}${sub.key === baseline ? " (baseline)" : ""} — click to sort rows by delta in this column`,
    label: (
      <>
        <ModelDot slot={ctx.slots[sub.key]} baseline={sub.key === baseline} />
        <span>{sub.model.step != null ? ctx.labelOf(sub.key).replace(`${sub.model.series_label} `, "") : sub.model.series_label}</span>
      </>
    ),
    sub: sub.model.step != null ? sub.model.series_label : sub.kind === "model" ? `${sub.run_ids.length} run${sub.run_ids.length === 1 ? "" : "s"}` : "run",
    active: rsort === `delta:${sub.key}`,
    onClick: baseline && sub.key !== baseline ? () => setSearch({ rsort: rsort === `delta:${sub.key}` ? undefined : `delta:${sub.key}` }, { replace: true }) : undefined,
  }));

  return (
    <ChartPanel
      title="Heatmap"
      caption={
        rtMode
          ? mode === "delta"
            ? "Runtime change vs baseline: teal faster, ochre slower, scaled by the change relative to the baseline's time. Suites sum their tasks."
            : mode === "rank"
              ? "Rank within each row, 1 is fastest."
              : `${RUNTIME_MODE_LABEL[rtMode]} runtime per task. Stronger teal is faster within each row; suites sum their tasks.`
          : wantsDelta && !baseline
          ? "Delta vs baseline."
          : mode === "abs"
          ? `Absolute scores. Stronger teal is better ${globalScale ? "on one scale for all percent rows" : "within each row"}.`
          : mode === "delta"
            ? "Delta vs baseline: teal better, ochre worse. A dot marks significance; faded cells are not resolved; hatched cells lack data."
            : "Rank within each row, 1 is best."
      }
      refetching={matrix.isFetching}
      csv={() => ({
        header: ["row", ...data.subjects.map((s) => s.label)],
        rows: data.rows.map((r) => [r.name, ...r.cells.map((cl) => (mode === "delta" ? cl.delta?.delta ?? null : cl.score))]),
      })}
      legend={
        <div className={c.toolbar} style={{ marginBottom: 0 }}>
          <Segmented
            value={wantsDelta ? "delta" : mode}
            onChange={(v) => setSearch({ mode: v === "abs" ? undefined : v }, { replace: true })}
            label="Heatmap mode"
            options={[
              { value: "abs", label: "Absolute" },
              { value: "delta", label: "Delta vs baseline", title: baseline ? undefined : "Set a baseline first" },
              { value: "rank", label: "Rank" },
            ]}
          />
          {mode === "abs" && !wantsDelta && !rtMode && (
            <Segmented
              value={globalScale ? "global" : "row"}
              onChange={(v) => setSearch({ scale: v === "row" ? undefined : v }, { replace: true })}
              label="Color scale"
              options={[
                { value: "row", label: "Per-row scale" },
                { value: "global", label: "Global scale" },
              ]}
            />
          )}
          <Select
            size="sm"
            label="Sort rows"
            value={rsort.startsWith("delta:") ? rsort : rsort}
            onChange={(v) => setSearch({ rsort: v === "name" ? undefined : v }, { replace: true })}
            options={[
              { value: "name", label: "Suite tree" },
              { value: "spread", label: "Spread across subjects" },
              ...(baseline ? data.subjects.filter((s) => s.key !== baseline).map((s) => ({ value: `delta:${s.key}`, label: `Delta: ${ctx.labelOf(s.key)}` })) : []),
            ]}
          />
          <span className="spacer" />
          {mode === "delta" ? (
            <span className={c.legendRamp}>
              {rtMode ? "slower" : "worse"}
              <span className={c.ramp}>
                {[-3, -2, -1, 0, 1, 2, 3].map((st) => (
                  <span key={st} style={{ background: divColor(st) }} />
                ))}
              </span>
              {rtMode ? "faster" : "better"}
            </span>
          ) : (
            <span className={c.legendRamp}>
              {mode === "rank" ? (rtMode ? "slowest" : "worst") : rtMode ? "slow" : "low"}
              <span className={c.ramp}>
                {[0, 1, 2, 3, 4, 5, 6].map((st) => (
                  <span key={st} style={{ background: `var(--seq-${st})` }} />
                ))}
              </span>
              {mode === "rank" ? (rtMode ? "fastest" : "best") : rtMode ? "fast" : "high"}
            </span>
          )}
          <Legend items={[{ label: "missing", color: "var(--border-strong)" }]} />
        </div>
      }
    >
      {wantsDelta && !baseline ? (
        <EmptyState
          title="Delta mode needs a baseline"
          actions={
            data.subjects[0] && (
              <Button variant="primary" onClick={() => setBaseline(data.subjects[0].key, data.subjects[0].label)}>
                Use {ctx.labelOf(data.subjects[0].key)} as baseline
              </Button>
            )
          }
        >
          Star a subject above to compare every column with it. Cells then show paired deltas with significance.
        </EmptyState>
      ) : (
        <Heatmap columns={columns} rows={heatRows} cellWidth={80} maxHeight="calc(100vh - 360px)" label="Compare heatmap" />
      )}
    </ChartPanel>
  );
}
