import type { MatrixRow } from "@contract/api-types";
import { useMemo } from "react";
import { ChartPanel, Legend } from "@/charts/core";
import { type MPanel, SmallMultiples } from "@/charts/SmallMultiples";
import { useAppNavigate } from "@/components/AppLink";
import { EmptyState, Skeleton } from "@/components/primitives";
import { formatStep, sig3 } from "@/lib/format";
import { catColor } from "@/lib/scales";
import type { CompareCtx } from "./types";

const MAX_PANELS = 24;

export function ProgressionView(ctx: CompareCtx) {
  const { matrix, baseline } = ctx;
  const navigate = useAppNavigate();
  const data = matrix.data;
  const { panels, families, refs, available } = useMemo(() => {
    if (!data) return { panels: [] as MPanel[], families: [] as string[], refs: [] as string[], available: 0 };
    const withStep = data.subjects.map((s, i) => ({ s, i })).filter(({ s }) => s.model.step != null);
    const refs = data.subjects.map((s, i) => ({ s, i })).filter(({ s }) => s.model.step == null);
    const families = Array.from(new Set(withStep.map(({ s }) => s.model.series)));
    let rows: MatrixRow[];
    if (ctx.scope.startsWith("suite:")) rows = data.rows.filter((r) => r.depth === 1 || (r.depth === 0 && r.kind === "suite"));
    else if (ctx.scope.startsWith("task:")) rows = data.rows;
    else rows = data.rows.filter((r) => (r.kind === "suite" && r.depth <= 1) || (r.kind === "task" && r.depth === 0));
    const available = rows.length;
    rows = rows.slice(0, MAX_PANELS);
    const panels: MPanel[] = rows.map((row) => {
      const pct = (row.meta?.display_format ?? "percent") === "percent";
      const scale = pct ? 100 : 1;
      return {
        key: row.key,
        title: row.name,
        subtitle: row.kind === "suite" ? "suite" : undefined,
        series: families.map((fam, fi) => {
          const members = withStep.filter(({ s }) => s.model.series === fam).sort((a, b) => (a.s.model.step ?? 0) - (b.s.model.step ?? 0));
          return {
            key: fam,
            label: data.subjects[members[0].i].model.series_label,
            color: catColor(fi),
            points: members.map(({ s, i }) => {
              const cell = row.cells[i];
              return {
                x: s.model.step!,
                y: cell.score == null ? null : cell.score * scale,
                lo: cell.score != null && cell.stderr != null ? (cell.score - 1.96 * cell.stderr) * scale : null,
                hi: cell.score != null && cell.stderr != null ? (cell.score + 1.96 * cell.stderr) * scale : null,
                id: s.run?.run_id ?? s.run_ids[0],
              };
            }),
          };
        }),
        refs: refs.map(({ s, i }) => ({
          key: s.key,
          label: ctx.labelOf(s.key),
          value: row.cells[i].score == null ? null : row.cells[i].score! * scale,
          color: s.key === baseline ? "var(--teal-line)" : "var(--muted)",
        })),
      };
    });
    return { panels, families, refs: refs.map(({ s }) => s.key), available };
  }, [data, ctx, baseline]);

  if (matrix.isLoading) return <Skeleton height={460} />;
  if (!families.length) {
    return (
      <EmptyState title="No checkpoints to plot" compact>
        Progression needs subjects with a known training step. Add several checkpoints of one model, or open a Model page.
      </EmptyState>
    );
  }
  return (
    <ChartPanel
      title="Progression"
      caption="Score against training step, one line per model series, with 95% bands. Subjects without a step are dashed reference lines."
      legend={
        <Legend
          items={[
            ...families.map((f, i) => ({ label: data!.subjects.find((s) => s.model.series === f)?.model.series_label ?? f, color: catColor(i) })),
            ...refs.map((k) => ({ label: ctx.labelOf(k), color: k === baseline ? "var(--teal-line)" : "var(--muted)", dashed: true })),
          ]}
        />
      }
      csv={() => ({
        header: ["panel", "series", "step", "score"],
        rows: panels.flatMap((p) => p.series.flatMap((s) => s.points.map((pt) => [p.title, s.label, pt.x, pt.y]))),
      })}
    >
      <SmallMultiples
        panels={panels}
        formatX={(v) => formatStep(v)}
        formatY={(v) => sig3(v)}
        xLabel="step"
        onPointClick={(_s, p) => p.id && navigate(`/runs/${p.id}`)}
      />
      {available > panels.length && (
        <p className="t-caption" style={{ marginTop: 6 }}>
          Showing the first {panels.length} of {available} panels. Pick a suite or task scope to see others.
        </p>
      )}
    </ChartPanel>
  );
}
