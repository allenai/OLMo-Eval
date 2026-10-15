import * as Popover from "@radix-ui/react-popover";
import { useParams } from "@tanstack/react-router";
import { ArrowRightLeft, Plus, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useModelSeries, useProgression, useSearch, useSuites } from "@/api/hooks/catalog";
import { useResolveSubjects } from "@/api/hooks/compare";
import { ChartPanel, Legend } from "@/charts/core";
import { Heatmap, type HeatRow } from "@/charts/Heatmap";
import { type MPanel, SmallMultiples } from "@/charts/SmallMultiples";
import { buildHref, useAppNavigate } from "@/components/AppLink";
import { MetaItem, PageHeader } from "@/components/PageHeader";
import { Badge, Button, Chip, ErrorPanel, Panel, SearchInput, Segmented, Select, Skeleton, Tip, uiStyles as ui } from "@/components/primitives";
import { formatCount, formatDelta, formatScore, formatStep, shortHash, sig3 } from "@/lib/format";
import { divColor, divStep, extent, normalize, seqColor, seqTextColor, divTextColor } from "@/lib/scales";
import { parseList } from "@/lib/url";
import { useSearchParams } from "@/state/nav";
import { addToTray, pushRecent } from "@/state/prefs";
import { NotFoundPage } from "../NotFoundPage";
import { RunsTable } from "../runs/RunsTable";
import { pressable } from "@/lib/pressable";

type ModelSearch = { variant?: string; scope?: string; tasks?: string; x?: string; refs?: string; merge?: string; delta?: string; sel?: string };

const REF_COLORS = ["var(--cat-2)", "var(--cat-5)", "var(--cat-6)", "var(--cat-3)"];

function AddReference({ onAdd }: { onAdd: (key: string) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const res = useSearch(q);
  const items = (res.data?.groups ?? []).flatMap((g) => g.items).filter((i) => i.subject);
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button type="button" className={ui.addChip}>
          <Plus /> Reference
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={ui.popover} align="start" sideOffset={4}>
          <div className={ui.facet}>
            <div className={ui.facetHead}>
              <span className="t-overline">Add a reference model</span>
              <SearchInput autoFocus placeholder="e.g. Llama, Qwen3" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <div className={ui.facetList}>
              {items.map((i) => (
                <div key={`${i.type}:${i.key}`} className={ui.facetItem} {...pressable(() => (onAdd(i.subject!), setOpen(false)))}>
                  <span className={ui.facetValue}>{i.label}</span>
                  <span className={ui.facetCount}>{i.type}</span>
                </div>
              ))}
              {!q && <div className="t-caption" style={{ padding: 8 }}>References draw as dashed horizontal lines.</div>}
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export function ModelPage() {
  const { series: raw } = useParams({ strict: false }) as { series: string };
  const series = decodeURIComponent(raw);
  const [search, setSearch] = useSearchParams<ModelSearch>();
  const navigate = useAppNavigate();
  const detail = useModelSeries(series);
  const suites = useSuites();
  const refs = parseList(search.refs);
  const refInfo = useResolveSubjects(refs);
  const xMode = (search.x as "step" | "tokens" | "date") ?? "step";
  const [brush, setBrush] = useState<[number, number] | null>(null);
  const progression = useProgression({
    series,
    settings_hash: search.variant,
    scope: search.scope,
    tasks: search.tasks,
    merge: search.merge,
    references: refs.join(",") || undefined,
  });

  useEffect(() => {
    if (detail.data) pushRecent({ type: "model", key: series, label: detail.data.series_label, href: `/models/${encodeURIComponent(series)}` });
  }, [detail.data, series]);

  const xOf = useCallback(
    (p: { step: number | null; tokens_seen: number | null; date: string }) =>
      xMode === "tokens" ? (p.tokens_seen ?? 0) / 1e9 : xMode === "date" ? new Date(p.date).getTime() : (p.step ?? 0),
    [xMode],
  );

  // Variants ordered by run count; a selected variant collapses to one line.
  const variantOf = useMemo(() => new Map((detail.data?.checkpoints ?? []).map((ck) => [ck.model.model_id, ck.settings_hash])), [detail.data]);
  const variantKeys = useMemo(() => {
    const vs = [...(detail.data?.variants ?? [])].sort((a, b) => b.n_runs - a.n_runs).map((v) => v.settings_hash);
    return search.variant || vs.length < 2 ? [search.variant ?? ""] : vs;
  }, [detail.data, search.variant]);

  const panels: MPanel[] = useMemo(
    () =>
      (progression.data?.panels ?? []).map((p) => {
        const pct = (p.meta?.display_format ?? "percent") === "percent";
        const k = pct ? 100 : 1;
        return {
          key: p.key,
          title: p.name,
          subtitle: p.kind === "suite" ? "suite" : p.meta?.higher_is_better === false ? "lower is better" : undefined,
          // With every variant shown, draw one line per settings variant so checkpoints that share
          // a step but differ in settings do not zigzag into one line.
          series: variantKeys.map((vk, vi) => ({
            key: vk || series,
            label: variantKeys.length > 1 ? `${detail.data?.series_label ?? series} · ${shortHash(vk, 6)}` : (detail.data?.series_label ?? series),
            color: `var(--cat-${(vi % 6) + 1})`,
            points: p.points
              .filter((pt) => (pt.step != null || xMode === "date") && (variantKeys.length === 1 || variantOf.get(pt.model_id) === vk))
              .map((pt) => ({
                x: xOf(pt),
                y: pt.score == null ? null : pt.score * k,
                lo: pt.score != null && pt.stderr != null ? (pt.score - 1.96 * pt.stderr) * k : null,
                hi: pt.score != null && pt.stderr != null ? (pt.score + 1.96 * pt.stderr) * k : null,
                id: pt.run_id,
              }))
              .sort((a, b) => a.x - b.x),
          })),
          refs: p.references.map((r, i) => ({ key: r.subject, label: r.label, value: r.score == null ? null : r.score * k, color: REF_COLORS[i % REF_COLORS.length] })),
        };
      }),
    [progression.data, xMode, xOf, detail.data, series, variantKeys, variantOf],
  );

  if (detail.error) return <NotFoundPage what={`Model ${series}`} />;
  const d = detail.data;
  const checkpoints = (d?.checkpoints ?? []).filter((c) => !search.variant || c.settings_hash === search.variant);
  const inBrush = brush ? checkpoints.filter((c) => c.model.step != null && c.model.step >= brush[0] && c.model.step <= brush[1]) : [];

  // Checkpoints x tasks table, from the progression panels.
  const steps = Array.from(new Set((progression.data?.panels ?? []).flatMap((p) => p.points.map((pt) => pt.model_id))));
  const byModel = new Map(checkpoints.map((c) => [c.model.model_id, c]));
  const ordered = steps.map((id) => byModel.get(id)).filter((c): c is NonNullable<typeof c> => !!c).sort((a, b) => (b.model.step ?? 0) - (a.model.step ?? 0));
  const deltaMode = search.delta === "1";
  const cols = progression.data?.panels ?? [];
  const colExt = cols.map((p) => extent(p.points.map((pt) => pt.score)));
  const heatRows: HeatRow[] = ordered.map((ck, ri) => ({
    key: ck.model.model_id,
    label: `step ${ck.model.step != null ? formatCount(ck.model.step) : "—"}${d && d.variants.length > 1 ? ` · ${shortHash(ck.settings_hash, 4)}` : ""}`,
    kind: "task",
    depth: 0,
    parent: null,
    onLabelClick: () => navigate(`/runs/${ck.run_ids[0]}`),
    cells: cols.map((p, j) => {
      const pt = p.points.find((x) => x.model_id === ck.model.model_id);
      const fmt = p.meta?.display_format ?? "percent";
      const hib = p.meta?.higher_is_better ?? true;
      const prevCk = ordered[ri + 1];
      const prev = prevCk ? p.points.find((x) => x.model_id === prevCk.model.model_id) : undefined;
      if (!pt || pt.score == null) return { text: "", status: "missing" as const };
      const tooltip = (
        <div>
          <strong>{p.name}</strong> · step {formatCount(ck.model.step ?? 0)}
          <div className="mono">
            {formatScore(pt.score, fmt)} {prev?.score != null && <span className="muted">({formatDelta(pt.score - prev.score, fmt)} vs previous)</span>}
          </div>
          <div className="muted" style={{ fontSize: 11 }}>Click to open the run</div>
        </div>
      );
      if (deltaMode) {
        if (prev?.score == null) return { text: "—", tooltip, muted: true, onClick: () => navigate(`/runs/${pt.run_id}`) };
        const delta = pt.score - prev.score;
        const step = divStep((hib === false ? -delta : delta) / (fmt === "percent" ? 1 : 100), 0.03);
        return { text: formatDelta(delta, fmt), background: divColor(step), color: divTextColor(step), tooltip, onClick: () => navigate(`/runs/${pt.run_id}`) };
      }
      const t = normalize(pt.score, colExt[j], hib !== false);
      return { text: formatScore(pt.score, fmt), background: seqColor(t), color: seqTextColor(t), tooltip, onClick: () => navigate(`/runs/${pt.run_id}`) };
    }),
  }));

  const compareCheckpoints = (list: typeof checkpoints) =>
    navigate(buildHref("/compare", { subjects: [...list.map((c) => `m:${c.model.model_id}`), ...refs].join(","), view: "progression" }));

  return (
    <div className="page">
      <PageHeader
        crumbs={[{ label: "Models", href: "/models" }]}
        title={d?.series_label ?? series}
        titleExtra={d?.family && <Badge tone="muted">{d.family}</Badge>}
        meta={
          d ? (
            <>
              <MetaItem>{formatCount(d.n_runs)} runs</MetaItem>
              <MetaItem>{formatCount(d.checkpoints.length)} checkpoints</MetaItem>
              <MetaItem label="latest step">{formatCount(Math.max(0, ...d.checkpoints.map((c) => c.model.step ?? 0)))}</MetaItem>
              {d.series !== d.series_label && <MetaItem label="series"><span className="mono">{d.series}</span></MetaItem>}
            </>
          ) : (
            <Skeleton width={300} height={14} />
          )
        }
        actions={
          <>
            <Button size="sm" icon={<ArrowRightLeft />} onClick={() => checkpoints.forEach((c) => addToTray({ key: `m:${c.model.model_id}`, label: `${c.model.series_label} @ ${formatStep(c.model.step)}` }))}>
              Add all to tray
            </Button>
            <Button size="sm" variant="primary" onClick={() => compareCheckpoints(checkpoints)} disabled={checkpoints.length < 2}>
              Compare all checkpoints
            </Button>
          </>
        }
      >
        {d && d.variants.length > 0 && (
          <div className="row-wrap" style={{ gap: 6 }}>
            <span className="t-overline">Variants</span>
            <Chip value="All" active={!search.variant} onClick={() => setSearch({ variant: undefined })} />
            {d.variants.map((v) => (
              <Tip key={v.settings_hash} content={Object.keys(v.differs).length ? `Differs: ${Object.entries(v.differs).map(([k, val]) => `${k}=${JSON.stringify(val)}`).join(", ")}` : "Most common settings"}>
                <span>
                  <Chip
                    value={
                      <>
                        <span className="mono">{shortHash(v.settings_hash, 6)}</span> <span className="muted">({v.n_runs} runs{Object.keys(v.differs).length ? `, ${Object.keys(v.differs).join(", ")}` : ""})</span>
                      </>
                    }
                    active={search.variant === v.settings_hash}
                    onClick={() => setSearch({ variant: search.variant === v.settings_hash ? undefined : v.settings_hash })}
                  />
                </span>
              </Tip>
            ))}
          </div>
        )}
      </PageHeader>

      <ChartPanel
        id="progression"
        title="Progression"
        caption="Score by checkpoint with 95% bands. Dashed lines are reference models. Drag across a panel to select checkpoints; click a point to open its run."
        refetching={progression.isFetching && !progression.isLoading}
        csv={() => ({
          header: ["panel", "step", "tokens", "date", "score", "stderr", "run_id"],
          rows: (progression.data?.panels ?? []).flatMap((p) => p.points.map((pt) => [p.name, pt.step, pt.tokens_seen, pt.date, pt.score, pt.stderr, pt.run_id])),
        })}
        legend={
          <div className="row-wrap" style={{ gap: 10 }}>
            <span className="t-caption">x</span>
            <Segmented
              value={xMode}
              onChange={(v) => setSearch({ x: v === "step" ? undefined : v }, { replace: true })}
              label="X axis"
              options={[
                { value: "step", label: "Step" },
                { value: "tokens", label: "Tokens (B)" },
                { value: "date", label: "Date" },
              ]}
            />
            <Select
              size="sm"
              label="Scope"
              value={search.scope ?? ""}
              onChange={(v) => setSearch({ scope: v || undefined, tasks: undefined })}
              options={[{ value: "", label: "Top-level suites" }, ...(suites.data?.items ?? []).map((s) => ({ value: `suite:${s.suite_name}`, label: `${s.suite_name} children`, hint: s.n_children }))]}
              width={230}
            />
            <Segmented
              value={search.merge === "all" ? "all" : "latest"}
              onChange={(v) => setSearch({ merge: v === "latest" ? undefined : v }, { replace: true })}
              label="Runs per checkpoint"
              options={[
                { value: "latest", label: "Latest run" },
                { value: "all", label: "All runs" },
              ]}
            />
            {refs.map((r, i) => (
              <Chip key={r} value={<span style={{ color: REF_COLORS[i % REF_COLORS.length] }}>— {refInfo.data?.items.find((x) => x.key === r)?.label ?? r}</span>} onRemove={() => setSearch({ refs: refs.filter((x) => x !== r).join(",") || undefined })} />
            ))}
            <AddReference onAdd={(k) => setSearch({ refs: [...refs, k].join(",") })} />
            {brush && (
              <span className="row" style={{ gap: 6, marginLeft: "auto" }}>
                <span className="t-caption">
                  {inBrush.length} checkpoints, steps {formatCount(Math.round(brush[0]))}–{formatCount(Math.round(brush[1]))}
                </span>
                <Button size="sm" variant="primary" disabled={inBrush.length < 2} onClick={() => compareCheckpoints(inBrush)}>
                  Compare these {inBrush.length}
                </Button>
                <Button size="sm" variant="ghost" icon={<X />} onClick={() => setBrush(null)} aria-label="Clear selection" />
              </span>
            )}
          </div>
        }
      >
        {progression.isLoading ? (
          <Skeleton height={340} />
        ) : progression.error ? (
          <ErrorPanel error={progression.error} onRetry={() => progression.refetch()} />
        ) : (
          <>
            <SmallMultiples
              panels={panels}
              formatX={(v) => (xMode === "date" ? new Date(v).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : xMode === "tokens" ? `${sig3(v)}B` : formatStep(v))}
              formatY={(v) => sig3(v)}
              xLabel={xMode === "step" ? "step" : undefined}
              onPointClick={(_s, p) => p.id && navigate(`/runs/${p.id}`)}
              onBrush={xMode === "step" ? setBrush : undefined}
              brush={brush}
            />
            {(refs.length > 0 || variantKeys.length > 1) && (
              <div style={{ marginTop: 8 }}>
                <Legend
                  items={[
                    ...(variantKeys.length > 1
                      ? variantKeys.map((vk, vi) => ({ label: `variant ${shortHash(vk, 6)}`, color: `var(--cat-${(vi % 6) + 1})` }))
                      : []),
                    ...refs.map((r, i) => ({ label: refInfo.data?.items.find((x) => x.key === r)?.label ?? r, color: REF_COLORS[i % REF_COLORS.length], dashed: true })),
                  ]}
                />
              </div>
            )}
          </>
        )}
      </ChartPanel>

      <ChartPanel
        title="Checkpoints × tasks"
        caption={deltaMode ? "Change from the previous checkpoint: teal better, ochre worse (no significance test)." : "Scores per column. Stronger teal is better within each column. Newest checkpoint first."}
        legend={
          <Segmented
            value={deltaMode ? "delta" : "abs"}
            onChange={(v) => setSearch({ delta: v === "delta" ? "1" : undefined }, { replace: true })}
            label="Cell values"
            options={[
              { value: "abs", label: "Scores" },
              { value: "delta", label: "Delta vs previous checkpoint" },
            ]}
          />
        }
      >
        {progression.isLoading ? (
          <Skeleton height={240} />
        ) : (
          <Heatmap
            cornerLabel="Checkpoint"
            columns={cols.map((p) => ({ key: p.key, title: p.name, label: <span>{p.name}</span>, sub: p.kind }))}
            rows={heatRows}
            rowHeaderWidth={170}
            cellWidth={86}
            collapsible={false}
            maxHeight={520}
          />
        )}
      </ChartPanel>

      <Panel title="All runs of this model" pad="none">
        <RunsTable tableId="model-runs" params={{ series, sort: "-step", limit: 100 }} keyboard={false} maxHeight={520} hideColumns={["group"]} />
      </Panel>
    </div>
  );
}
