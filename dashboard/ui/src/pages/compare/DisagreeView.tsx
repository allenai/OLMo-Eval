import type { CompareInstancesResponse, ContingencyTaskRow } from "@contract/api-types";
import { Check, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useCompareInstances, useContingency } from "@/api/hooks/compare";
import { useInstanceDetail } from "@/api/hooks/runs";
import { ChartPanel, ChartTooltip, Legend, TipRow, useSize, useTooltip } from "@/charts/core";
import { ContingencyTable } from "@/charts/basic";
import { Checkbox, cx, EmptyState, ModelDot, Panel, Segmented, Select, Skeleton } from "@/components/primitives";
import { formatCount, formatDelta } from "@/lib/format";
import { themeStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { ChoicesTable, DiffText, OutputView, parseRecord, PromptView, Section } from "../run/instanceRender";
import c from "./compare.module.css";
import type { CompareCtx } from "./types";

const CELL_COLORS = {
  both_right: "var(--seq-1)",
  only_a: "var(--div-n2)",
  only_b: "var(--div-p2)",
  both_wrong: "var(--div-0)",
} as const;

type CellKey = keyof typeof CELL_COLORS;

function StackedBars({
  items,
  aLabel,
  bLabel,
  activeTask,
  activeCell,
  onSelect,
}: {
  items: ContingencyTaskRow[];
  aLabel: string;
  bLabel: string;
  activeTask?: string;
  activeCell?: string;
  onSelect: (task: string, cell: CellKey) => void;
}) {
  const tip = useTooltip();
  const labels: Record<CellKey, string> = { both_right: "both right", only_a: `only ${aLabel}`, only_b: `only ${bLabel}`, both_wrong: "both wrong" };
  return (
    <div className={c.stack} onMouseLeave={tip.hide}>
      {items.map((row) => {
        const ct = row.contingency;
        if (!ct) return null;
        return (
          <div key={row.task_name} className={cx(c.stackRow, row.task_name === activeTask && c.stackRowActive)}>
            <span className="truncate" title={row.task_name}>{row.task_name}</span>
            <span className={c.stackBar}>
              {(["both_right", "only_a", "only_b", "both_wrong"] as CellKey[]).map((key) =>
                ct[key] > 0 ? (
                  <button
                    type="button"
                    key={key}
                    className={c.stackSeg}
                    aria-label={`${row.task_name}: ${labels[key]}, ${formatCount(ct[key])} instances. List them.`}
                    aria-pressed={row.task_name === activeTask && activeCell === key}
                    style={{ flex: ct[key], background: CELL_COLORS[key], outline: row.task_name === activeTask && activeCell === key ? "2px solid var(--accent)" : undefined }}
                    onClick={() => onSelect(row.task_name, key)}
                    onBlur={tip.hide}
                    onMouseMove={(e) =>
                      tip.show(
                        e,
                        <>
                          <div style={{ fontWeight: 800, marginBottom: 4 }}>{row.task_name}</div>
                          {(["both_right", "only_a", "only_b", "both_wrong"] as CellKey[]).map((k) => (
                            <TipRow key={k} color={CELL_COLORS[k]} label={labels[k]} value={`${formatCount(ct[k])} (${((ct[k] / ct.n_shared) * 100).toFixed(1)}%)`} />
                          ))}
                          <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Click a segment to list its instances</div>
                        </>,
                      )
                    }
                  />
                ) : null,
              )}
            </span>
            <span className="mono" style={{ textAlign: "right", color: ct.net < 0 ? "var(--better)" : ct.net > 0 ? "var(--worse)" : "var(--muted)" }}>
              {ct.net < 0 ? "+" : ct.net > 0 ? "−" : ""}
              {Math.abs(ct.net)}
            </span>
          </div>
        );
      })}
      <ChartTooltip state={tip.state} />
    </div>
  );
}

function SideBySide({ trA, trB, native, aLabel, bLabel }: { trA: number | null; trB: number | null; native: string; aLabel: string; bLabel: string }) {
  const da = useInstanceDetail(trA, native);
  const db = useInstanceDetail(trB, native);
  const [diff, setDiff] = useState(true);
  const ra = useMemo(() => parseRecord(da.data?.prediction ?? null, da.data?.request ?? null), [da.data]);
  const rb = useMemo(() => parseRecord(db.data?.prediction ?? null, db.data?.request ?? null), [db.data]);
  if (da.isLoading || db.isLoading) return <Skeleton height={240} />;
  const samePrompt = JSON.stringify(da.data?.request?.request) === JSON.stringify(db.data?.request?.request);
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className="row" style={{ gap: 10 }}>
        <span className="mono" style={{ fontWeight: 500 }}>{native}</span>
        <span className="spacer" />
        <Checkbox checked={diff} onChange={setDiff} label="Word diff (B vs A)" />
      </div>
      <Section label={samePrompt ? "Prompt (shared)" : "Prompt (differs between A and B, showing A)"}>
        <PromptView rec={ra} />
      </Section>
      {ra.gold && !ra.choices && (
        <Section label="Gold">
          <span className="mono">{ra.gold}</span>
        </Section>
      )}
      <div className="grid-12" style={{ gap: 10 }}>
        {[
          { label: aLabel, rec: ra, metrics: da.data?.metrics, score: da.data?.primary_score },
          { label: bLabel, rec: rb, metrics: db.data?.metrics, score: db.data?.primary_score },
        ].map((side, i) => (
          <div key={i} className="span-6">
            <Section label={<span className="truncate">{i === 0 ? "A" : "B"} · {side.label} · score {side.score ?? "—"}</span>}>
              {side.rec.choices ? (
                <ChoicesTable choices={side.rec.choices} />
              ) : i === 1 && diff && ra.output != null && side.rec.output != null ? (
                <DiffText a={ra.output} b={side.rec.output} />
              ) : (
                <OutputView rec={side.rec} compact />
              )}
            </Section>
          </div>
        ))}
      </div>
    </div>
  );
}

function PairMode(ctx: CompareCtx) {
  const { keys, baseline, search, setSearch } = ctx;
  const a = search.a && keys.includes(search.a) ? search.a : (baseline ?? keys[0]);
  const b = search.b && keys.includes(search.b) && search.b !== a ? search.b : keys.find((k) => k !== a)!;
  const contingency = useContingency({ a, b, group: ctx.group ?? null, scope: ctx.scope, metric: ctx.metric });
  const items = useMemo(() => (contingency.data?.items ?? []).filter((r) => r.contingency), [contingency.data]);
  const task = search.task && items.some((r) => r.task_name === search.task) ? search.task : items[0]?.task_name;
  const cell = (search.cell as CellKey | undefined) ?? undefined;
  const instBody = task
    ? { subjects: [a, b], group: ctx.group ?? null, task_name: task, metric: ctx.metric, filter: cell ? ("cell" as const) : ("disagree" as const), cell: cell ?? null, a, b, include_previews: true, limit: 300 }
    : null;
  const instances = useCompareInstances(instBody);
  const list = instances.data?.items ?? [];
  const native = search.inst && list.some((r) => r.native_id === search.inst) ? search.inst : list[0]?.native_id;
  const subjectOptions = keys.map((k) => ({ value: k, label: ctx.labelOf(k) }));
  const row = items.find((r) => r.task_name === task);
  const totals = contingency.data?.totals;
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className={c.toolbar}>
        <Select size="sm" label="A" value={a} onChange={(v) => setSearch({ a: v, inst: undefined }, { replace: true })} options={subjectOptions} width={240} />
        <Select size="sm" label="B" value={b} onChange={(v) => setSearch({ b: v, inst: undefined }, { replace: true })} options={subjectOptions.filter((o) => o.value !== a)} width={240} />
        {totals && (
          <span className="t-caption">
            Across {formatCount(totals.n_shared)} shared instances: B gains <strong className="mono" style={{ color: "var(--better)" }}>{formatCount(totals.only_b)}</strong>, loses{" "}
            <strong className="mono" style={{ color: "var(--worse)" }}>{formatCount(totals.only_a)}</strong>
          </span>
        )}
      </div>
      <div className="grid-12">
        <div className="span-7">
          <ChartPanel
            title="Where A and B disagree"
            caption="Per task: both right, only A right, only B right, both wrong. Tasks where B gains most come first. The number is net instances B gets right that A misses."
            legend={
              <Legend
                items={[
                  { label: "both right", color: CELL_COLORS.both_right },
                  { label: `only A`, color: CELL_COLORS.only_a },
                  { label: `only B`, color: CELL_COLORS.only_b },
                  { label: "both wrong", color: CELL_COLORS.both_wrong },
                ]}
              />
            }
            refetching={contingency.isFetching}
          >
            {contingency.isLoading ? <Skeleton height={300} /> : items.length ? (
              <div style={{ maxHeight: 460, overflow: "auto" }} role="region" aria-label="Per-task disagreement bars" tabIndex={0}>
                <StackedBars items={items} aLabel="A" bLabel="B" activeTask={task} activeCell={cell} onSelect={(t, k) => setSearch({ task: t, cell: k, inst: undefined }, { replace: true })} />
              </div>
            ) : (
              <EmptyState title="No paired tasks" compact>A and B share no binary or bounded tasks in this scope.</EmptyState>
            )}
          </ChartPanel>
        </div>
        <div className="span-5">
          <Panel title={task ?? "Task"} caption={row && row.delta != null ? `B − A ${formatDelta(-row.delta, "percent")} · click a cell to filter` : undefined}>
            {row?.contingency ? (
              <ContingencyTable
                both_right={row.contingency.both_right}
                only_a={row.contingency.only_b}
                only_b={row.contingency.only_a}
                both_wrong={row.contingency.both_wrong}
                aLabel="B"
                bLabel="A"
                onCell={(k) => setSearch({ cell: k === "only_a" ? "only_b" : k === "only_b" ? "only_a" : k, inst: undefined }, { replace: true })}
              />
            ) : (
              <Skeleton height={120} />
            )}
          </Panel>
        </div>
      </div>
      {task && (
        <div className="grid-12">
          <div className="span-4">
            <Panel
              title={`${cell ? cell.replace("_", " ").replace("only a", "only A").replace("only b", "only B") : "Disagreements"} · ${formatCount(instances.data?.total ?? 0)}`}
              pad="none"
              refetching={instances.isFetching}
            >
              <div
                className={c.instList}
                role="listbox"
                aria-label="Disagreeing instances"
                tabIndex={0}
                aria-activedescendant={native ? `dis-${native}` : undefined}
                onKeyDown={(e) => {
                  const step = e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : 0;
                  if (!step || !list.length) return;
                  e.preventDefault();
                  e.stopPropagation();
                  const i = Math.max(0, list.findIndex((r) => r.native_id === native));
                  const next = list[Math.min(list.length - 1, Math.max(0, i + step))];
                  setSearch({ inst: next.native_id }, { replace: true });
                  document.getElementById(`dis-${next.native_id}`)?.scrollIntoView({ block: "nearest" });
                }}
              >
                {list.map((r) => (
                  <div
                    key={r.native_id}
                    id={`dis-${r.native_id}`}
                    role="option"
                    aria-selected={r.native_id === native}
                    className={cx(c.instRow, r.native_id === native && c.instRowActive)}
                    style={{ ["--n" as string]: 2 }}
                    onClick={() => setSearch({ inst: r.native_id }, { replace: true })}
                  >
                    <span className="mono truncate-start" title={r.native_id}><bdi>{r.native_id}</bdi></span>
                    {r.correct.map((ok, i) => (
                      <span key={i} title={i === 0 ? "A" : "B"}>{ok ? <Check size={13} color="var(--status-complete)" /> : <X size={13} color="var(--status-failed)" />}</span>
                    ))}
                    <span className="truncate muted">{r.output_previews?.[1] ?? r.prompt_preview}</span>
                  </div>
                ))}
                {!instances.isLoading && !list.length && <span className="t-caption" style={{ padding: 12 }}>No instances in this cell.</span>}
              </div>
            </Panel>
          </div>
          <div className="span-8">
            <Panel title="Side by side">
              {native ? (
                <SideBySide
                  trA={instances.data?.subjects[0]?.task_result_id ?? null}
                  trB={instances.data?.subjects[1]?.task_result_id ?? null}
                  native={native}
                  aLabel={ctx.labelOf(a)}
                  bLabel={ctx.labelOf(b)}
                />
              ) : (
                <EmptyState title="Pick an instance" compact />
              )}
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}

const GRID_CELL = 10;
const GRID_VIEW_H = 560;

/**
 * Instances × subjects agreement grid. The canvas is only as tall as the viewport and redraws the
 * visible rows on scroll, so 10k instances stay cheap. Up/Down (or j/k) move the selected row.
 */
function AgreementCanvas({
  data,
  selected,
  onSelect,
}: {
  data: CompareInstancesResponse;
  selected?: string;
  onSelect: (native: string) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [wrapRef, { width }] = useSize<HTMLDivElement>();
  const [scrollTop, setScrollTop] = useState(0);
  // Colors are read from CSS at draw time, so redraw when the theme changes.
  const theme = useStore(themeStore);
  const tip = useTooltip();
  const cell = GRID_CELL;
  const n = data.subjects.length;
  const rows = data.items;
  const gridW = n * cell;
  const marginW = Math.max(60, Math.min(160, width - gridW - 20));
  const canvasW = gridW + marginW + 12;
  const totalH = rows.length * cell;
  const viewH = Math.max(1, Math.min(GRID_VIEW_H, totalH));
  const first = Math.floor(scrollTop / cell);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasW * dpr;
    canvas.height = viewH * dpr;
    const g = canvas.getContext("2d")!;
    g.scale(dpr, dpr);
    const v = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const colors = { sel: v("--accent-soft"), missing: v("--row"), ok: v("--teal-line"), wrong: v("--border-strong"), bar: v("--seq-3") };
    g.clearRect(0, 0, canvasW, viewH);
    const offset = scrollTop - first * cell;
    const last = Math.min(rows.length, first + Math.ceil(viewH / cell) + 1);
    for (let i = first; i < last; i++) {
      const r = rows[i];
      const y = (i - first) * cell - offset;
      if (r.native_id === selected) {
        g.fillStyle = colors.sel;
        g.fillRect(0, y, canvasW, cell);
      }
      r.correct.forEach((ok, j) => {
        const x = j * cell;
        if (ok == null) {
          g.fillStyle = colors.missing;
          g.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        } else if (ok) {
          g.fillStyle = colors.ok;
          g.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        } else {
          g.strokeStyle = colors.wrong;
          g.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
        }
      });
      const frac = r.n_present ? r.n_correct / r.n_present : 0;
      g.fillStyle = colors.bar;
      g.fillRect(gridW + 12, y + 2, frac * marginW, cell - 4);
    }
  }, [rows, selected, gridW, marginW, canvasW, viewH, scrollTop, first, cell, theme]);
  const locate = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect();
    const i = Math.floor((e.clientY - rect.top + scrollTop) / cell);
    const j = Math.floor((e.clientX - rect.left) / cell);
    return { i, j };
  };
  const index = rows.findIndex((r) => r.native_id === selected);
  const select = (i: number) => {
    const r = rows[Math.max(0, Math.min(rows.length - 1, i))];
    if (!r) return;
    onSelect(r.native_id);
    const el = scrollRef.current;
    if (!el) return;
    const y = rows.indexOf(r) * cell;
    if (y < el.scrollTop) el.scrollTop = y;
    else if (y + cell > el.scrollTop + el.clientHeight) el.scrollTop = y + cell - el.clientHeight;
  };
  return (
    <div ref={wrapRef}>
      <div
        ref={scrollRef}
        style={{ height: viewH, overflow: "auto", position: "relative" }}
        onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        tabIndex={0}
        role="group"
        aria-label={`Agreement grid, ${formatCount(rows.length)} instances by ${n} subjects. Up and Down move the selected instance.`}
        onKeyDown={(e) => {
          const step = e.key === "ArrowDown" || e.key === "j" ? 1 : e.key === "ArrowUp" || e.key === "k" ? -1 : e.key === "PageDown" ? 40 : e.key === "PageUp" ? -40 : 0;
          if (!step) return;
          e.preventDefault();
          e.stopPropagation();
          select(index < 0 ? 0 : index + step);
        }}
      >
        <div style={{ height: totalH, width: canvasW }}>
          <canvas
            ref={ref}
            style={{ position: "sticky", top: 0, width: canvasW, height: viewH, display: "block", cursor: "pointer" }}
            onClick={(e) => {
              const { i } = locate(e);
              if (rows[i]) onSelect(rows[i].native_id);
            }}
            onMouseMove={(e) => {
              const { i, j } = locate(e);
              const r = rows[i];
              if (!r) return tip.hide();
              tip.show(
                e,
                <>
                  <div style={{ fontWeight: 800 }}>{r.native_id}</div>
                  <TipRow label="subjects correct" value={`${r.n_correct} / ${r.n_present}`} />
                  {j < n && <TipRow label={`subject ${j + 1}`} value={r.correct[j] == null ? "missing" : r.correct[j] ? "correct" : "wrong"} />}
                  <div className="muted" style={{ marginTop: 4, fontSize: 11 }}>Click to see every output</div>
                </>,
              );
            }}
            onMouseLeave={tip.hide}
          />
        </div>
      </div>
      <ChartTooltip state={tip.state} />
    </div>
  );
}

function SubjectOutput({ trId, native, label, slot, baseline }: { trId: number | null; native: string; label: string; slot: number | null; baseline: boolean }) {
  const d = useInstanceDetail(trId, native);
  const rec = useMemo(() => parseRecord(d.data?.prediction ?? null, d.data?.request ?? null), [d.data]);
  return (
    <Panel
      title={
        <span className="row" style={{ gap: 6, textTransform: "none", letterSpacing: 0 }}>
          <ModelDot slot={slot} baseline={baseline} /> <span className="truncate">{label}</span>
        </span>
      }
      caption={d.data ? `score ${d.data.primary_score ?? "—"}` : undefined}
    >
      {!trId ? <span className="t-caption">No result for this task.</span> : d.isLoading ? <Skeleton height={100} /> : d.error ? <span className="t-caption">Instance missing.</span> : rec.choices ? <ChoicesTable choices={rec.choices} /> : <OutputView rec={rec} compact />}
    </Panel>
  );
}

function GridMode(ctx: CompareCtx) {
  const { keys, search, setSearch, matrix, baseline } = ctx;
  const tasks = (matrix.data?.rows ?? []).filter((r) => r.kind === "task" && r.meta?.kind !== "unbounded").map((r) => r.name);
  const task = search.task && tasks.includes(search.task) ? search.task : tasks[0];
  const filter = (search.filter as "all" | "disagree" | "all_wrong" | "baseline_wrong") ?? "disagree";
  const body = task ? { subjects: keys.slice(0, 30), group: ctx.group ?? null, task_name: task, metric: ctx.metric, filter, baseline: baseline ?? null, sort: "disagreement" as const, include_previews: false, limit: 10000 } : null;
  const inst = useCompareInstances(body);
  const selected = search.inst;
  const d = inst.data;
  return (
    <div className="col" style={{ gap: 12 }}>
      <div className={c.toolbar}>
        <Select size="sm" label="Task" value={task ?? ""} onChange={(v) => setSearch({ task: v, inst: undefined }, { replace: true })} options={tasks.map((t) => ({ value: t, label: t }))} width={280} />
        <Segmented
          value={filter}
          onChange={(v) => setSearch({ filter: v === "disagree" ? undefined : v, inst: undefined }, { replace: true })}
          label="Filter"
          options={[
            { value: "disagree", label: "Only disagreements" },
            { value: "all", label: "All" },
            { value: "all_wrong", label: "All wrong" },
            ...(baseline ? [{ value: "baseline_wrong" as const, label: "Baseline wrong" }] : []),
          ]}
        />
        {d && <span className="t-caption">{formatCount(d.total)} instances · sorted by how evenly subjects split</span>}
      </div>
      <div className="grid-12">
        <div className="span-5">
          <ChartPanel
            title="Agreement grid"
            caption="Rows are instances, columns are subjects. Filled = correct, outlined = wrong, gray = missing. The bar on the right is the share of subjects correct."
            refetching={inst.isFetching}
            legend={
              <div className="row-wrap" style={{ gap: 8 }}>
                {keys.slice(0, 30).map((k, i) => (
                  <span key={k} className="row t-caption" style={{ gap: 4 }}>
                    <span className="mono">{i + 1}</span>
                    <ModelDot slot={ctx.slots[k]} baseline={k === baseline} />
                    <span className="truncate" style={{ maxWidth: 140 }}>{ctx.labelOf(k)}</span>
                  </span>
                ))}
              </div>
            }
          >
            {inst.isLoading || !d ? <Skeleton height={300} /> : d.items.length ? (
              <AgreementCanvas data={d} selected={selected} onSelect={(native) => setSearch({ inst: native }, { replace: true })} />
            ) : (
              <EmptyState title="No instances match" compact />
            )}
            {d && (
              <div className="row-wrap t-caption" style={{ gap: 10, marginTop: 8 }}>
                {d.counts_by_n_correct.map((n, i) => (
                  <span key={i}>
                    {i} correct: <span className="mono">{formatCount(n)}</span>
                  </span>
                ))}
              </div>
            )}
          </ChartPanel>
        </div>
        <div className="span-7">
          {selected && task ? (
            <div className="col" style={{ gap: 8 }}>
              <span className="mono" style={{ fontWeight: 500 }}>{selected}</span>
              <div className={c.multi}>
                {keys.slice(0, 30).map((k, i) => (
                  <SubjectOutput key={k} trId={d?.subjects[i]?.task_result_id ?? null} native={selected} label={ctx.labelOf(k)} slot={ctx.slots[k]} baseline={k === baseline} />
                ))}
              </div>
            </div>
          ) : (
            <Panel>
              <EmptyState title="Click a row in the grid" compact>
                See every subject's output for that instance side by side.
              </EmptyState>
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

export function DisagreeView(ctx: CompareCtx) {
  const mode = ctx.search.x === "grid" ? "grid" : "pair";
  return (
    <div className="col" style={{ gap: 12 }}>
      <Segmented
        value={mode}
        onChange={(v) => ctx.setSearch({ x: v === "pair" ? undefined : v, inst: undefined }, { replace: true })}
        label="Disagreement mode"
        options={[
          { value: "pair", label: "Pair (A vs B)" },
          { value: "grid", label: `All subjects (${ctx.keys.length})` },
        ]}
      />
      {mode === "pair" ? <PairMode {...ctx} /> : <GridMode {...ctx} />}
    </div>
  );
}
