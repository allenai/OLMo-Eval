import type { InstanceRow, TaskResultRow } from "@contract/api-types";
import * as Dialog from "@radix-ui/react-dialog";
import { Check, ChevronLeft, ChevronRight, Columns2, Download, Minus, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useResolveSubjects, useSubjectTaskResult } from "@/api/hooks/compare";
import { useInstanceDetail, useInstances } from "@/api/hooks/runs";
import { type Column, DataTable } from "@/components/DataTable";
import { JsonTree } from "@/components/Json";
import {
  Badge,
  Button,
  Checkbox,
  CopyText,
  DebouncedSearchInput,
  EmptyState,
  ErrorPanel,
  IconButton,
  Input,
  Kbd,
  Panel,
  Segmented,
  Select,
  Skeleton,
  Slider,
  uiStyles as ui,
} from "@/components/primitives";
import { formatCount, formatScore, metricLabel, sig3 } from "@/lib/format";
import { useHotkeys } from "@/lib/keyboard";
import { parseList } from "@/lib/url";
import { trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { ChoicesTable, ExecutionView, JudgeView, OutputView, parseRecord, PromptView, Section, TrajectoryView } from "./instanceRender";
import s from "./run.module.css";
import { openSigned } from "./TaskDrilldown";
import { type RunCtx, scoreFormat } from "./types";

const CELL_TO_VS: Record<string, string> = { only_a: "gained", only_b: "lost", both_right: "both_right", both_wrong: "both_wrong", changed: "changed" };

function parseRange(value: string | undefined): [string | undefined, string | undefined] {
  if (!value) return [undefined, undefined];
  const [lo, hi] = value.split("-");
  return [lo || undefined, hi || undefined];
}

function Correctness({ value, correct, format }: { value: number | null; correct: boolean | null; format: "percent" | "raw" }) {
  if (value == null) return <span className="faint">—</span>;
  if (correct === true && (value === 1 || value === 0)) return <Check size={15} color="var(--status-complete)" aria-label="correct" />;
  if (correct === false && (value === 1 || value === 0)) return <X size={15} color="var(--status-failed)" aria-label="incorrect" />;
  return (
    <span className="mono" style={{ color: correct === true ? "var(--better)" : correct === false ? "var(--worse)" : undefined, fontSize: 12 }}>
      {format === "percent" ? value.toFixed(2) : sig3(value)}
    </span>
  );
}

function InstanceDetail({
  ctx,
  row,
  item,
  onPrev,
  onNext,
}: {
  ctx: RunCtx;
  row: TaskResultRow;
  item: InstanceRow | undefined;
  onPrev: () => void;
  onNext: () => void;
}) {
  const { search, setSearch } = ctx;
  const native = search.inst;
  const detail = useInstanceDetail(row.task_result_id, native);
  const tray = useStore(trayStore);
  const sideBySide = !!search.side;
  const sideSubject = search.side === "1" ? ctx.baseline : search.side;
  // The other subject: the side-by-side pick, else the baseline (for per-instance metric values).
  const otherSubject = sideBySide ? sideSubject : ctx.baseline;
  const other = useSubjectTaskResult(otherSubject, row.task_name);
  const otherDetail = useInstanceDetail(other.data?.task_result_id ?? null, otherSubject ? native : undefined);
  const otherInfo = useResolveSubjects(sideSubject ? [sideSubject] : []);
  const diff = search.diff === "1";
  const rec = useMemo(() => parseRecord(detail.data?.prediction ?? null, detail.data?.request ?? null), [detail.data]);
  const otherRec = useMemo(() => parseRecord(otherDetail.data?.prediction ?? null, otherDetail.data?.request ?? null), [otherDetail.data]);
  const fmt = scoreFormat(row);

  if (!native) {
    return (
      <Panel>
        <EmptyState title="Select an instance" compact>
          Click a row, or use <Kbd>j</Kbd> <Kbd>k</Kbd> to move through the list.
        </EmptyState>
      </Panel>
    );
  }
  const sideOptions = [
    ...(ctx.baseline ? [{ value: ctx.baseline, label: `Baseline · ${ctx.baselineInfo?.label ?? ""}` }] : []),
    ...tray.filter((t) => t.key !== ctx.baseline && t.key !== `r:${ctx.run.run_id}`).map((t) => ({ value: t.key, label: t.label })),
  ];
  const metrics = detail.data?.metrics ?? item?.metrics ?? {};
  const otherMetrics = otherDetail.data?.metrics;

  return (
    <Panel className={s.detail}>
      <div className="col" style={{ gap: 14 }}>
        <div className="row-wrap" style={{ gap: 8 }}>
          <CopyText text={native} display={<span className="mono" style={{ fontSize: 13, fontWeight: 500 }}>{native}</span>} />
          {item && (
            <span className="row" style={{ gap: 6 }}>
              <span className="t-caption">{metricLabel(row.primary_metric)}</span>
              <Correctness value={item.primary_score} correct={item.correct} format={fmt} />
              {item.baseline_score != null && (
                <>
                  <span className="t-caption">base</span>
                  <Correctness value={item.baseline_score} correct={item.baseline_correct} format={fmt} />
                </>
              )}
              {item.correct === true && item.baseline_correct === false && <Badge tone="teal">gained</Badge>}
              {item.correct === false && item.baseline_correct === true && <Badge tone="warn">lost</Badge>}
              {item.has_scoring_error && <Badge tone="danger">scoring error</Badge>}
            </span>
          )}
          <span className="spacer" />
          <IconButton size="sm" label="Previous instance (←)" icon={<ChevronLeft />} onClick={onPrev} />
          <IconButton size="sm" label="Next instance (→)" icon={<ChevronRight />} onClick={onNext} />
        </div>
        <div className="row-wrap" style={{ gap: 8 }}>
          <Segmented
            value={sideBySide ? "side" : "single"}
            onChange={(v) => setSearch({ side: v === "side" ? (ctx.baseline ? "1" : sideOptions[0]?.value) : undefined }, { replace: true })}
            label="Layout"
            options={[
              { value: "single", label: "Single" },
              { value: "side", label: <><Columns2 /> Side by side</>, title: sideOptions.length ? "Compare with the baseline or a tray subject (d)" : "Set a baseline or add runs to compare first" },
            ]}
          />
          {sideBySide && sideOptions.length > 1 && (
            <Select size="sm" value={sideSubject ?? ""} onChange={(v) => setSearch({ side: v === ctx.baseline ? "1" : v }, { replace: true })} options={sideOptions} width={240} />
          )}
          {sideBySide && <Checkbox checked={diff} onChange={(on) => setSearch({ diff: on ? "1" : undefined }, { replace: true })} label="Word diff" />}
          <span className="spacer" />
          {detail.data?.predictions_uri && (
            <Button size="sm" variant="ghost" icon={<Download />} onClick={() => openSigned(detail.data!.predictions_uri!)}>
              predictions.jsonl
            </Button>
          )}
        </div>
        {detail.isLoading ? (
          <div className="col" style={{ gap: 8 }}>
            <Skeleton height={80} />
            <Skeleton height={140} />
          </div>
        ) : detail.error ? (
          <ErrorPanel error={detail.error} onRetry={() => detail.refetch()} />
        ) : (
          <>
            {detail.data?.unavailable_reason && <div className="t-caption">Full record unavailable: {detail.data.unavailable_reason}</div>}
            <Section label="Prompt" extra={rec.requestType && <span className="mono t-caption">{rec.requestType}</span>}>
              <PromptView rec={rec} />
            </Section>
            {rec.gold != null && !rec.choices && (
              <Section label="Gold">
                <div className="mono" style={{ fontSize: 13 }}>{rec.gold}</div>
              </Section>
            )}
            {rec.choices && (
              <Section label="Choices" extra="teal outline = chosen · check = gold">
                <ChoicesTable choices={rec.choices} />
              </Section>
            )}
            {sideBySide ? (
              <div className={s.sideBySide}>
                <Section label="This run">
                  <OutputView rec={rec} compact />
                </Section>
                <Section label={<span className="truncate">{otherInfo.data?.items[0]?.label ?? "Other"}</span>}>
                  {other.isLoading || otherDetail.isLoading ? (
                    <Skeleton height={120} />
                  ) : !other.data?.task_result_id ? (
                    <span className="t-caption">This subject has no result for {row.task_name}.</span>
                  ) : otherDetail.error ? (
                    <span className="t-caption">This instance is not in the other subject's results.</span>
                  ) : otherRec.choices ? (
                    <ChoicesTable choices={otherRec.choices} />
                  ) : (
                    <OutputView rec={otherRec} compact diffAgainst={diff ? rec.output : null} />
                  )}
                </Section>
              </div>
            ) : (
              !rec.choices && (
                <Section label="Output">
                  <OutputView rec={rec} />
                </Section>
              )
            )}
            {rec.execution && (
              <Section label="Code execution">
                <ExecutionView exec={rec.execution} />
              </Section>
            )}
            {rec.scoringErrors.length > 0 && (
              <Section label="Scoring errors">
                <div className={s.text} style={{ color: "var(--danger-fg)" }}>{rec.scoringErrors.join("\n")}</div>
              </Section>
            )}
            {rec.judge && (
              <Section label="Judge">
                <JudgeView judge={rec.judge} />
              </Section>
            )}
            {rec.trajectory && (
              <Section label="Trajectory">
                <TrajectoryView trajectory={rec.trajectory} />
              </Section>
            )}
            <Section label="Metrics">
              <table style={{ borderCollapse: "collapse", fontSize: 12.5, width: "100%", maxWidth: 520 }}>
                {otherMetrics && (
                  <thead>
                    <tr className="t-caption">
                      <th style={{ textAlign: "left", fontWeight: 400, padding: "0 0 4px" }}>metric</th>
                      <th style={{ textAlign: "right", fontWeight: 400, padding: "0 8px 4px" }}>this run</th>
                      <th style={{ textAlign: "right", fontWeight: 400, padding: "0 0 4px" }}>{sideBySide && search.side !== "1" ? "other" : "baseline"}</th>
                    </tr>
                  </thead>
                )}
                <tbody>
                  {Object.entries(metrics).map(([k, v]) => (
                    <tr key={k} style={{ borderTop: "1px solid var(--border)" }}>
                      <td className="mono" style={{ padding: "4px 0", fontSize: 11.5 }}>
                        {metricLabel(k)}
                      </td>
                      <td className="num" style={{ padding: "4px 8px" }}>{v == null ? "—" : sig3(v)}</td>
                      {otherMetrics && <td className="num muted" style={{ padding: "4px 0" }}>{otherMetrics[k] == null ? "—" : sig3(otherMetrics[k]!)}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section label="Raw records">
              <div className="col" style={{ gap: 4 }}>
                <details>
                  <summary className="t-caption" style={{ cursor: "pointer" }}>prediction</summary>
                  <JsonTree value={detail.data?.prediction ?? null} filename={`${native}-prediction.json`} maxHeight={360} />
                </details>
                <details>
                  <summary className="t-caption" style={{ cursor: "pointer" }}>request</summary>
                  <JsonTree value={detail.data?.request ?? null} filename={`${native}-request.json`} maxHeight={360} />
                </details>
              </div>
            </Section>
          </>
        )}
      </div>
    </Panel>
  );
}

export function InstancesTab(ctx: RunCtx) {
  const { search, setSearch } = ctx;
  const tasks = useMemo(() => (ctx.tasks.data?.items ?? []).filter((t) => !t.error), [ctx.tasks.data]);
  const row = tasks.find((t) => t.task_name === search.task) ?? tasks[0];
  const kind = row ? (row.metric_meta[row.primary_metric ?? ""]?.kind ?? "bounded") : "bounded";
  const fmt = row ? scoreFormat(row) : "percent";
  const [lenMin, lenMax] = parseRange(search.len);
  const [scoreMin, scoreMax] = parseRange(search.score);
  const params = useMemo(
    () => ({
      baseline: ctx.baseline,
      correct: kind === "unbounded" ? undefined : search.correct,
      threshold: search.thr,
      vs: ctx.baseline && search.cell ? CELL_TO_VS[search.cell] : undefined,
      finish_reason: parseList(search.fin),
      len_min: lenMin,
      len_max: lenMax,
      score_min: scoreMin,
      score_max: scoreMax,
      has: parseList(search.has),
      q: search.iq,
      sort: search.isort,
      limit: 200,
    }),
    [ctx.baseline, kind, search.correct, search.thr, search.cell, search.fin, lenMin, lenMax, scoreMin, scoreMax, search.has, search.iq, search.isort],
  );
  const instances = useInstances(ctx.run.run_id, row?.task_result_id ?? null, params);
  const items = useMemo(() => instances.data?.pages.flatMap((p) => p.items) ?? [], [instances.data]);
  const total = instances.data?.pages[0]?.total;
  const idx = items.findIndex((i) => i.native_id === search.inst);
  const [narrow, setNarrow] = useState(() => typeof window !== "undefined" && window.innerWidth < 1024);
  useEffect(() => {
    const onResize = () => setNarrow(window.innerWidth < 1024);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Pick the first instance by default on wide screens so the detail pane is never empty.
  useEffect(() => {
    if (!narrow && !search.inst && items.length) setSearch({ inst: items[0].native_id }, { replace: true });
  }, [narrow, search.inst, items, setSearch]);

  // After a filter change, move off an open instance that the new filter excludes. A deep link
  // keeps its instance on first load, since it may sit on a page that has not loaded yet.
  const lastParams = useRef(params);
  useEffect(() => {
    if (instances.isPlaceholderData || !instances.data) return;
    if (lastParams.current === params) return;
    lastParams.current = params;
    if (search.inst && !items.some((i) => i.native_id === search.inst)) {
      setSearch({ inst: narrow ? undefined : items[0]?.native_id }, { replace: true });
    }
  }, [params, instances.isPlaceholderData, instances.data, items, search.inst, narrow, setSearch]);

  const step = (delta: number) => {
    if (!items.length) return;
    const next = items[Math.max(0, Math.min(items.length - 1, (idx < 0 ? -1 : idx) + delta))];
    setSearch({ inst: next.native_id }, { replace: true });
  };
  const moveTask = (delta: number) => {
    if (!tasks.length || !row) return;
    const i = tasks.indexOf(row);
    const next = tasks[(i + delta + tasks.length) % tasks.length];
    setSearch({ task: next.task_name, inst: undefined }, { replace: true });
  };
  useHotkeys(
    {
      ArrowLeft: () => step(-1),
      ArrowRight: () => step(1),
      "[": () => moveTask(-1),
      "]": () => moveTask(1),
      d: () => (ctx.baseline || trayStore.get().length) && setSearch({ side: search.side ? undefined : ctx.baseline ? "1" : trayStore.get()[0]?.key }, { replace: true }),
      f: () => ctx.baseline && setSearch({ cell: search.cell === "changed" ? undefined : "changed" }, { replace: true }),
    },
    { allowInDialog: true },
  );

  const columns: Column<InstanceRow>[] = useMemo(
    () => [
      { id: "id", header: "Instance", title: "Instance", width: 196, pin: true, cell: (r) => <span className="mono truncate-start" style={{ fontSize: 12 }} title={r.native_id}><bdi>{r.native_id}</bdi></span>, csv: (r) => r.native_id },
      { id: "this", header: "This", title: "This run", width: 52, align: "center", cell: (r) => <Correctness value={r.primary_score} correct={r.correct} format={fmt} />, csv: (r) => r.primary_score },
      ...(ctx.baseline
        ? [
            {
              id: "base",
              header: "Base",
              title: "Baseline",
              width: 52,
              align: "center" as const,
              cell: (r: InstanceRow) => (r.baseline_score == null ? <Minus size={12} color="var(--faint)" /> : <Correctness value={r.baseline_score} correct={r.baseline_correct} format={fmt} />),
              csv: (r: InstanceRow) => r.baseline_score,
            },
          ]
        : []),
      { id: "len", header: "Len", title: "Output tokens", width: 60, align: "right", cell: (r) => <span className="mono muted" style={{ fontSize: 12 }}>{r.completion_tokens ?? "—"}</span>, csv: (r) => r.completion_tokens },
      {
        id: "fin",
        header: "Fin",
        title: "Finish reason",
        width: 56,
        cell: (r) => <span className="mono" style={{ fontSize: 11.5, color: r.finish_reason === "length" ? "var(--warn-fg)" : "var(--muted)" }}>{r.finish_reason ?? "—"}</span>,
        csv: (r) => r.finish_reason,
      },
      { id: "preview", header: "Output", title: "Output preview", width: 360, minWidth: 140, grow: true, cell: (r) => <span className="truncate muted" style={{ fontSize: 12.5 }}>{r.output_preview ?? r.prompt_preview ?? ""}</span>, csv: (r) => r.output_preview },
    ],
    [ctx.baseline, fmt],
  );

  if (ctx.tasks.isLoading) return <Skeleton height={400} />;
  if (!row) return <EmptyState title="No task results" compact>This run has no successful task results to browse.</EmptyState>;

  const detail = <InstanceDetail ctx={ctx} row={row} item={items[idx]} onPrev={() => step(-1)} onNext={() => step(1)} />;
  const setRange = (key: "len" | "score", lo: string, hi: string) => setSearch({ [key]: lo || hi ? `${lo}-${hi}` : undefined }, { replace: true });

  return (
    <div className={s.split}>
      <Panel pad="none">
        <div className={s.instFilters}>
          <Select
            size="sm"
            label="Task"
            value={row.task_name}
            onChange={(v) => setSearch({ task: v, inst: undefined, correct: undefined, score: undefined })}
            options={tasks.map((t) => ({ value: t.task_name, label: t.task_name, hint: formatCount(t.n) }))}
            width={260}
          />
          <DebouncedSearchInput
            placeholder="Search prompt and output"
            value={search.iq ?? ""}
            onCommit={(v) => setSearch({ iq: v || undefined, inst: undefined }, { replace: true })}
            wrapStyle={{ flex: 1, minWidth: 160 }}
          />
        </div>
        <div className={s.instFilters} style={{ paddingTop: 8 }}>
          {kind !== "unbounded" && (
            <Select
              size="sm"
              label="Correct"
              value={search.correct ?? ""}
              onChange={(v) => setSearch({ correct: v || undefined, inst: undefined }, { replace: true })}
              options={[
                { value: "", label: "any" },
                { value: "correct", label: "correct" },
                { value: "incorrect", label: "incorrect" },
                ...(kind === "bounded" ? [{ value: "partial", label: "partial" }] : []),
              ]}
            />
          )}
          {ctx.baseline && (
            <Select
              size="sm"
              label="Vs base"
              value={search.cell ?? ""}
              onChange={(v) => setSearch({ cell: v || undefined, inst: undefined }, { replace: true })}
              options={[
                { value: "", label: "any" },
                { value: "only_a", label: "gained" },
                { value: "only_b", label: "lost" },
                { value: "both_right", label: "both right" },
                { value: "both_wrong", label: "both wrong" },
                { value: "changed", label: "changed" },
              ]}
            />
          )}
          <Select
            size="sm"
            label="Finish"
            value={search.fin ?? ""}
            onChange={(v) => setSearch({ fin: v || undefined, inst: undefined }, { replace: true })}
            options={[
              { value: "", label: "any" },
              ...Object.keys(row.finish_reason_counts).map((f) => ({ value: f, label: f, hint: formatCount(row.finish_reason_counts[f]) })),
            ]}
          />
          <Select
            size="sm"
            label="Has"
            value={search.has ?? ""}
            onChange={(v) => setSearch({ has: v || undefined, inst: undefined }, { replace: true })}
            options={[
              { value: "", label: "anything" },
              { value: "scoring_error", label: "scoring error" },
              { value: "execution_result", label: "execution result" },
              { value: "judge_result", label: "judge result" },
              { value: "trajectory", label: "trajectory" },
            ]}
          />
          <Select
            size="sm"
            label="Sort"
            value={search.isort ?? "doc_id"}
            onChange={(v) => setSearch({ isort: v === "doc_id" ? undefined : v }, { replace: true })}
            options={[
              { value: "doc_id", label: "id" },
              { value: "-score", label: "score ↓" },
              { value: "score", label: "score ↑" },
              ...(ctx.baseline ? [{ value: "-delta", label: "delta ↓" }, { value: "delta", label: "delta ↑" }] : []),
              { value: "-length", label: "length ↓" },
              { value: "length", label: "length ↑" },
            ]}
          />
          <span className="row" style={{ gap: 4 }}>
            <span className="t-caption">len</span>
            <Input
              key={search.len ?? ""}
              style={{ width: 92, height: 26, fontSize: 12 }}
              placeholder="min-max"
              defaultValue={search.len ?? ""}
              onBlur={(e) => setSearch({ len: e.currentTarget.value || undefined }, { replace: true })}
              onKeyDown={(e) => e.key === "Enter" && setSearch({ len: e.currentTarget.value || undefined }, { replace: true })}
              aria-label="Output length range"
            />
          </span>
          {kind === "bounded" && (
            <span className="row" style={{ gap: 6, width: 170 }}>
              <span className="t-caption nowrap">threshold {Number(search.thr ?? 0.5).toFixed(2)}</span>
              <Slider value={Number(search.thr ?? 0.5)} min={0.05} max={0.95} step={0.05} onChange={(v) => setSearch({ thr: v === 0.5 ? undefined : v.toFixed(2) }, { replace: true })} ariaLabel="Correctness threshold" />
            </span>
          )}
          {search.score && (
            <span className={ui.chip}>
              <span>score {search.score}</span>
              <button type="button" className={ui.chipRemove} onClick={() => setRange("score", "", "")} aria-label="Clear score range">
                <X />
              </button>
            </span>
          )}
        </div>
        {instances.error ? (
          <div style={{ padding: 12 }}>
            <ErrorPanel error={instances.error} onRetry={() => instances.refetch()} />
          </div>
        ) : (
          <DataTable
            tableId="instances"
            columns={columns}
            rows={items}
            getRowId={(r) => r.native_id}
            loading={instances.isLoading}
            fetching={instances.isFetching}
            total={total}
            hasMore={instances.hasNextPage}
            onLoadMore={() => void instances.fetchNextPage()}
            onRowOpen={(r) => setSearch({ inst: r.native_id }, { replace: true })}
            onFocusChange={(r) => r && setSearch({ inst: r.native_id }, { replace: true })}
            focusedId={search.inst ?? null}
            maxHeight="calc(100vh - 380px)"
            exportName={`${ctx.run.run_id}-${row.task_name}-instances`}
            rowClassName={(r) => (r.native_id === search.inst ? s.flash : undefined)}
            empty={<EmptyState title="No instances match" compact>Loosen the filters above.</EmptyState>}
            toolbar={
              <span className="t-caption">
                {total != null ? `${formatCount(total)} instances` : "…"} · score <span className="mono">{formatScore(row.score, fmt)}</span>
              </span>
            }
          />
        )}
      </Panel>
      {narrow ? (
        <Dialog.Root open={!!search.inst} onOpenChange={(o) => !o && setSearch({ inst: undefined }, { replace: true })}>
          <Dialog.Portal>
            <Dialog.Content className={ui.sheet} aria-describedby={undefined}>
              <div className={s.drawerHead}>
                <Dialog.Title style={{ fontSize: 14, margin: 0 }} className="mono truncate">
                  {search.inst}
                </Dialog.Title>
                <span className="spacer" />
                <Dialog.Close asChild>
                  <IconButton label="Close" icon={<X />} />
                </Dialog.Close>
              </div>
              <div className={s.drawerBody}>{detail}</div>
            </Dialog.Content>
          </Dialog.Portal>
        </Dialog.Root>
      ) : (
        detail
      )}
    </div>
  );
}
