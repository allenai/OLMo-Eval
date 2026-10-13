import type { TaskResultRow } from "@contract/api-types";
import * as Dialog from "@radix-ui/react-dialog";
import { ChevronLeft, ChevronRight, Maximize2, X } from "lucide-react";
import { useMemo } from "react";
import { AppLink, buildHref } from "@/components/AppLink";
import { DeltaValue, MetricName, ScoreValue, SigLegend } from "@/components/cells";
import { type Column, DataTable, parseSortList, sortRows } from "@/components/DataTable";
import { Badge, Checkbox, EmptyState, ErrorPanel, IconButton, Kbd, Panel, SearchInput, Segmented, Select, Tip, uiStyles as ui } from "@/components/primitives";
import { formatCount, formatDuration, formatP, formatRate, formatRuntime, formatScore, metricLabel, shortHash } from "@/lib/format";
import { BASIS_HELP, BASIS_LABEL, basisMark, parseRuntimeMode, RUNTIME_MODE_HELP, RUNTIME_MODE_OPTIONS, runtimeModeParam, runtimeValue, shares } from "@/lib/runtime";
import { useHotkeys } from "@/lib/keyboard";
import { parseList } from "@/lib/url";
import s from "./run.module.css";
import { RuntimeTimeline } from "./RuntimeTimeline";
import { TaskDrilldown } from "./TaskDrilldown";
import { type RunCtx, rowMeta, scoreFormat } from "./types";

export function TasksTab(ctx: RunCtx) {
  const { search, setSearch } = ctx;
  const items = useMemo(() => ctx.tasks.data?.items ?? [], [ctx.tasks.data]);
  const flags = parseList(search.tf);
  const suiteFilter = search.sel;
  const allMetrics = search.all === "1";
  const query = (search.tq ?? "").toLowerCase();
  const mode = parseRuntimeMode(search.rt);
  // Share of the run's task compute: each task's own inference time over the sum for the run.
  const shareById = useMemo(() => {
    const values = shares(items.map((t) => t.runtime.inference_seconds));
    return new Map(items.map((t, i) => [t.task_result_id, values[i]]));
  }, [items]);
  const maxShare = useMemo(() => Math.max(0, ...[...shareById.values()].map((v) => v ?? 0)), [shareById]);

  const rows = useMemo(
    () =>
      items.filter((t) => {
        if (query && !query.split(/[\s:]+/).filter(Boolean).every((tok) => t.task_name.toLowerCase().includes(tok))) return false;
        if (suiteFilter && !t.suites.includes(suiteFilter)) return false;
        if (flags.includes("changed") && !t.baseline?.delta.significant) return false;
        if (flags.includes("failed") && !t.error && !t.instances_failed) return false;
        if (flags.includes("config") && !t.baseline?.delta.hash_mismatch) return false;
        return true;
      }),
    [items, query, suiteFilter, flags],
  );
  const suites = Array.from(new Set(items.flatMap((t) => t.suites))).sort();
  const metricKeys = useMemo(() => Array.from(new Set(items.flatMap((t) => Object.keys(t.metrics)))).sort(), [items]);

  const toggleFlag = (flag: string, on: boolean) => {
    const next = on ? [...flags, flag] : flags.filter((f) => f !== flag);
    setSearch({ tf: next.join(",") || undefined }, { replace: true });
  };

  const columns: Column<TaskResultRow>[] = useMemo(() => {
    const cols: Column<TaskResultRow>[] = [
      {
        id: "task",
        header: "Task",
        title: "Task",
        width: 250,
        pin: true,
        sortValue: (r) => r.task_name,
        cell: (r) => (
          <span className="row" style={{ gap: 6, minWidth: 0 }}>
            <span className="truncate">{r.task_name}</span>
            {r.error && <Badge tone="danger">failed</Badge>}
          </span>
        ),
        csv: (r) => r.task_name,
      },
      {
        id: "hash",
        header: "Variant",
        title: "Variant hash",
        width: 92,
        cell: (r) =>
          r.baseline?.delta.hash_mismatch ? (
            <Tip content={`Differs from baseline (${shortHash(r.baseline.task_hash)})`}>
              <span>
                <Badge tone="warn">{shortHash(r.task_hash)}</Badge>
              </span>
            </Tip>
          ) : (
            <span className="mono muted" style={{ fontSize: 11.5 }}>
              {shortHash(r.task_hash)}
            </span>
          ),
        csv: (r) => r.task_hash,
      },
      {
        id: "metric",
        header: "Primary metric",
        title: "Primary metric",
        width: 170,
        cell: (r) => <MetricName metric={r.primary_metric} />,
        csv: (r) => r.primary_metric,
      },
      {
        id: "score",
        header: "Score",
        title: "Score",
        width: 110,
        align: "right",
        sortValue: (r) => r.score,
        cell: (r) => <ScoreValue score={r.score} stderr={r.stderr} format={scoreFormat(r)} />,
        csv: (r) => r.score,
      },
      {
        id: "n",
        header: "n",
        title: "Instances (saved / processed / failed)",
        headerTitle: "Instances saved / processed / failed",
        width: 118,
        align: "right",
        sortValue: (r) => r.n,
        cell: (r) => (
          <span className="mono" style={{ fontSize: 12 }}>
            {formatCount(r.instances_stored)}
            <span className="muted">/{formatCount(r.instances_processed)}</span>
            {r.instances_failed ? <span style={{ color: "var(--danger-fg)" }}>/{formatCount(r.instances_failed)}</span> : null}
          </span>
        ),
        csv: (r) => r.n,
      },
      {
        id: "base",
        header: "Base",
        title: "Baseline score",
        width: 80,
        align: "right",
        sortValue: (r) => r.baseline?.score,
        cell: (r) => (r.baseline ? <ScoreValue score={r.baseline.score} format={scoreFormat(r)} showStderr={false} muted /> : <span className="faint">—</span>),
        csv: (r) => r.baseline?.score,
      },
      {
        id: "delta",
        header: "Delta",
        title: "Delta vs baseline",
        width: 92,
        align: "right",
        sortValue: (r) => r.baseline?.delta.delta,
        cell: (r) => (r.baseline ? <DeltaValue stats={r.baseline.delta} format={scoreFormat(r)} higherIsBetter={rowMeta(r).higherIsBetter} /> : <span className="faint">—</span>),
        csv: (r) => r.baseline?.delta.delta,
      },
      {
        id: "ci",
        header: "95% CI",
        title: "95% CI",
        width: 128,
        align: "right",
        cell: (r) =>
          r.baseline?.delta.ci_low != null ? (
            <span className="mono muted" style={{ fontSize: 11 }}>
              [{formatScore(r.baseline.delta.ci_low, scoreFormat(r))}, {formatScore(r.baseline.delta.ci_high, scoreFormat(r))}]
            </span>
          ) : (
            <span className="faint">—</span>
          ),
        csv: (r) => (r.baseline?.delta.ci_low != null ? `${r.baseline.delta.ci_low}..${r.baseline.delta.ci_high}` : null),
      },
      {
        id: "p",
        header: "p",
        title: "p-value",
        width: 76,
        align: "right",
        sortValue: (r) => r.baseline?.delta.p_value,
        cell: (r) => <span className="mono muted" style={{ fontSize: 11.5 }}>{r.baseline ? formatP(r.baseline.delta.p_value) : "—"}</span>,
        csv: (r) => r.baseline?.delta.p_value,
      },
      {
        id: "runtime",
        header: (
          <Tip content={RUNTIME_MODE_HELP[mode]}>
            <span>{mode === "inference" ? "Runtime" : "With startup"}</span>
          </Tip>
        ),
        title: mode === "inference" ? "Runtime (inference)" : "Runtime with startup",
        width: 100,
        align: "right",
        sortValue: (r) => runtimeValue(r.runtime, mode),
        cell: (r) => {
          const v = runtimeValue(r.runtime, mode);
          return <span className="mono">{v == null ? <span className="faint">—</span> : `${basisMark(r.runtime.basis)}${formatRuntime(v)}`}</span>;
        },
        csv: (r) => runtimeValue(r.runtime, mode),
      },
      {
        id: "share",
        header: (
          <Tip content="This task's share of the run's task compute (its own inference time over the total for all tasks). The bar is relative to the largest task.">
            <span>Share</span>
          </Tip>
        ),
        title: "Share of compute",
        width: 104,
        align: "right",
        sortValue: (r) => shareById.get(r.task_result_id),
        cell: (r) => {
          const sh = shareById.get(r.task_result_id);
          if (sh == null) return <span className="faint">—</span>;
          return (
            <span className="row" style={{ gap: 6, justifyContent: "flex-end", width: "100%" }}>
              <span style={{ width: 36, height: 4, borderRadius: 2, background: "var(--grid)", position: "relative", flex: "none" }} aria-hidden>
                <span style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${maxShare > 0 ? Math.max(4, (sh / maxShare) * 100) : 0}%`, borderRadius: 2, background: "var(--teal-line)" }} />
              </span>
              <span className="mono muted" style={{ minWidth: 30, textAlign: "right" }}>{formatRate(sh)}</span>
            </span>
          );
        },
        csv: (r) => shareById.get(r.task_result_id),
      },
      {
        id: "basis",
        header: "Basis",
        title: "Runtime basis",
        width: 96,
        sortValue: (r) => r.runtime.basis,
        cell: (r) => (
          <Tip content={<span style={{ display: "block", maxWidth: 260 }}>{BASIS_HELP[r.runtime.basis]}</span>}>
            <span className={r.runtime.basis === "not_recorded" ? "faint" : "muted"} style={{ fontSize: 12 }}>
              {BASIS_LABEL[r.runtime.basis]}
            </span>
          </Tip>
        ),
        csv: (r) => r.runtime.basis,
      },
      {
        id: "duration",
        header: (
          <Tip content="When the task finished, counted from the start of processing. Tasks share the inference workers, so this is not the task's own runtime.">
            <span>Done after</span>
          </Tip>
        ),
        title: "Done after (from processing start)",
        width: 92,
        align: "right",
        defaultHidden: true,
        sortValue: (r) => r.duration_seconds,
        cell: (r) => <span className="mono muted">{formatDuration(r.duration_seconds)}</span>,
        csv: (r) => r.duration_seconds,
      },
      {
        id: "tokens",
        header: "Compl. tokens",
        title: "Completion tokens",
        width: 108,
        align: "right",
        sortValue: (r) => r.completion_tokens_total,
        cell: (r) => <span className="mono muted">{r.completion_tokens_total != null ? formatCount(r.completion_tokens_total) : "—"}</span>,
        csv: (r) => r.completion_tokens_total,
      },
      {
        id: "meanlen",
        header: "Mean len",
        title: "Mean output length",
        width: 84,
        align: "right",
        sortValue: (r) => r.mean_completion_tokens,
        cell: (r) => <span className="mono muted">{r.mean_completion_tokens != null ? Math.round(r.mean_completion_tokens) : "—"}</span>,
        csv: (r) => r.mean_completion_tokens,
      },
      {
        id: "trunc",
        header: "Trunc.",
        title: "Truncation rate",
        width: 70,
        align: "right",
        sortValue: (r) => r.truncation_rate,
        cell: (r) => (
          <span className="mono" style={{ color: (r.truncation_rate ?? 0) >= 0.05 ? "var(--warn-fg)" : "var(--muted)" }}>
            {r.truncation_rate != null ? formatRate(r.truncation_rate) : "—"}
          </span>
        ),
        csv: (r) => r.truncation_rate,
      },
    ];
    if (allMetrics) {
      for (const key of metricKeys) {
        cols.push({
          id: `m:${key}`,
          header: <span style={{ textTransform: "none", letterSpacing: 0 }}>{metricLabel(key)}</span>,
          title: key,
          width: 120,
          align: "right",
          sortValue: (r) => r.metrics[key],
          cell: (r) => {
            const v = r.metrics[key];
            const f = r.metric_meta[key]?.display_format ?? "percent";
            return v == null ? <span className="faint">—</span> : <span className="mono">{formatScore(v, key === r.primary_metric ? scoreFormat(r) : f)}</span>;
          },
          csv: (r) => r.metrics[key],
        });
      }
    }
    return cols;
  }, [allMetrics, metricKeys, mode, shareById, maxShare]);

  // [ and ] step through the rows in the order the table shows them.
  const sortedRows = useMemo(() => sortRows(rows, parseSortList(search.ts), columns), [rows, search.ts, columns]);
  const openIdx = search.task ? sortedRows.findIndex((r) => r.task_name === search.task) : -1;
  const openRow = openIdx >= 0 ? sortedRows[openIdx] : items.find((r) => r.task_name === search.task);
  const move = (delta: number) => {
    if (!sortedRows.length) return;
    const idx = openIdx < 0 ? 0 : (openIdx + delta + sortedRows.length) % sortedRows.length;
    setSearch({ task: sortedRows[idx].task_name, cell: undefined }, { replace: true });
  };
  useHotkeys({ "[": () => move(-1), "]": () => move(1) }, { enabled: !!search.task, allowInDialog: true });
  const baselineRunId = ctx.baselineInfo?.run?.run_id ?? ctx.baselineInfo?.run_ids[0];

  if (ctx.tasks.error) return <ErrorPanel error={ctx.tasks.error} onRetry={() => ctx.tasks.refetch()} />;

  const recorded = items.some((t) => t.runtime.span_start_s != null);
  return (
    <>
      {(recorded || ctx.run.startup_seconds != null) && (
        <Panel
          title="Timeline"
          caption="Startup, then each task's span while it shared the inference workers. A span is longer than the task's own runtime; hover a bar for both."
          refetching={ctx.tasks.isFetching && !ctx.tasks.isLoading}
          style={{ marginBottom: 14 }}
        >
          <RuntimeTimeline run={ctx.run} rows={rows} selected={search.task} onOpen={(task) => setSearch({ task })} />
        </Panel>
      )}
      <Panel pad="none" refetching={ctx.tasks.isFetching && !ctx.tasks.isLoading}>
        <DataTable
          tableId="run-tasks"
          columns={columns}
          rows={rows}
          getRowId={(r) => String(r.task_result_id)}
          loading={ctx.tasks.isLoading}
          sort={search.ts}
          onSortChange={(v) => setSearch({ ts: v }, { replace: true })}
          clientSort
          onRowOpen={(r) => setSearch({ task: r.task_name })}
          focusedId={openRow ? String(openRow.task_result_id) : null}
          maxHeight="calc(100vh - 330px)"
          exportName={`${ctx.run.run_id}-tasks`}
          toolbar={
            <>
              <SearchInput placeholder="Find task" value={search.tq ?? ""} onChange={(e) => setSearch({ tq: e.target.value || undefined }, { replace: true })} wrapStyle={{ width: 200 }} />
              <Select
                size="sm"
                label="Suite"
                value={suiteFilter ?? ""}
                onChange={(v) => setSearch({ sel: v || undefined }, { replace: true })}
                options={[{ value: "", label: "Any" }, ...suites.map((su) => ({ value: su, label: su }))]}
              />
              {ctx.baseline && <Checkbox checked={flags.includes("changed")} onChange={(on) => toggleFlag("changed", on)} label="Only changed" />}
              <Checkbox checked={flags.includes("failed")} onChange={(on) => toggleFlag("failed", on)} label="Only failed" />
              {ctx.baseline && <Checkbox checked={flags.includes("config")} onChange={(on) => toggleFlag("config", on)} label="Config differs" />}
              <Checkbox checked={allMetrics} onChange={(on) => setSearch({ all: on ? "1" : undefined }, { replace: true })} label="All metrics" />
              <Segmented value={mode} onChange={(v) => setSearch({ rt: runtimeModeParam(v) }, { replace: true })} label="Runtime view" options={RUNTIME_MODE_OPTIONS} />
              {ctx.baseline && <SigLegend />}
            </>
          }
          empty={<EmptyState title="No tasks match" compact>Clear the filters above.</EmptyState>}
        />
      </Panel>
      <Dialog.Root open={!!openRow} onOpenChange={(o) => !o && setSearch({ task: undefined, cell: undefined })}>
        <Dialog.Portal>
          <Dialog.Overlay className={ui.overlay} />
          <Dialog.Content className={ui.sheet} aria-describedby={undefined}>
            {openRow && (
              <>
                <div className={s.drawerHead}>
                  <Dialog.Title style={{ fontSize: 16, fontWeight: 800, margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {openRow.task_name}
                  </Dialog.Title>
                  <span className="t-caption mono hide-sm">
                    {openIdx + 1}/{rows.length}
                  </span>
                  <span className="spacer" />
                  <span className="t-caption hide-sm">
                    <Kbd>[</Kbd> <Kbd>]</Kbd> prev / next
                  </span>
                  <IconButton label="Previous task ([)" icon={<ChevronLeft />} onClick={() => move(-1)} />
                  <IconButton label="Next task (])" icon={<ChevronRight />} onClick={() => move(1)} />
                  <AppLink href={buildHref(`/runs/${ctx.run.run_id}/tasks/${encodeURIComponent(openRow.task_name)}`, {})} title="Open as a full page">
                    <IconButton label="Open as page" icon={<Maximize2 />} />
                  </AppLink>
                  <Dialog.Close asChild>
                    <IconButton label="Close (Esc)" icon={<X />} />
                  </Dialog.Close>
                </div>
                <div className={s.drawerBody}>
                  <TaskDrilldown
                    run={ctx.run}
                    row={openRow}
                    baseline={ctx.baseline}
                    baselineRunId={baselineRunId}
                    onOpenInstances={(patch) =>
                      setSearch({ tab: "instances", task: openRow.task_name, cell: patch.cell, fin: patch.fin, score: patch.score, inst: undefined })
                    }
                  />
                </div>
              </>
            )}
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </>
  );
}
