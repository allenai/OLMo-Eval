import type { Distribution, RunDetail, SuiteResultRow, TaskResultRow } from "@contract/api-types";
import { AlertTriangle, CheckCircle2, ChevronDown, FileWarning, Scissors, XCircle } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { useDistributions } from "@/api/hooks/compare";
import { useRunConfigs, useRunsList } from "@/api/hooks/runs";
import { DivergingBars, type DivergingItem, PercentileStrip } from "@/charts/basic";
import { ChartPanel } from "@/charts/core";
import { AppLink, buildHref } from "@/components/AppLink";
import { DeltaValue, ScoreValue, SigLegend } from "@/components/cells";
import { Badge, Button, Checkbox, CopyText, cx, EmptyState, ErrorPanel, KV, LinkOut, Panel, Skeleton, Tip } from "@/components/primitives";
import { type Cell, downloadText, toCSV } from "@/lib/csv";
import { jsonDiff } from "@/lib/diff";
import { formatCI, formatCount, formatDuration, formatRate, formatRuntime, formatScore, formatStderr, shortHash } from "@/lib/format";
import { useHotkeys } from "@/lib/keyboard";
import { goodness } from "@/lib/scales";
import { absoluteLocal } from "@/lib/time";
import s from "./run.module.css";
import { type RunCtx, rowMeta, scoreFormat } from "./types";
import { pressable } from "@/lib/pressable";

// ---------------------------------------------------------------- headline tiles

interface TileSpec {
  key: string;
  name: string;
  score: number | null;
  stderr: number | null;
  format: "percent" | "raw";
  higherIsBetter: boolean | null;
  delta: SuiteResultRow["baseline"] extends infer B ? (B extends { delta: infer D } ? D | null : null) : null;
  colKey: string;
  scrollTo: string;
}

function Tiles({ ctx, tiles }: { ctx: RunCtx; tiles: TileSpec[] }) {
  const group = ctx.run.experiment_group;
  const peers = useRunsList(
    { group: group ?? undefined, cols: tiles.map((t) => t.colKey).join(","), limit: 500 },
    { enabled: !!group && tiles.length > 0 },
  );
  return (
    <div className={s.tiles}>
      {tiles.map((t) => {
        const values = (peers.data?.items ?? [])
          .map((r) => r.scores[t.colKey]?.score)
          .filter((v): v is number => v != null);
        const better = values.filter((v) => (t.higherIsBetter === false ? v < (t.score ?? 0) : v > (t.score ?? 0))).length;
        return (
          <button
            key={t.key}
            type="button"
            className={s.tile}
            onClick={() => document.getElementById(t.scrollTo)?.scrollIntoView({ behavior: "smooth", block: "center" })}
            title={`Jump to ${t.name} in the table`}
          >
            <span className={s.tileName}>{t.name}</span>
            <span className={s.tileValue}>
              {formatScore(t.score, t.format)}
              <span className={s.tileStderr}>{formatStderr(t.stderr, t.format)}</span>
            </span>
            <span className={s.tileRow}>
              {t.delta ? (
                <>
                  <DeltaValue stats={t.delta} format={t.format} higherIsBetter={t.higherIsBetter} />
                  <span>vs base</span>
                </>
              ) : (
                <span className="faint">no baseline</span>
              )}
            </span>
            <span className={s.tileRow}>
              {group && values.length > 1 && t.score != null ? (
                <>
                  <span>
                    rank <span className="mono" style={{ color: "var(--fg)" }}>{better + 1}</span> / {values.length} in group
                  </span>
                  <PercentileStrip scores={values} value={t.score} higherIsBetter={t.higherIsBetter} format={t.format} width={72} />
                </>
              ) : (
                <span className="faint">{peers.isLoading && group ? "…" : "no group peers"}</span>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- suite tree

type TreeNode =
  | { kind: "suite"; id: string; name: string; depth: number; row: SuiteResultRow; children: TreeNode[] }
  | { kind: "task"; id: string; name: string; depth: number; row: TaskResultRow | null }
  | { kind: "group"; id: string; name: string; depth: number; children: TreeNode[] };

function buildTree(suites: SuiteResultRow[], tasks: TaskResultRow[], unsuited: string[]): TreeNode[] {
  const byName = new Map(suites.map((r) => [r.name, r]));
  const taskByName = new Map(tasks.map((t) => [t.task_name, t]));
  const build = (row: SuiteResultRow, depth: number): TreeNode => ({
    kind: "suite",
    id: `suite:${row.name}`,
    name: row.name,
    depth,
    row,
    children: row.children.flatMap((c): TreeNode[] => {
      if (c.type === "suite") {
        const child = byName.get(c.name);
        return child ? [build(child, depth + 1)] : [];
      }
      return [{ kind: "task", id: `task:${c.name}`, name: c.name, depth: depth + 1, row: taskByName.get(c.name) ?? null }];
    }),
  });
  const roots = suites.filter((r) => !r.parent || !byName.has(r.parent)).map((r) => build(r, 0));
  if (unsuited.length) {
    roots.push({
      kind: "group",
      id: "group:unsuited",
      name: `Tasks not in any suite (${unsuited.length})`,
      depth: 0,
      children: unsuited.map((n) => ({ kind: "task" as const, id: `task:${n}`, name: n, depth: 1, row: taskByName.get(n) ?? null })),
    });
  }
  return roots;
}

function isChanged(node: TreeNode): boolean {
  if (node.kind === "task") return !!node.row?.baseline?.delta.significant;
  if (node.kind === "suite" && node.row.baseline?.delta.significant) return true;
  return node.children.some(isChanged);
}

function TreeRows({
  nodes,
  collapsed,
  toggle,
  onlyChanged,
  dist,
  baseline,
  onOpenTask,
}: {
  nodes: TreeNode[];
  collapsed: Set<string>;
  toggle: (id: string) => void;
  onlyChanged: boolean;
  dist: Map<string, Distribution>;
  baseline: boolean;
  onOpenTask: (row: TaskResultRow) => void;
}): ReactNode {
  return nodes.map((node) => {
    if (onlyChanged && !isChanged(node)) return null;
    const pad = { paddingLeft: node.depth * 16 };
    if (node.kind === "group" || node.kind === "suite") {
      const closed = collapsed.has(node.id);
      const row = node.kind === "suite" ? node.row : null;
      const d = row?.baseline?.delta;
      return (
        <div key={node.id}>
          <div id={`ov-${node.id}`} className={cx(s.treeRow, s.treeSuite)} {...pressable(() => toggle(node.id))} aria-expanded={!closed}>
            <span className={s.treeName} style={pad}>
              <span className={cx(s.treeToggle, closed && s.treeClosed)}>
                <ChevronDown />
              </span>
              <Tip content={row ? `${row.aggregation.replace(/_/g, " ")} of ${row.children.length} ${row.children[0]?.type === "suite" ? "child suites" : "tasks"}${row.description ? `. ${row.description}` : ""}` : null}>
                <span>{node.name}</span>
              </Tip>
              {row && row.tasks_missing.length > 0 && (
                <Tip content={`Missing: ${row.tasks_missing.join(", ")}`}>
                  <span>
                    <Badge tone="warn">{row.tasks_missing.length} missing</Badge>
                  </span>
                </Tip>
              )}
            </span>
            <span className={s.right}>{row ? <ScoreValue score={row.score} format={row.display_format} showStderr={false} strong /> : null}</span>
            <span className={cx(s.right, "mono muted")} style={{ fontSize: 11.5 }}>
              {row ? formatStderr(row.stderr, row.display_format) : ""}
            </span>
            <span className={cx(s.right, "mono muted")} style={{ fontSize: 12 }}>
              {row ? formatCount(row.n_instances) : ""}
            </span>
            <span className={s.right}>{row?.baseline ? <ScoreValue score={row.baseline.score} format={row.display_format} showStderr={false} muted /> : null}</span>
            <span className={s.right}>{d ? <DeltaValue stats={d} format={row!.display_format} higherIsBetter={row!.higher_is_better} /> : null}</span>
            <span className={cx(s.right, "mono muted")} style={{ fontSize: 11 }}>
              {d ? formatCI(d.ci_low, d.ci_high, row!.display_format) : ""}
            </span>
            <span />
          </div>
          {!closed && (
            <TreeRows nodes={node.children} collapsed={collapsed} toggle={toggle} onlyChanged={onlyChanged} dist={dist} baseline={baseline} onOpenTask={onOpenTask} />
          )}
        </div>
      );
    }
    const row = node.row;
    if (!row) {
      return (
        <div key={node.id} className={s.treeRow} style={{ cursor: "default" }}>
          <span className={s.treeName} style={pad}>
            <span style={{ width: 16 }} />
            <span className="muted">{node.name}</span>
            <Badge tone="muted">not run</Badge>
          </span>
        </div>
      );
    }
    const m = rowMeta(row);
    const fmt = scoreFormat(row);
    const b = row.baseline;
    const d = dist.get(row.task_name);
    return (
      <div key={node.id} id={`ov-${node.id}`} className={s.treeRow} {...pressable(() => onOpenTask(row))}>
        <span className={s.treeName} style={pad}>
          <span style={{ width: 16, flex: "none" }} />
          <span>{row.task_name}</span>
          {row.error && <Badge tone="danger">failed</Badge>}
          {b?.delta.hash_mismatch && (
            <Tip content={`Task config differs from the baseline (hash ${shortHash(row.task_hash)} vs ${shortHash(b.task_hash)})`}>
              <span>
                <Badge tone="warn">config differs</Badge>
              </span>
            </Tip>
          )}
        </span>
        <span className={s.right}>{row.error ? <span className="faint">—</span> : <ScoreValue score={row.score} format={fmt} showStderr={false} />}</span>
        <span className={cx(s.right, "mono muted")} style={{ fontSize: 11.5 }}>
          {formatStderr(row.stderr, fmt)}
        </span>
        <span className={cx(s.right, "mono muted")} style={{ fontSize: 12 }}>
          {formatCount(row.n)}
        </span>
        <span className={s.right}>{b ? <ScoreValue score={b.score} format={fmt} showStderr={false} muted /> : null}</span>
        <span className={s.right}>{b ? <DeltaValue stats={b.delta} format={fmt} higherIsBetter={m.higherIsBetter} /> : null}</span>
        <span className={cx(s.right, "mono muted")} style={{ fontSize: 11 }}>
          {b ? formatCI(b.delta.ci_low, b.delta.ci_high, fmt) : ""}
        </span>
        <span className={s.right}>
          {d && row.score != null ? (
            <PercentileStrip
              scores={d.scores}
              value={row.score}
              baseline={b?.score}
              higherIsBetter={m.higherIsBetter}
              format={fmt}
              topLabel={d.top ? `${d.top.model.series_label}${d.top.model.step != null ? ` @ ${d.top.model.step}` : ""} (${formatScore(d.top.score, fmt)})` : undefined}
            />
          ) : (
            <span className="faint mono">—</span>
          )}
        </span>
      </div>
    );
  });
}

function SuiteTree({ ctx, onOpenTask }: { ctx: RunCtx; onOpenTask: (row: TaskResultRow) => void }) {
  const tasks = useMemo(() => ctx.tasks.data?.items ?? [], [ctx.tasks.data]);
  const suites = useMemo(() => ctx.suites.data?.items ?? [], [ctx.suites.data]);
  const unsuited = useMemo(() => ctx.suites.data?.unsuited_tasks ?? [], [ctx.suites.data]);
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set(suites.filter((r) => r.depth >= 1).map((r) => `suite:${r.name}`)));
  const [onlyChanged, setOnlyChanged] = useState(false);
  const tree = useMemo(() => buildTree(suites, tasks, unsuited), [suites, tasks, unsuited]);
  const distReq = useMemo(
    () => (tasks.length ? { items: tasks.filter((t) => !t.error).map((t) => ({ task_name: t.task_name, task_hash: t.task_hash, metric: "primary" })) } : null),
    [tasks],
  );
  const dist = useDistributions(distReq);
  const distMap = useMemo(() => new Map((dist.data?.items ?? []).map((d) => [d.task_name, d])), [dist.data]);
  const allIds = useMemo(() => {
    const ids: string[] = [];
    const walk = (nodes: TreeNode[]) => nodes.forEach((n) => n.kind !== "task" && (ids.push(n.id), walk(n.children)));
    walk(tree);
    return ids;
  }, [tree]);
  const toggle = (id: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  useHotkeys({
    e: () => setCollapsed((prev) => (prev.size ? new Set() : new Set(allIds))),
    f: () => ctx.baseline && setOnlyChanged((v) => !v),
  });
  const exportCsv = () => {
    const rows: Cell[][] = [];
    for (const r of suites) rows.push(["suite", r.name, r.score, r.stderr, r.n_instances, r.baseline?.score ?? null, r.baseline?.delta.delta ?? null, r.baseline?.delta.ci_low ?? null, r.baseline?.delta.ci_high ?? null]);
    for (const t of tasks) rows.push(["task", t.task_name, t.score, t.stderr, t.n, t.baseline?.score ?? null, t.baseline?.delta.delta ?? null, t.baseline?.delta.ci_low ?? null, t.baseline?.delta.ci_high ?? null]);
    downloadText(`${ctx.run.run_id}-scores.csv`, toCSV(["kind", "name", "score", "stderr", "n", "baseline", "delta", "ci_low", "ci_high"], rows));
  };
  const loading = ctx.tasks.isLoading || ctx.suites.isLoading;
  return (
    <Panel
      title="Suites and tasks"
      caption={
        <>
          Raw scores with stderr. {ctx.baseline ? "Deltas use a paired bootstrap over shared instances. " : ""}“Where” places the score among the latest run of every model on that task variant.
        </>
      }
      pad="none"
      refetching={(ctx.tasks.isFetching || ctx.suites.isFetching) && !loading}
      actions={
        <>
          {ctx.baseline && <SigLegend />}
          <Button size="sm" variant="ghost" onClick={() => setCollapsed((prev) => (prev.size ? new Set() : new Set(allIds)))}>
            {collapsed.size ? "Expand all" : "Collapse all"}
          </Button>
          {ctx.baseline && <Checkbox checked={onlyChanged} onChange={setOnlyChanged} label="Only changed" />}
          <Button size="sm" variant="ghost" onClick={exportCsv}>
            CSV
          </Button>
        </>
      }
    >
      {loading ? (
        <div style={{ padding: 12, display: "grid", gap: 10 }}>
          {Array.from({ length: 10 }, (_, i) => (
            <Skeleton key={i} height={14} />
          ))}
        </div>
      ) : ctx.tasks.error ? (
        <div style={{ padding: 12 }}>
          <ErrorPanel error={ctx.tasks.error} onRetry={() => ctx.tasks.refetch()} />
        </div>
      ) : !tasks.length ? (
        <EmptyState title="No task results yet" compact>
          {ctx.run.status === "running" ? "This run is still running. Task results appear as they finish." : "This run recorded no task results."}
        </EmptyState>
      ) : (
        <div className={s.tree}>
          <div className={s.treeGrid}>
            <div className={s.treeHead}>
              <span>Name</span>
              <span className={s.right}>Score</span>
              <span className={s.right}>±</span>
              <span className={s.right}>n</span>
              <span className={s.right}>Base</span>
              <span className={s.right}>Delta</span>
              <span className={s.right}>95% CI</span>
              <span className={s.right}>Where</span>
            </div>
            <TreeRows nodes={tree} collapsed={collapsed} toggle={toggle} onlyChanged={onlyChanged} dist={distMap} baseline={!!ctx.baseline} onOpenTask={onOpenTask} />
          </div>
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- health

function HealthPanel({ ctx }: { ctx: RunCtx }) {
  const tasks = ctx.tasks.data?.items ?? [];
  const baseRunId = ctx.baselineInfo?.run?.run_id ?? ctx.baselineInfo?.run_ids[0];
  const mismatched = tasks.filter((t) => t.baseline?.delta.hash_mismatch);
  const mine = useRunConfigs(mismatched.length ? ctx.run.run_id : null);
  const theirs = useRunConfigs(mismatched.length ? baseRunId : null);
  const items: { icon: ReactNode; tone: string; text: ReactNode; detail?: ReactNode; href?: string }[] = [];
  for (const e of ctx.run.errors) {
    items.push({
      icon: <XCircle />,
      tone: s.bad,
      text: e.task ? <><strong>{e.task}</strong> failed</> : <strong>The run failed</strong>,
      detail: e.error,
      href: e.task ? buildHref(`/runs/${ctx.run.run_id}`, { tab: "tasks", task: e.task }) : undefined,
    });
  }
  for (const t of tasks) {
    if (t.instances_failed) {
      items.push({
        icon: <AlertTriangle />,
        tone: s.warn,
        text: (
          <>
            <strong>{t.task_name}</strong>: {formatCount(t.instances_failed)}/{formatCount(t.instances_processed ?? t.n)} instances failed (
            {formatRate((t.instances_failed ?? 0) / Math.max(1, t.instances_processed ?? t.n), 1)})
          </>
        ),
        detail: t.error_summary ? Object.entries(t.error_summary).map(([k, v]) => `${k} ×${v}`).join(", ") : undefined,
        href: buildHref(`/runs/${ctx.run.run_id}`, { tab: "instances", task: t.task_name, has: "scoring_error" }),
      });
    }
  }
  for (const t of tasks.filter((x) => (x.truncation_rate ?? 0) >= 0.05).sort((a, b) => (b.truncation_rate ?? 0) - (a.truncation_rate ?? 0))) {
    items.push({
      icon: <Scissors />,
      tone: s.warn,
      text: (
        <>
          {formatRate(t.truncation_rate, 0)} of <strong>{t.task_name}</strong> outputs hit max length
        </>
      ),
      href: buildHref(`/runs/${ctx.run.run_id}`, { tab: "instances", task: t.task_name, fin: "length" }),
    });
  }
  if (mismatched.length) {
    const diffs = mismatched.map((t) => {
      const a = theirs.data?.task_configs.find((c) => c.task_name === t.task_name)?.config;
      const b = mine.data?.task_configs.find((c) => c.task_name === t.task_name)?.config;
      const keys = a && b ? jsonDiff(a, b).filter((e) => e.kind !== "same" && e.path !== "task_hash").map((e) => e.path) : [];
      return `${t.task_name}${keys.length ? `: ${keys.join(", ")}` : ""}`;
    });
    items.push({
      icon: <FileWarning />,
      tone: s.warn,
      text: (
        <>
          {mismatched.length} task{mismatched.length > 1 ? "s have" : " has"} a different config than the baseline
        </>
      ),
      detail: diffs.join(" · "),
      href: buildHref(`/runs/${ctx.run.run_id}`, { tab: "config" }),
    });
  }
  return (
    <Panel title="Health" caption="Failures, truncation and config drift. Each line opens the matching instances.">
      {ctx.tasks.isLoading ? (
        <Skeleton height={80} />
      ) : items.length === 0 ? (
        <div className={s.healthItem}>
          <CheckCircle2 className={s.ok} />
          <span>No problems found: no failed tasks, no failed instances, truncation under 5%.</span>
        </div>
      ) : (
        <div className={s.health}>
          {items.map((it, i) => {
            const body = (
              <>
                <span className={it.tone}>{it.icon}</span>
                <span style={{ minWidth: 0 }}>
                  {it.text}
                  {it.detail && <div className={s.healthDetail}>{it.detail}</div>}
                </span>
              </>
            );
            return it.href ? (
              <AppLink key={i} href={it.href} className={s.healthItem}>
                {body}
              </AppLink>
            ) : (
              <div key={i} className={s.healthItem}>
                {body}
              </div>
            );
          })}
        </div>
      )}
    </Panel>
  );
}

// ---------------------------------------------------------------- provenance

export function Provenance({ run }: { run: RunDetail }) {
  const b = run.beaker;
  const env = run.environment;
  const id = (value: string | null | undefined, href?: string | null) =>
    value ? (
      <span className="row" style={{ gap: 6 }}>
        <CopyText text={value} display={<span className="mono">{value}</span>} />
        {href && <LinkOut href={href}>open</LinkOut>}
      </span>
    ) : null;
  return (
    <Panel title="Provenance" caption="Where and how this run executed. Click an ID to copy it.">
      <div className={s.provGrid}>
        <KV
          items={[
            ["Experiment", id(b?.experiment_id, run.links.beaker_experiment)],
            ["Job", id(b?.job_id, b?.job_id ? `https://beaker.org/job/${b.job_id}` : null)],
            ["Workload", id(b?.workload_id)],
            ["Result dataset", id(b?.result_dataset_id, run.links.beaker_result_dataset)],
            ["Workspace", b?.workspace ? <LinkOut href={run.links.beaker_workspace}>{b.workspace}</LinkOut> : null],
          ]}
        />
        <KV
          items={[
            ["Cluster", b?.cluster],
            ["Node", b?.node_hostname ? <span className="mono" style={{ fontSize: 11.5 }}>{b.node_hostname}</span> : null],
            ["Priority", b?.priority],
            ["Image", b?.image ? <span className="mono" style={{ fontSize: 11.5 }}>{b.image}</span> : null],
            ["GPUs", env.gpu_type ? `${env.gpu_count ?? b?.gpu_count ?? "?"}× ${env.gpu_type}` : null],
          ]}
        />
        <KV
          items={[
            ["Started", absoluteLocal(run.started_at)],
            ["Finished", run.finished_at ? absoluteLocal(run.finished_at) : <span className="muted">not finished</span>],
            ["Duration", formatDuration(run.duration_seconds)],
            ["Startup", run.startup_seconds != null ? formatRuntime(run.startup_seconds) : <span className="muted">not recorded</span>],
            ["Processing", run.processing_seconds != null ? formatRuntime(run.processing_seconds) : <span className="muted">{run.startup_seconds != null && !run.finished_at ? "in progress" : "not recorded"}</span>],
            ["olmo-eval", run.olmo_eval_version ? <span className="mono">{run.olmo_eval_version} @ {shortHash(run.git.commit)}</span> : null],
            ["Stack", <span className="mono" style={{ fontSize: 11.5 }}>{["vllm", "torch", "transformers"].filter((p) => env.packages[p]).map((p) => `${p} ${env.packages[p]}`).join(" · ") || "—"}</span>],
            ["Results", <CopyText text={run.gcs_prefix} display={<span className="mono" style={{ fontSize: 11.5 }}>{run.gcs_prefix}</span>} />],
          ]}
        />
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- tab

export function OverviewTab(ctx: RunCtx) {
  const tasks = useMemo(() => ctx.tasks.data?.items ?? [], [ctx.tasks.data]);
  const suites = useMemo(() => ctx.suites.data?.items ?? [], [ctx.suites.data]);
  const openTask = (row: TaskResultRow, cell?: string) => ctx.setSearch({ tab: "tasks", task: row.task_name, cell });

  const tiles: TileSpec[] = useMemo(() => {
    const top = suites.filter((r) => r.depth === 0).slice(0, 6);
    if (top.length) {
      return top.map((r) => ({
        key: `suite:${r.name}`,
        name: r.name,
        score: r.score,
        stderr: r.stderr,
        format: r.display_format,
        higherIsBetter: r.higher_is_better,
        delta: r.baseline?.delta ?? null,
        colKey: `suite:${r.name}`,
        scrollTo: `ov-suite:${r.name}`,
      }));
    }
    return [...tasks]
      .sort((a, b) => a.task_name.localeCompare(b.task_name))
      .slice(0, 6)
      .map((t) => ({
        key: `task:${t.task_name}`,
        name: t.task_name,
        score: t.score,
        stderr: t.stderr,
        format: scoreFormat(t),
        higherIsBetter: rowMeta(t).higherIsBetter,
        delta: t.baseline?.delta ?? null,
        colKey: `task:${t.task_name}`,
        scrollTo: `ov-task:${t.task_name}`,
      }));
  }, [suites, tasks]);

  const changes: DivergingItem[] = useMemo(() => {
    const withDelta = tasks
      .filter((t) => t.baseline && t.baseline.delta.delta != null)
      .map((t) => ({
        key: t.task_name,
        label: t.task_name,
        stats: t.baseline!.delta,
        higherIsBetter: rowMeta(t).higherIsBetter,
        format: scoreFormat(t),
        thisScore: t.score,
        baseScore: t.baseline!.score,
        g: goodness(t.baseline!.delta.delta, rowMeta(t).higherIsBetter) / (scoreFormat(t) === "percent" ? 1 : 100),
      }));
    const gains = withDelta.filter((x) => x.g > 0).sort((a, b) => b.g - a.g).slice(0, 8);
    const losses = withDelta.filter((x) => x.g < 0).sort((a, b) => a.g - b.g).slice(0, 8);
    return [...gains, ...losses.reverse()];
  }, [tasks]);

  return (
    <div className="col" style={{ gap: 14 }}>
      {ctx.tasks.isLoading ? (
        <div className={s.tiles}>
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} height={104} />
          ))}
        </div>
      ) : (
        tiles.length > 0 && <Tiles ctx={ctx} tiles={tiles} />
      )}
      <SuiteTree ctx={ctx} onOpenTask={(row) => openTask(row)} />
      <div className="grid-12">
        <div className="span-7">
          <ChartPanel
            id="biggest-changes"
            title="Biggest changes vs baseline"
            caption="Top gains and losses by task, with 95% paired CIs. Faded bars are not significant."
            csv={() => ({
              header: ["task", "delta", "ci_low", "ci_high", "p", "significant"],
              rows: changes.map((c) => [c.label, c.stats.delta, c.stats.ci_low, c.stats.ci_high, c.stats.p_value, c.stats.significant]),
            })}
            table={() => (
              <div className="col" style={{ gap: 2, fontSize: 12.5 }}>
                {changes.map((c) => (
                  <div key={c.key} className="row" style={{ justifyContent: "space-between" }}>
                    <span>{c.label}</span>
                    <DeltaValue stats={c.stats} format={c.format} higherIsBetter={c.higherIsBetter} ci />
                  </div>
                ))}
              </div>
            )}
          >
            {!ctx.baseline ? (
              <EmptyState title="Pick a baseline to see what changed" compact>
                Use the suggestions above (previous checkpoint, same group) or press <code>b</code> on another run.
              </EmptyState>
            ) : ctx.tasks.isLoading ? (
              <Skeleton height={240} />
            ) : (
              <DivergingBars
                items={changes}
                onClick={(it) => {
                  const row = tasks.find((t) => t.task_name === it.key);
                  if (row) openTask(row, goodness(it.stats.delta, it.higherIsBetter) > 0 ? "only_a" : "only_b");
                }}
              />
            )}
          </ChartPanel>
        </div>
        <div className="span-5">
          <HealthPanel ctx={ctx} />
        </div>
      </div>
      <Provenance run={ctx.run} />
    </div>
  );
}
