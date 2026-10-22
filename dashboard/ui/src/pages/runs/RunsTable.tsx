import type { RunRow, RunsListResponse, ScoreColumnMeta } from "@contract/api-types";
import { ArrowRightLeft, ExternalLink, Rocket, Star } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useMemo } from "react";
import { apiGet } from "@/api/client";
import { useRunsInfinite } from "@/api/hooks/runs";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { DeltaValue, ScoreValue, SubjectLabel } from "@/components/cells";
import { type Column, DataTable } from "@/components/DataTable";
import { Badge, displayStatus, EmptyState, ErrorPanel, IconButton, LinkOut, RelativeTime, StatusBadge, Tip } from "@/components/primitives";
import { pageActions } from "@/components/shell/registry";
import { formatCount, formatDuration, formatScore, shortHash } from "@/lib/format";
import { modelLabel, runSubject } from "@/lib/subjects";
import { useSubjectSlots } from "@/state/colors";
import { useBaseline } from "@/state/nav";
import { addToTray, removeFromTray, trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";

export type GroupByKey = "none" | "group" | "launch" | "model";

export function runLabel(run: Pick<RunRow, "model">): string {
  return modelLabel(run.model);
}

export function RunsTable({
  params,
  tableId,
  sort,
  onSortChange,
  groupBy = "none",
  toolbar,
  maxHeight,
  empty,
  onResponse,
  keyboard = true,
  hideColumns = [],
}: {
  params: Record<string, unknown>;
  tableId: string;
  sort?: string;
  onSortChange?: (sort: string | undefined) => void;
  groupBy?: GroupByKey;
  toolbar?: ReactNode;
  maxHeight?: number | string;
  empty?: ReactNode;
  onResponse?: (data: RunsListResponse | undefined) => void;
  keyboard?: boolean;
  hideColumns?: string[];
}) {
  const query = useRunsInfinite(params);
  const [baseline, setBaseline] = useBaseline();
  const navigate = useAppNavigate();
  const tray = useStore(trayStore);
  const rows = useMemo(() => query.data?.pages.flatMap((p) => p.items) ?? [], [query.data]);
  const first = query.data?.pages[0];
  const scoreCols: ScoreColumnMeta[] = useMemo(() => first?.columns ?? [], [first]);
  // Only subjects in the compare tray get identity colors; other rows stay neutral.
  const slots = useSubjectSlots(tray.map((t) => t.key));

  useEffect(() => {
    onResponse?.(first);
  }, [first, onResponse]);

  const selected = useMemo(() => {
    const keys = new Set(tray.map((t) => t.key));
    return new Set(rows.filter((r) => keys.has(runSubject(r.run_id))).map((r) => r.run_id));
  }, [tray, rows]);

  const onSelectionChange = (next: Set<string>) => {
    for (const r of rows) {
      const key = runSubject(r.run_id);
      const was = selected.has(r.run_id);
      const now = next.has(r.run_id);
      if (now && !was) addToTray({ key, label: runLabel(r) });
      if (!now && was) removeFromTray(key);
    }
  };

  const columns: Column<RunRow>[] = useMemo(() => {
    const cols: Column<RunRow>[] = [
      {
        id: "status",
        header: "",
        title: "Status",
        width: 34,
        pin: true,
        align: "center",
        cell: (r) => <StatusBadge status={displayStatus(r)} iconOnly />,
        csv: (r) => displayStatus(r),
      },
      {
        id: "model",
        header: "Model",
        title: "Model",
        width: 230,
        pin: true,
        sortKey: "model",
        cell: (r) => {
          const key = runSubject(r.run_id);
          return (
            <Tip
              content={
                <span>
                  {r.model.name}
                  <br />
                  <span className="muted mono">hash {shortHash(r.model.model_hash, 12)} · run {r.run_id}</span>
                </span>
              }
              delay={500}
            >
              <AppLink href={`/runs/${r.run_id}`} className="truncate" style={{ display: "inline-flex", minWidth: 0 }}>
                <SubjectLabel model={r.model} slot={slots[key]} baseline={baseline === key} showStep={false} />
              </AppLink>
            </Tip>
          );
        },
        csv: (r) => r.model.name,
      },
      {
        id: "step",
        header: "Step",
        title: "Step",
        width: 76,
        align: "right",
        sortKey: "step",
        cell: (r) => <span className="mono">{r.model.step != null ? formatCount(r.model.step) : "—"}</span>,
        csv: (r) => r.model.step,
      },
      {
        id: "group",
        grow: true,
        minWidth: 130,
        header: "Group",
        title: "Group",
        width: 200,
        sortKey: "group",
        cell: (r) =>
          r.experiment_group ? (
            <AppLink href={`/groups/${encodeURIComponent(r.experiment_group)}`} className="truncate link" style={{ textDecorationColor: "transparent" }}>
              {r.experiment_group}
            </AppLink>
          ) : (
            <span className="faint">—</span>
          ),
        csv: (r) => r.experiment_group,
      },
      {
        id: "tasks",
        header: "Tasks",
        title: "Tasks",
        width: 76,
        align: "right",
        sortKey: "num_tasks",
        cell: (r) => (
          <span className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
            <span className="mono">{r.num_tasks}</span>
            {r.num_failed_tasks > 0 && (
              <Badge tone="danger" title={`${r.num_failed_tasks} failed`}>
                {r.num_failed_tasks}
              </Badge>
            )}
          </span>
        ),
        csv: (r) => r.num_tasks,
      },
      {
        id: "headline",
        header: "Headline",
        title: "Headline score",
        width: 130,
        align: "right",
        defaultHidden: true,
        cell: (r) =>
          r.headline ? (
            <Tip content={r.headline.name ?? "Mean of primary metrics"}>
              <span>
                <ScoreValue score={r.headline.score} stderr={r.headline.stderr} format={r.headline.display_format} />
              </span>
            </Tip>
          ) : (
            <span className="faint">—</span>
          ),
        csv: (r) => r.headline?.score,
      },
      ...scoreCols.map(
        (col): Column<RunRow> => ({
          id: `score:${col.key}`,
          header: (
            <span title={col.key} style={{ textTransform: "none", letterSpacing: 0, fontWeight: 800, fontSize: 11.5 }}>
              {col.name}
            </span>
          ),
          title: col.name,
          width: baseline ? 150 : 110,
          align: "right",
          sortKey: `score:${col.key}`,
          cell: (r) => {
            const v = r.scores[col.key];
            if (!v || v.score == null) return <span className="faint mono">—</span>;
            return (
              <span className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
                <ScoreValue score={v.score} stderr={v.stderr} format={col.display_format} />
                {v.delta && <DeltaValue stats={v.delta} format={col.display_format} higherIsBetter={col.higher_is_better} />}
              </span>
            );
          },
          csv: (r) => {
            const v = r.scores[col.key];
            return v?.score != null ? Number(formatScore(v.score, col.display_format).replace("−", "-")) : null;
          },
        }),
      ),
      {
        id: "user",
        header: "User",
        title: "User",
        width: 90,
        sortKey: "user",
        cell: (r) => <span className="truncate">{r.author ?? "—"}</span>,
        csv: (r) => r.author,
      },
      {
        id: "tags",
        grow: true,
        minWidth: 80,
        header: "Tags",
        title: "Tags",
        width: 150,
        cell: (r) => (
          <span className="row" style={{ gap: 4, overflow: "hidden" }}>
            {r.tags.map((t) => (
              <Badge key={t} tone="muted">
                {t}
              </Badge>
            ))}
          </span>
        ),
        csv: (r) => r.tags.join(" "),
      },
      {
        id: "created",
        header: "Created",
        title: "Created",
        width: 88,
        sortKey: "created_at",
        cell: (r) => <RelativeTime value={r.created_at} className="muted" />,
        csv: (r) => r.created_at,
      },
      {
        id: "duration",
        header: "Duration",
        title: "Duration",
        width: 84,
        align: "right",
        sortKey: "duration",
        cell: (r) => <span className="mono muted">{formatDuration(r.duration_seconds)}</span>,
        csv: (r) => r.duration_seconds,
      },
      {
        id: "workspace",
        header: "Workspace",
        title: "Workspace",
        width: 130,
        defaultHidden: true,
        cell: (r) => <span className="truncate muted">{r.beaker_workspace ?? "—"}</span>,
        csv: (r) => r.beaker_workspace,
      },
      {
        id: "commit",
        header: "Commit",
        title: "Commit",
        width: 90,
        defaultHidden: true,
        cell: (r) => (
          <LinkOut href={r.links.github_commit}>
            <span className="mono">{shortHash(r.git_commit)}</span>
          </LinkOut>
        ),
        csv: (r) => r.git_commit,
      },
      {
        id: "links",
        header: "Links",
        title: "Links",
        width: 124,
        hideable: false,
        cell: (r) => (
          <span className="row" style={{ gap: 10 }}>
            <LinkOut href={r.links.beaker_experiment} title="Beaker experiment">
              Beaker
            </LinkOut>
            <LinkOut href={r.links.gcs_console} title="GCS results">
              GCS
            </LinkOut>
          </span>
        ),
        csv: (r) => r.links.beaker_experiment,
      },
    ];
    return cols.filter((c) => !hideColumns.includes(c.id));
  }, [scoreCols, slots, baseline, hideColumns]);

  const fetchAll = useCallback(async () => {
    const all: RunRow[] = [];
    let cursor: string | null = null;
    do {
      const page: RunsListResponse = await apiGet<RunsListResponse>("/runs", { ...params, limit: 500, cursor: cursor ?? undefined });
      all.push(...page.items);
      cursor = page.next_cursor;
    } while (cursor && all.length < 100_000);
    return all;
  }, [params]);

  useEffect(() => {
    if (!keyboard) return;
    pageActions.exportTable = () => {
      document.querySelector<HTMLButtonElement>('[aria-label="Table actions"]')?.click();
    };
    return () => {
      pageActions.exportTable = undefined;
    };
  }, [keyboard]);

  const groupFn =
    groupBy === "group"
      ? (r: RunRow) => r.experiment_group ?? "No group"
      : groupBy === "launch"
        ? (r: RunRow) => r.launch_id ?? `run ${r.run_id}`
        : groupBy === "model"
          ? (r: RunRow) => r.model.series_label
          : undefined;

  if (query.error) return <div style={{ padding: 12 }}><ErrorPanel error={query.error} onRetry={() => query.refetch()} /></div>;

  return (
    <DataTable
      tableId={tableId}
      columns={columns}
      rows={groupBy === "model" ? [...rows].sort((a, b) => a.model.series.localeCompare(b.model.series) || (a.model.step ?? 0) - (b.model.step ?? 0)) : rows}
      getRowId={(r) => r.run_id}
      loading={query.isLoading}
      fetching={query.isFetching}
      sort={sort}
      onSortChange={onSortChange}
      selected={selected}
      onSelectionChange={onSelectionChange}
      onRowOpen={(r) => navigate(`/runs/${r.run_id}`)}
      groupBy={groupFn}
      renderGroup={(key, list) => (
        <>
          <strong>{key}</strong>
          <span className="muted">{list.length} runs</span>
        </>
      )}
      total={first?.total}
      hasMore={query.hasNextPage}
      onLoadMore={() => void query.fetchNextPage()}
      fetchAll={fetchAll}
      maxHeight={maxHeight}
      keyboard={keyboard}
      toolbar={toolbar}
      exportName={tableId}
      rowActions={(r) => {
        const key = runSubject(r.run_id);
        return (
          <>
            <IconButton size="sm" label="Add to compare (x)" icon={<ArrowRightLeft />} onClick={() => addToTray({ key, label: runLabel(r) })} />
            <IconButton
              size="sm"
              label={baseline === key ? "Baseline" : "Set as baseline"}
              icon={<Star fill={baseline === key ? "currentColor" : "none"} />}
              onClick={() => setBaseline(baseline === key ? null : key, runLabel(r))}
            />
            {r.links.beaker_experiment && (
              <IconButton size="sm" label="Open in Beaker" icon={<ExternalLink />} onClick={() => window.open(r.links.beaker_experiment!, "_blank", "noopener")} />
            )}
          </>
        );
      }}
      empty={
        empty ?? (
          <EmptyState icon={<Rocket />} title="No runs match these filters">
            Remove a filter to widen the search, or launch an eval with <code>olmo-eval run --beaker ...</code>.
          </EmptyState>
        )
      }
    />
  );
}
