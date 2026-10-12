import type { SuiteRow, TaskRow } from "@contract/api-types";
import { usePagedList, useSuites, useTasks } from "@/api/hooks/catalog";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { MetricName } from "@/components/cells";
import { type Column, DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";
import { Badge, ErrorPanel, Panel, RelativeTime, SearchInput, Select, Tabs } from "@/components/primitives";
import { useSearchParams } from "@/state/nav";

export function TasksPage() {
  const [search, setSearch] = useSearchParams<{ q?: string; suite?: string; sort?: string; tab?: string }>();
  const tab = search.tab === "suites" ? "suites" : "tasks";
  const tasks = usePagedList<TaskRow>("/tasks", { q: tab === "tasks" ? search.q : undefined, suite: search.suite }, 200);
  const suites = usePagedList<SuiteRow>("/suites", { q: tab === "suites" ? search.q : undefined }, 200);
  // Unfiltered counts for the header and tabs, and every suite name for the suite filter.
  const allTasks = useTasks({ limit: 1 });
  const allSuites = useSuites();
  const navigate = useAppNavigate();
  const taskCols: Column<TaskRow>[] = [
    {
      id: "task",
      header: "Task",
      title: "Task",
      width: 280,
      pin: true,
      sortValue: (r) => r.task_name,
      cell: (r) => (
        <AppLink href={`/tasks/${encodeURIComponent(r.task_name)}`} className="truncate">
          {r.task_name}
        </AppLink>
      ),
      csv: (r) => r.task_name,
    },
    {
      id: "suites",
      header: "Suites",
      title: "Suites",
      width: 240,
      cell: (r) => (
        <span className="row" style={{ gap: 4, overflow: "hidden" }}>
          {r.suites.length ? r.suites.map((s) => <Badge key={s} tone="muted">{s}</Badge>) : <span className="faint">—</span>}
        </span>
      ),
      csv: (r) => r.suites.join(" "),
    },
    { id: "metric", header: "Primary metric", title: "Primary metric", width: 200, cell: (r) => <MetricName metric={r.primary_metric} />, csv: (r) => r.primary_metric },
    { id: "variants", header: "Variants", title: "Variants", width: 86, align: "right", sortValue: (r) => r.n_variants, cell: (r) => <span className="mono">{r.n_variants}</span>, csv: (r) => r.n_variants },
    { id: "runs", header: "Runs", title: "Runs", width: 76, align: "right", sortValue: (r) => r.n_runs, cell: (r) => <span className="mono">{r.n_runs}</span>, csv: (r) => r.n_runs },
    { id: "models", header: "Models", title: "Models", width: 80, align: "right", sortValue: (r) => r.n_models, cell: (r) => <span className="mono">{r.n_models}</span>, csv: (r) => r.n_models },
    { id: "last", header: "Last run", title: "Last run", width: 100, sortValue: (r) => r.last_run_at, cell: (r) => <RelativeTime value={r.last_run_at} className="muted" />, csv: (r) => r.last_run_at },
  ];
  const suiteCols: Column<SuiteRow>[] = [
    {
      id: "suite",
      header: "Suite",
      title: "Suite",
      width: 220,
      pin: true,
      sortValue: (r) => r.suite_name,
      cell: (r) => (
        <AppLink href={`/suites/${encodeURIComponent(r.suite_name)}`} className="truncate">
          {r.suite_name}
        </AppLink>
      ),
      csv: (r) => r.suite_name,
    },
    { id: "desc", header: "Description", title: "Description", width: 360, cell: (r) => <span className="truncate muted">{r.description ?? "—"}</span>, csv: (r) => r.description },
    { id: "agg", header: "Aggregation", title: "Aggregation", width: 160, cell: (r) => <span className="muted">{r.aggregation.replace(/_/g, " ")}</span>, csv: (r) => r.aggregation },
    { id: "children", header: "Children", title: "Children", width: 86, align: "right", sortValue: (r) => r.n_children, cell: (r) => <span className="mono">{r.n_children}</span>, csv: (r) => r.n_children },
    { id: "tasks", header: "Tasks", title: "Tasks", width: 76, align: "right", sortValue: (r) => r.n_tasks, cell: (r) => <span className="mono">{r.n_tasks}</span>, csv: (r) => r.n_tasks },
    { id: "runs", header: "Runs", title: "Runs", width: 76, align: "right", sortValue: (r) => r.n_runs, cell: (r) => <span className="mono">{r.n_runs}</span>, csv: (r) => r.n_runs },
    { id: "last", header: "Last run", title: "Last run", width: 100, sortValue: (r) => r.last_run_at, cell: (r) => <RelativeTime value={r.last_run_at} className="muted" />, csv: (r) => r.last_run_at },
  ];
  const suiteNames = (allSuites.data?.items ?? []).map((s) => s.suite_name);
  const nSuites = allSuites.data?.total ?? suiteNames.length;
  return (
    <div className="page">
      <PageHeader title="Tasks and suites" meta={`${allTasks.data?.total ?? "…"} tasks · ${nSuites} suites. Open one for its leaderboard, score distribution and history.`} />
      <Tabs
        label="Tasks or suites"
        value={tab}
        onChange={(v) => setSearch({ tab: v === "tasks" ? undefined : v, sort: undefined })}
        tabs={[
          { value: "tasks", label: "Tasks", count: allTasks.data?.total },
          { value: "suites", label: "Suites", count: nSuites },
        ]}
      />
      <Panel pad="none">
        {tab === "tasks" ? (
          tasks.error ? (
            <ErrorPanel error={tasks.error} onRetry={tasks.refetch} />
          ) : (
            <DataTable
              tableId="tasks"
              columns={taskCols}
              rows={tasks.rows}
              getRowId={(r) => r.task_name}
              loading={tasks.isLoading}
              fetching={tasks.isFetching}
              total={tasks.total}
              hasMore={tasks.hasMore}
              onLoadMore={tasks.loadMore}
              sort={search.sort}
              onSortChange={(v) => setSearch({ sort: v }, { replace: true })}
              clientSort
              onRowOpen={(r) => navigate(`/tasks/${encodeURIComponent(r.task_name)}`)}
              toolbar={
                <>
                  <SearchInput placeholder="Find a task (gsm cot matches gsm8k:cot)" value={search.q ?? ""} onChange={(e) => setSearch({ q: e.target.value || undefined }, { replace: true })} wrapStyle={{ width: 300 }} />
                  <Select size="sm" label="Suite" value={search.suite ?? ""} onChange={(v) => setSearch({ suite: v || undefined })} options={[{ value: "", label: "any" }, ...suiteNames.map((s) => ({ value: s, label: s }))]} />
                </>
              }
            />
          )
        ) : (
          <DataTable
            tableId="suites"
            columns={suiteCols}
            rows={suites.rows}
            getRowId={(r) => r.suite_name}
            loading={suites.isLoading}
            fetching={suites.isFetching}
            total={suites.total}
            hasMore={suites.hasMore}
            onLoadMore={suites.loadMore}
            sort={search.sort}
            onSortChange={(v) => setSearch({ sort: v }, { replace: true })}
            clientSort
            onRowOpen={(r) => navigate(`/suites/${encodeURIComponent(r.suite_name)}`)}
            toolbar={<SearchInput placeholder="Find a suite" value={search.q ?? ""} onChange={(e) => setSearch({ q: e.target.value || undefined }, { replace: true })} wrapStyle={{ width: 260 }} />}
          />
        )}
      </Panel>
    </div>
  );
}
