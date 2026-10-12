import type { GroupRow } from "@contract/api-types";
import { usePagedList } from "@/api/hooks/catalog";
import { MiniHeatmap } from "@/charts/basic";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { type Column, DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";
import { ErrorPanel, Panel, RelativeTime, SearchInput } from "@/components/primitives";
import { shortDate } from "@/lib/time";
import { useSearchParams } from "@/state/nav";

export function GroupsPage() {
  const [search, setSearch] = useSearchParams<{ q?: string; sort?: string }>();
  const groups = usePagedList<GroupRow>("/groups", { q: search.q, include_heatmap: true }, 50);
  const navigate = useAppNavigate();
  const columns: Column<GroupRow>[] = [
    {
      id: "name",
      header: "Group",
      title: "Group",
      width: 280,
      pin: true,
      sortValue: (r) => r.name,
      cell: (r) => (
        <AppLink href={`/groups/${encodeURIComponent(r.name)}`} className="truncate">
          {r.name}
        </AppLink>
      ),
      csv: (r) => r.name,
    },
    { id: "runs", header: "Runs", title: "Runs", width: 70, align: "right", sortValue: (r) => r.n_runs, cell: (r) => <span className="mono">{r.n_runs}</span>, csv: (r) => r.n_runs },
    { id: "models", header: "Models", title: "Models", width: 76, align: "right", sortValue: (r) => r.n_models, cell: (r) => <span className="mono">{r.n_models}</span>, csv: (r) => r.n_models },
    { id: "tasks", header: "Tasks", title: "Tasks", width: 70, align: "right", sortValue: (r) => r.n_tasks, cell: (r) => <span className="mono">{r.n_tasks}</span>, csv: (r) => r.n_tasks },
    { id: "users", header: "Users", title: "Users", width: 150, cell: (r) => <span className="truncate muted">{r.users.join(", ")}</span>, csv: (r) => r.users.join(" ") },
    {
      id: "coverage",
      header: "Suite scores",
      title: "Suite scores (models × suites)",
      width: 160,
      cell: (r) => <div style={{ width: 140 }}>{r.heatmap ? <MiniHeatmap data={{ ...r.heatmap, rows: r.heatmap.rows.slice(0, 2), values: r.heatmap.values.slice(0, 2) }} /> : null}</div>,
    },
    { id: "range", header: "Active", title: "Date range", width: 140, sortValue: (r) => r.first_run_at, cell: (r) => <span className="muted">{shortDate(r.first_run_at)} – {shortDate(r.last_run_at)}</span>, csv: (r) => r.first_run_at },
    { id: "last", header: "Updated", title: "Last run", width: 96, sortValue: (r) => r.last_run_at, cell: (r) => <RelativeTime value={r.last_run_at} className="muted" />, csv: (r) => r.last_run_at },
  ];
  return (
    <div className="page">
      <PageHeader title="Groups" meta="Experiment groups (--experiment-group). Open one to see its runs, coverage and a preset comparison." />
      <Panel pad="none">
        {groups.error ? (
          <ErrorPanel error={groups.error} onRetry={groups.refetch} />
        ) : (
          <DataTable
            tableId="groups"
            columns={columns}
            rows={groups.rows}
            getRowId={(r) => r.name}
            loading={groups.isLoading}
            fetching={groups.isFetching}
            total={groups.total}
            hasMore={groups.hasMore}
            onLoadMore={groups.loadMore}
            sort={search.sort}
            onSortChange={(v) => setSearch({ sort: v }, { replace: true })}
            clientSort
            onRowOpen={(r) => navigate(`/groups/${encodeURIComponent(r.name)}`)}
            toolbar={<SearchInput placeholder="Find a group" value={search.q ?? ""} onChange={(e) => setSearch({ q: e.target.value || undefined }, { replace: true })} wrapStyle={{ width: 260 }} />}
          />
        )}
      </Panel>
    </div>
  );
}
