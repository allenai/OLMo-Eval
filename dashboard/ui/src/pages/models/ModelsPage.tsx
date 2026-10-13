import type { ModelSeriesRow } from "@contract/api-types";
import { useMemo } from "react";
import { useModels, usePagedList } from "@/api/hooks/catalog";
import { AppLink, useAppNavigate } from "@/components/AppLink";
import { type Column, DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";
import { Badge, ErrorPanel, Panel, RelativeTime, SearchInput, Select } from "@/components/primitives";
import { formatCount, formatScore } from "@/lib/format";
import { useSearchParams } from "@/state/nav";

export function ModelsPage() {
  const [search, setSearch] = useSearchParams<{ q?: string; family?: string; sort?: string }>();
  const models = usePagedList<ModelSeriesRow>("/models", { q: search.q, family: search.family, sort: search.sort }, 100);
  const all = useModels({});
  const navigate = useAppNavigate();
  const families = useMemo(() => Array.from(new Set((all.data?.items ?? []).map((m) => m.family).filter((f): f is string => !!f))).sort(), [all.data]);
  const columns: Column<ModelSeriesRow>[] = [
    {
      id: "series",
      header: "Model series",
      title: "Model series",
      width: 300,
      pin: true,
      sortValue: (r) => r.series_label.toLowerCase(),
      cell: (r) => (
        <AppLink href={`/models/${encodeURIComponent(r.series)}`} className="truncate" title={r.series}>
          {r.series_label}
        </AppLink>
      ),
      csv: (r) => r.series,
    },
    { id: "family", header: "Family", title: "Family", width: 100, sortValue: (r) => r.family, cell: (r) => (r.family ? <Badge tone="muted">{r.family}</Badge> : <span className="faint">—</span>), csv: (r) => r.family },
    { id: "ckpt", header: "Checkpoints", title: "Checkpoints", width: 110, align: "right", sortValue: (r) => r.n_checkpoints, cell: (r) => <span className="mono">{r.n_checkpoints}</span>, csv: (r) => r.n_checkpoints },
    { id: "runs", header: "Runs", title: "Runs", width: 80, align: "right", sortValue: (r) => r.n_runs, cell: (r) => <span className="mono">{r.n_runs}</span>, csv: (r) => r.n_runs },
    { id: "step", header: "Latest step", title: "Latest step", width: 110, align: "right", sortValue: (r) => r.latest_step, cell: (r) => <span className="mono">{r.latest_step != null ? formatCount(r.latest_step) : "—"}</span>, csv: (r) => r.latest_step },
    {
      id: "headline",
      header: "Latest headline",
      title: "Latest headline score",
      width: 170,
      align: "right",
      sortValue: (r) => r.latest_run?.headline?.score,
      cell: (r) =>
        r.latest_run?.headline ? (
          <span className="row" style={{ gap: 6 }}>
            <span className="t-caption">{r.latest_run.headline.name ?? "mean"}</span>
            <span className="mono">{formatScore(r.latest_run.headline.score, r.latest_run.headline.display_format)}</span>
          </span>
        ) : (
          <span className="faint">—</span>
        ),
      csv: (r) => r.latest_run?.headline?.score,
    },
    { id: "variants", header: "Variants", title: "Settings variants", width: 90, align: "right", sortValue: (r) => r.variants.length, cell: (r) => <span className="mono">{r.variants.length}</span>, csv: (r) => r.variants.length },
    { id: "last", header: "Last run", title: "Last run", width: 100, sortValue: (r) => r.last_run_at, cell: (r) => <RelativeTime value={r.last_run_at} className="muted" />, csv: (r) => r.last_run_at },
  ];
  return (
    <div className="page">
      <PageHeader title="Models" titleExtra={models.total != null && <span className="muted mono" style={{ fontSize: 14, marginTop: 4 }}>{formatCount(models.total)}</span>} meta="One row per model series. Checkpoints of a series share a name apart from the step." />
      <Panel pad="none">
        {models.error ? (
          <ErrorPanel error={models.error} onRetry={models.refetch} />
        ) : (
          <DataTable
            tableId="models"
            columns={columns}
            rows={models.rows}
            getRowId={(r) => r.series}
            loading={models.isLoading}
            fetching={models.isFetching}
            total={models.total}
            hasMore={models.hasMore}
            onLoadMore={models.loadMore}
            sort={search.sort}
            onSortChange={(v) => setSearch({ sort: v }, { replace: true })}
            clientSort
            onRowOpen={(r) => navigate(`/models/${encodeURIComponent(r.series)}`)}
            toolbar={
              <>
                <SearchInput placeholder="Find a model" value={search.q ?? ""} onChange={(e) => setSearch({ q: e.target.value || undefined }, { replace: true })} wrapStyle={{ width: 240 }} />
                <Select size="sm" label="Family" value={search.family ?? ""} onChange={(v) => setSearch({ family: v || undefined })} options={[{ value: "", label: "any" }, ...families.map((f) => ({ value: f, label: f }))]} />
              </>
            }
          />
        )}
      </Panel>
    </div>
  );
}
