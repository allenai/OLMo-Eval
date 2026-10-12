import { Bookmark, Trash2 } from "lucide-react";
import { useMe, useSavedViews, useViewMutations } from "@/api/hooks/catalog";
import { AppLink } from "@/components/AppLink";
import { PageHeader } from "@/components/PageHeader";
import { Badge, Checkbox, EmptyState, ErrorPanel, IconButton, Panel, RelativeTime, Skeleton } from "@/components/primitives";
import { toast } from "@/components/toast";
import type { SavedView } from "@contract/api-types";

function ViewList({ views, own }: { views: SavedView[]; own: boolean }) {
  const { update, remove } = useViewMutations();
  if (!views.length) return <EmptyState icon={<Bookmark />} title={own ? "No saved views yet" : "No shared views"} compact>{own ? "Use Save view on the Runs page to keep a set of filters, columns and sort." : "Views others share appear here."}</EmptyState>;
  return (
    <div className="col" style={{ gap: 0 }}>
      {views.map((v) => (
        <div key={v.id} className="row" style={{ padding: "8px 12px", borderTop: "1px solid var(--border)", gap: 12 }}>
          <Bookmark size={14} color="var(--muted)" />
          <div className="col" style={{ gap: 2, flex: 1, minWidth: 0 }}>
            <AppLink href={`/${v.page}?${v.query}`} keepBaseline={false} className="link" style={{ fontSize: 13.5 }}>
              {v.name}
            </AppLink>
            <span className="mono muted truncate" style={{ fontSize: 11 }}>
              /{v.page}?{v.query}
            </span>
          </div>
          {!own && <span className="t-caption">{v.owner_email.split("@")[0]}</span>}
          {v.shared && <Badge tone="teal">shared</Badge>}
          <RelativeTime value={v.updated_at} className="t-caption" />
          {own && (
            <>
              <Checkbox checked={v.shared} onChange={(on) => update.mutate({ id: v.id, body: { shared: on } }, { onSuccess: () => toast(on ? "View shared" : "View is now private") })} label="Shared" />
              <IconButton size="sm" label="Delete view" icon={<Trash2 />} onClick={() => window.confirm(`Delete view “${v.name}”?`) && remove.mutate(v.id, { onSuccess: () => toast("View deleted") })} />
            </>
          )}
        </div>
      ))}
    </div>
  );
}

export function ViewsPage() {
  const views = useSavedViews("runs");
  const me = useMe();
  return (
    <div className="page">
      <PageHeader title="Saved views" meta={`Views store only the URL query string. Signed in as ${me.data?.email ?? "…"}.`} />
      {views.error ? (
        <ErrorPanel error={views.error} onRetry={() => views.refetch()} />
      ) : views.isLoading ? (
        <Skeleton height={200} />
      ) : (
        <div className="grid-12">
          <div className="span-6">
            <Panel title="My views" pad="none">
              <ViewList views={views.data!.mine} own />
            </Panel>
          </div>
          <div className="span-6">
            <Panel title="Shared with everyone" pad="none">
              <ViewList views={views.data!.shared} own={false} />
            </Panel>
          </div>
        </div>
      )}
    </div>
  );
}
