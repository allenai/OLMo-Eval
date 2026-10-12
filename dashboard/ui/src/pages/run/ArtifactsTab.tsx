import type { ArtifactRow } from "@contract/api-types";
import { Copy, Download, ExternalLink } from "lucide-react";
import { useArtifacts } from "@/api/hooks/runs";
import { type Column, DataTable } from "@/components/DataTable";
import { Badge, CopyText, ErrorPanel, IconButton, KV, LinkOut, Panel, RelativeTime, Skeleton } from "@/components/primitives";
import { toast } from "@/components/toast";
import { copyText } from "@/lib/csv";
import { formatBytes } from "@/lib/format";
import { openSigned } from "./TaskDrilldown";
import type { RunCtx } from "./types";

export function ArtifactsTab(ctx: RunCtx) {
  const art = useArtifacts(ctx.run.run_id);
  if (art.isLoading) return <Skeleton height={400} />;
  if (art.error) return <ErrorPanel error={art.error} onRetry={() => art.refetch()} />;
  const d = art.data!;
  const total = d.items.reduce((a, r) => a + r.size_bytes, 0);
  const columns: Column<ArtifactRow>[] = [
    { id: "path", header: "Path", title: "Path", width: 460, minWidth: 240, grow: true, sortValue: (r) => r.path, cell: (r) => <span className="mono truncate-start" style={{ fontSize: 12 }} title={r.path}><bdi>{r.path}</bdi></span>, csv: (r) => r.path },
    { id: "kind", header: "Kind", title: "Kind", width: 112, sortValue: (r) => r.kind, cell: (r) => <Badge tone="muted">{r.kind.replace("_", " ")}</Badge>, csv: (r) => r.kind },
    { id: "task", header: "Task", title: "Task", width: 200, minWidth: 120, grow: true, sortValue: (r) => r.task_name, cell: (r) => <span className="truncate muted">{r.task_name ?? "—"}</span>, csv: (r) => r.task_name },
    { id: "size", header: "Size", title: "Size", width: 90, align: "right", sortValue: (r) => r.size_bytes, cell: (r) => <span className="mono">{formatBytes(r.size_bytes)}</span>, csv: (r) => r.size_bytes },
    { id: "status", header: "Uploaded", title: "Uploaded", width: 100, cell: (r) => (r.uploaded ? <RelativeTime value={r.updated_at} className="muted" /> : <Badge tone="warn">missing</Badge>), csv: (r) => r.uploaded },
    {
      id: "actions",
      header: "",
      title: "Actions",
      width: 110,
      hideable: false,
      cell: (r) => (
        <span className="row" style={{ gap: 2 }}>
          <IconButton size="sm" label="Download (signed URL)" icon={<Download />} disabled={!r.uploaded} onClick={() => openSigned(r.gs_uri)} />
          <IconButton size="sm" label="Copy gs:// URI" icon={<Copy />} onClick={() => copyText(r.gs_uri).then(() => toast("Copied gs:// URI"))} />
          <IconButton size="sm" label="Open in Cloud Console" icon={<ExternalLink />} onClick={() => window.open(r.console_url, "_blank", "noopener")} />
        </span>
      ),
    },
  ];
  const b = d.beaker;
  return (
    <div className="grid-12">
      <div className="span-8">
        <Panel title="Files in GCS" caption={<>{d.items.length} files · {formatBytes(total)} · <CopyText text={d.gcs_prefix} display={<span className="mono">{d.gcs_prefix}</span>} /></>} pad="none" actions={<LinkOut href={d.gcs_console}>Cloud Console</LinkOut>}>
          <DataTable tableId="artifacts" columns={columns} rows={d.items} getRowId={(r) => r.path} clientSort maxHeight="calc(100vh - 340px)" exportName={`${ctx.run.run_id}-artifacts`} />
        </Panel>
      </div>
      <div className="span-4">
        <Panel title="Beaker" caption="Downloads use short-lived signed URLs created by the API.">
          {b ? (
            <KV
              items={[
                ["Experiment", b.experiment_id ? <LinkOut href={d.links.beaker_experiment}><span className="mono">{b.experiment_id}</span></LinkOut> : null],
                ["Job", b.job_id ? <LinkOut href={`https://beaker.org/job/${b.job_id}`}><span className="mono">{b.job_id}</span></LinkOut> : null],
                ["Workload", b.workload_id ? <span className="mono">{b.workload_id}</span> : null],
                ["Result dataset", b.result_dataset_id ? <LinkOut href={d.links.beaker_result_dataset}><span className="mono">{b.result_dataset_id}</span></LinkOut> : null],
                ["Workspace", b.workspace ? <LinkOut href={d.links.beaker_workspace}>{b.workspace}</LinkOut> : null],
                ["Budget", b.budget],
              ]}
            />
          ) : (
            <span className="t-caption">This run did not run on Beaker.</span>
          )}
        </Panel>
      </div>
    </div>
  );
}
