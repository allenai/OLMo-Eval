import type { TaskConfigEntry } from "@contract/api-types";
import { Copy } from "lucide-react";
import { type ReactNode, useState } from "react";
import { useRunConfigs } from "@/api/hooks/runs";
import { type Column, DataTable } from "@/components/DataTable";
import { JsonDiff, JsonTree } from "@/components/Json";
import { Badge, Button, Checkbox, EmptyState, ErrorPanel, KV, LinkOut, Panel, Skeleton } from "@/components/primitives";
import { toast } from "@/components/toast";
import { copyText } from "@/lib/csv";
import { shortHash } from "@/lib/format";
import s from "./run.module.css";
import type { RunCtx } from "./types";

export function ConfigTab(ctx: RunCtx) {
  const configs = useRunConfigs(ctx.run.run_id);
  const baseRunId = ctx.baselineInfo?.run?.run_id ?? ctx.baselineInfo?.run_ids[0];
  const [diff, setDiff] = useState(!!baseRunId);
  const base = useRunConfigs(diff ? baseRunId : null);
  const [openTask, setOpenTask] = useState<string | null>(null);
  if (configs.isLoading) return <Skeleton height={400} />;
  if (configs.error) return <ErrorPanel error={configs.error} onRetry={() => configs.refetch()} />;
  const c = configs.data!;
  const b = diff ? base.data : undefined;
  const env = c.environment;
  const baseTaskHash = new Map((b?.task_configs ?? []).map((t) => [t.task_name, t.task_hash]));
  const selected = c.task_configs.find((t) => t.task_name === openTask) ?? c.task_configs[0];
  const baseSelected = b?.task_configs.find((t) => t.task_name === selected?.task_name);
  const columns: Column<TaskConfigEntry>[] = [
    { id: "task", header: "Task", title: "Task", width: 240, sortValue: (r) => r.task_name, cell: (r) => <span className="truncate">{r.task_name}</span>, csv: (r) => r.task_name },
    {
      id: "hash",
      header: "Variant",
      title: "Variant hash",
      width: 120,
      cell: (r) => {
        const bh = baseTaskHash.get(r.task_name);
        return (
          <span className="row" style={{ gap: 6 }}>
            <span className="mono muted" style={{ fontSize: 11.5 }}>{shortHash(r.task_hash)}</span>
            {bh && bh !== r.task_hash && <Badge tone="warn">differs</Badge>}
          </span>
        );
      },
      csv: (r) => r.task_hash,
    },
  ];
  const label = ctx.baselineInfo?.label ?? "Baseline";
  return (
    <div className="col" style={{ gap: 14 }}>
      {baseRunId ? (
        <div className="row-wrap" style={{ gap: 10 }}>
          <Checkbox checked={diff} onChange={setDiff} label={`Diff against baseline (${label})`} />
          <span className="t-caption">Changed keys are highlighted, added in teal, removed in ochre. Unchanged keys are hidden.</span>
        </div>
      ) : (
        <span className="t-caption">Set a baseline to diff configs and see why two runs differ.</span>
      )}
      <div className="grid-12">
        <div className="span-7">
          <Panel title="Model and provider config" caption={`${ctx.run.model_detail.provider_kind} · settings ${shortHash(ctx.run.model_detail.settings_hash, 10)}`}>
            {diff && b ? <JsonDiff a={b.model_config} b={c.model_config} aLabel={label} /> : diff && base.isLoading ? <Skeleton height={160} /> : <JsonTree value={c.model_config} filename="model-config.json" maxHeight={420} />}
          </Panel>
        </div>
        <div className="span-5">
          <Panel title="Environment">
            {diff && b ? (
              <JsonDiff a={b.environment} b={env} aLabel={label} />
            ) : (
              <KV
                items={[
                  ["olmo-eval", <span className="mono">{ctx.run.olmo_eval_version ?? "—"}</span>],
                  [
                    "Commit",
                    ctx.run.git.commit ? (
                      <span className="row" style={{ gap: 6 }}>
                        <LinkOut href={ctx.run.links.github_commit}>
                          <span className="mono">{shortHash(ctx.run.git.commit, 10)}</span>
                        </LinkOut>
                        <span className="muted">{ctx.run.git.branch}</span>
                        {ctx.run.git.dirty && <Badge tone="warn">dirty</Badge>}
                      </span>
                    ) : null,
                  ],
                  ["Python", <span className="mono">{env.python_version ?? "—"}</span>],
                  ...Object.entries(env.packages).map(([k, v]) => [k, <span className="mono">{v}</span>] as [string, ReactNode]),
                  ["CUDA", <span className="mono">{env.cuda_version ?? "—"}</span>],
                  ["GPU", env.gpu_type ? `${env.gpu_count ?? "?"}× ${env.gpu_type}` : "—"],
                  ["Image", <span className="mono" style={{ fontSize: 11.5 }}>{ctx.run.beaker?.image ?? "—"}</span>],
                  ["Cluster", ctx.run.beaker?.cluster ?? "—"],
                  ["Host", <span className="mono" style={{ fontSize: 11.5 }}>{env.hostname ?? "—"}</span>],
                  ["Platform", <span className="mono" style={{ fontSize: 11 }}>{env.platform ?? "—"}</span>],
                ]}
              />
            )}
          </Panel>
        </div>
        <div className="span-12">
          <Panel title="Command" caption="The argv that launched this run, ready to copy.">
            <div className={s.text} style={{ maxHeight: 140 }}>
              {ctx.run.reproduce_command ?? c.argv.join(" ")}
            </div>
            <div className="row" style={{ marginTop: 6, gap: 12 }}>
              <Button
                size="sm"
                icon={<Copy />}
                onClick={() => copyText(ctx.run.reproduce_command ?? c.argv.join(" ")).then((ok) => ok && toast("Command copied"))}
              >
                Copy command
              </Button>
              <span className="t-caption">Task specs: {ctx.run.task_specs.join(", ") || "—"}</span>
            </div>
          </Panel>
        </div>
        {c.harness_config && (
          <div className="span-12">
            <Panel title="Harness config">
              {diff && b?.harness_config ? <JsonDiff a={b.harness_config} b={c.harness_config} aLabel={label} /> : <JsonTree value={c.harness_config} filename="harness-config.json" maxHeight={260} />}
            </Panel>
          </div>
        )}
        <div className="span-5">
          <Panel title="Per-task configs" caption="Click a task to view its config." pad="none">
            <DataTable
              tableId="task-configs"
              columns={columns}
              rows={c.task_configs}
              getRowId={(r) => r.task_name}
              onRowOpen={(r) => setOpenTask(r.task_name)}
              focusedId={selected?.task_name ?? null}
              keyboard={false}
              clientSort
              maxHeight={520}
              hideFooter
            />
          </Panel>
        </div>
        <div className="span-7">
          <Panel title={selected ? selected.task_name : "Task config"}>
            {!selected ? (
              <EmptyState title="No task configs" compact />
            ) : diff && baseSelected ? (
              <JsonDiff a={baseSelected.config} b={selected.config} aLabel={label} />
            ) : (
              <JsonTree value={selected.config} filename={`${selected.task_name}-config.json`} maxHeight={480} />
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
