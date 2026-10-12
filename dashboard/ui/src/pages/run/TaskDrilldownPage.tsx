import { useParams } from "@tanstack/react-router";
import { useResolveSubjects } from "@/api/hooks/compare";
import { useRun, useTaskResults } from "@/api/hooks/runs";
import { buildHref, useAppNavigate } from "@/components/AppLink";
import { PageHeader } from "@/components/PageHeader";
import { Badge, ErrorPanel, Skeleton } from "@/components/primitives";
import { modelLabel, runSubject } from "@/lib/subjects";
import { useBaseline } from "@/state/nav";
import { NotFoundPage } from "../NotFoundPage";
import { TaskDrilldown } from "./TaskDrilldown";

export function TaskDrilldownPage() {
  const { runId, taskName } = useParams({ strict: false }) as { runId: string; taskName: string };
  const task = decodeURIComponent(taskName);
  const [baseline] = useBaseline();
  const effective = baseline && baseline !== runSubject(runId) ? baseline : undefined;
  const run = useRun(runId);
  const tasks = useTaskResults(runId, effective);
  const resolved = useResolveSubjects(effective ? [effective] : []);
  const navigate = useAppNavigate();
  if (run.error) return <NotFoundPage what={`Run ${runId}`} />;
  const row = tasks.data?.items.find((t) => t.task_name === task);
  const info = resolved.data?.items[0];
  return (
    <div className="page">
      <PageHeader
        crumbs={[
          { label: "Runs", href: "/runs" },
          { label: run.data ? modelLabel(run.data.model) : runId, href: `/runs/${runId}` },
          { label: "Tasks", href: buildHref(`/runs/${runId}`, { tab: "tasks" }) },
        ]}
        title={task}
        titleExtra={row?.error ? <Badge tone="danger">failed</Badge> : undefined}
      />
      {tasks.error ? (
        <ErrorPanel error={tasks.error} onRetry={() => tasks.refetch()} />
      ) : !run.data || !tasks.data ? (
        <Skeleton height={400} />
      ) : !row ? (
        <NotFoundPage what={`Task ${task} in this run`} />
      ) : (
        <TaskDrilldown
          run={run.data}
          row={row}
          baseline={effective}
          baselineRunId={info?.run?.run_id ?? info?.run_ids[0]}
          onOpenInstances={(patch) => navigate(buildHref(`/runs/${runId}`, { tab: "instances", task, ...patch }))}
        />
      )}
    </div>
  );
}
