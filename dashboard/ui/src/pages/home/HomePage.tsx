import type { ActivityItem, RunSummary } from "@contract/api-types";
import { ArrowRight, History, Rocket } from "lucide-react";
import { useMe, useActivity, useGroups, useStatsSummary } from "@/api/hooks/catalog";
import { useRunsList } from "@/api/hooks/runs";
import { MiniHeatmap, Sparkline } from "@/charts/basic";
import { AppLink, buildHref, useAppNavigate } from "@/components/AppLink";
import { ScoreValue, SubjectLabel } from "@/components/cells";
import { Stat } from "@/components/PageHeader";
import { Button, displayStatus, EmptyState, ErrorPanel, Panel, RelativeTime, Skeleton, StatusBadge } from "@/components/primitives";
import { formatCompact, formatCount, pluralize } from "@/lib/format";
import { useSubjectSlots } from "@/state/colors";
import { recentStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { greeting, timeOfDay } from "@/lib/time";
import h from "./home.module.css";

function Headline({ run }: { run: RunSummary }) {
  if (!run.headline) return <span className="faint mono">—</span>;
  return (
    <span className="row" style={{ gap: 6, justifyContent: "flex-end" }}>
      <span className="t-caption hide-sm">{run.headline.name ?? "mean"}</span>
      <ScoreValue score={run.headline.score} stderr={run.headline.stderr} format={run.headline.display_format} showStderr={false} />
    </span>
  );
}

function RecentRuns() {
  const runs = useRunsList({ user: "me", limit: 8 });
  const navigate = useAppNavigate();
  const slots = useSubjectSlots((runs.data?.items ?? []).map((r) => `r:${r.run_id}`));
  return (
    <Panel
      title="Your recent runs"
      pad="none"
      refetching={runs.isFetching && !runs.isLoading}
      actions={
        <AppLink href="/runs?user=me">
          <Button size="sm" variant="ghost">
            All my runs <ArrowRight />
          </Button>
        </AppLink>
      }
    >
      {runs.isLoading ? (
        <div style={{ padding: 12, display: "grid", gap: 10 }}>
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} height={14} />
          ))}
        </div>
      ) : runs.error ? (
        <div style={{ padding: 12 }}>
          <ErrorPanel error={runs.error} onRetry={() => runs.refetch()} />
        </div>
      ) : !runs.data?.items.length ? (
        <EmptyState icon={<Rocket />} title="No runs yet" compact>
          Launch an eval with <code>olmo-eval run --beaker ...</code> and results appear here when it finishes.
        </EmptyState>
      ) : (
        <div className={h.list}>
          {runs.data.items.map((run) => (
            <a
              key={run.run_id}
              href={`/runs/${run.run_id}`}
              className={h.listRow}
              onClick={(e) => {
                if (e.metaKey || e.ctrlKey) return;
                e.preventDefault();
                navigate(`/runs/${run.run_id}`);
              }}
            >
              <StatusBadge status={displayStatus(run)} iconOnly />
              <span className={h.listMain}>
                <SubjectLabel model={run.model} slot={slots[`r:${run.run_id}`]} />
                <span className={h.listSub}>{run.experiment_group ?? "no group"}</span>
              </span>
              <Headline run={run} />
              <span className={h.age}>
                <RelativeTime value={run.created_at} />
              </span>
            </a>
          ))}
        </div>
      )}
    </Panel>
  );
}

function ActiveGroups() {
  const groups = useGroups({ active_days: 14, include_heatmap: true, limit: 6 });
  return (
    <Panel
      title="Active groups"
      caption="Groups with runs in the last 14 days. Cells are suite scores. Stronger teal is better within each column."
      refetching={groups.isFetching && !groups.isLoading}
      actions={
        <AppLink href="/groups">
          <Button size="sm" variant="ghost">
            All groups <ArrowRight />
          </Button>
        </AppLink>
      }
    >
      {groups.isLoading ? (
        <div style={{ display: "grid", gap: 12 }}>
          {Array.from({ length: 3 }, (_, i) => (
            <Skeleton key={i} height={64} />
          ))}
        </div>
      ) : groups.error ? (
        <ErrorPanel error={groups.error} onRetry={() => groups.refetch()} />
      ) : !groups.data?.items.length ? (
        <EmptyState title="No active groups" compact>
          Pass <code>--experiment-group</code> when launching to group related runs.
        </EmptyState>
      ) : (
        <div className={h.groups}>
          {groups.data.items.map((g) => (
            <AppLink key={g.name} href={`/groups/${encodeURIComponent(g.name)}`} className={h.groupCard}>
              <div className={h.groupHead}>
                <span className={h.groupName}>{g.name}</span>
                <RelativeTime value={g.last_run_at} className="t-caption" />
              </div>
              <div className={h.groupMeta}>
                {pluralize(g.n_runs, "run")} · {pluralize(g.n_models, "model")} · {pluralize(g.n_tasks, "task")} · {g.users.join(", ")}
              </div>
              {g.heatmap && <MiniHeatmap data={g.heatmap} />}
            </AppLink>
          ))}
        </div>
      )}
    </Panel>
  );
}

function activityText(item: ActivityItem) {
  const models = Array.from(new Set(item.runs.map((r) => r.model.series_label)));
  const verb = item.status === "failed" ? "run failed" : item.status === "running" ? "is running" : item.status === "partial" ? "finished with failures" : "launched";
  return { models, verb };
}

function OrgActivity() {
  const activity = useActivity({ limit: 15 });
  const items = activity.data?.pages.flatMap((p) => p.items) ?? [];
  return (
    <Panel title="Org activity" caption="All users, newest first, grouped by launch." pad="none">
      {activity.isLoading ? (
        <div style={{ padding: 12, display: "grid", gap: 10 }}>
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} height={16} />
          ))}
        </div>
      ) : activity.error ? (
        <div style={{ padding: 12 }}>
          <ErrorPanel error={activity.error} onRetry={() => activity.refetch()} />
        </div>
      ) : (
        <div className={h.feed}>
          {items.map((item) => {
            const { models, verb } = activityText(item);
            const first = item.runs[0];
            const href = item.runs.length === 1 ? `/runs/${first.run_id}` : buildHref("/runs", { launch: item.launch_id ?? undefined });
            const scores = item.runs.map((r) => r.headline?.score).filter((v): v is number => v != null);
            return (
              <AppLink key={item.key} href={href} className={h.feedRow}>
                <span className={h.feedTime} title={item.created_at}>
                  {timeOfDay(item.created_at)}
                  <span className={h.feedDay}>
                    <RelativeTime value={item.created_at} />
                  </span>
                </span>
                <StatusBadge status={item.status === "running" && first.stale ? "stale" : item.status} iconOnly />
                <span className={h.feedMain}>
                  <span>
                    <strong>{item.user ?? "someone"}</strong> <span className="muted">{verb}</span> {models.slice(0, 2).join(", ")}
                    {models.length > 2 && <span className="muted"> +{models.length - 2}</span>}
                    {item.runs.length > 1 && <span className="muted"> · {item.runs.length} runs</span>}
                    <span className="muted"> · {pluralize(item.n_tasks, "task")}</span>
                  </span>
                  {item.failure_reason && <span className={h.feedError}>{item.failure_reason}</span>}
                </span>
                <span className="hide-sm">{scores.length > 1 && <Sparkline values={scores} width={56} />}</span>
                <ArrowRight className={h.feedArrow} />
              </AppLink>
            );
          })}
          {activity.hasNextPage && (
            <div style={{ padding: 10, display: "flex", justifyContent: "center" }}>
              <Button size="sm" onClick={() => activity.fetchNextPage()} disabled={activity.isFetchingNextPage}>
                {activity.isFetchingNextPage ? "Loading…" : "Load more"}
              </Button>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}

function Continue() {
  const recents = useStore(recentStore);
  return (
    <Panel title="Continue where you left off" caption="Pages you visited recently on this device.">
      {!recents.length ? (
        <EmptyState icon={<History />} title="Nothing yet" compact>
          Runs, compare views and models you open show up here.
        </EmptyState>
      ) : (
        <div className={h.recentList}>
          {recents.slice(0, 7).map((r) => (
            <AppLink key={`${r.type}:${r.key}`} href={r.href} className={h.recentRow} keepBaseline={false}>
              <span className={h.recentType}>{r.type}</span>
              <span className="truncate">{r.label}</span>
              <span className="t-caption nowrap" style={{ marginLeft: "auto" }}>
                <RelativeTime value={new Date(r.at).toISOString()} />
              </span>
            </AppLink>
          ))}
        </div>
      )}
    </Panel>
  );
}

function Kpis() {
  const stats = useStatsSummary();
  const d = stats.data;
  const tile = (label: string, value: number | undefined, series: number[]) => (
    <Stat
      label={`${label} · 7d`}
      value={
        stats.isLoading ? (
          <Skeleton width={60} height={22} />
        ) : (
          <span style={{ fontFamily: "var(--font-sans)", fontWeight: 200, fontSize: "clamp(20px, 6vw, 30px)" }}>{formatCompact(value ?? 0)}</span>
        )
      }
      extra={<div style={{ marginTop: 2 }}>{series.length > 1 && <Sparkline values={series} width={120} height={20} fill fluid />}</div>}
      title={value != null ? formatCount(value) : undefined}
    />
  );
  return (
    <div className={h.kpis}>
      {tile("Runs", d?.runs, d?.daily.map((x) => x.runs) ?? [])}
      {tile("Models", d?.models, d?.daily.map((x) => x.models) ?? [])}
      {tile("Instances scored", d?.instances, d?.daily.map((x) => x.instances) ?? [])}
    </div>
  );
}

export function HomePage() {
  const me = useMe();
  return (
    <div className="page">
      <div className={h.top}>
        <div>
          <h1 className="t-title">
            {greeting()}
            {me.data ? `, ${me.data.username}` : ""}
          </h1>
          <p className="muted" style={{ marginTop: 2 }}>
            Evaluation results from olmo-eval runs.
          </p>
        </div>
        <Kpis />
      </div>
      <div className="grid-12">
        <div className="span-7">
          <RecentRuns />
        </div>
        <div className="span-5">
          <ActiveGroups />
        </div>
        <div className="span-8">
          <OrgActivity />
        </div>
        <div className="span-4">
          <Continue />
        </div>
      </div>
    </div>
  );
}
