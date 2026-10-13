import type { RunDetail, SubjectInfo, SubjectKey } from "@contract/api-types";
import * as Popover from "@radix-ui/react-popover";
import { useParams } from "@tanstack/react-router";
import { ArrowRightLeft, Braces, Copy, ExternalLink, Link2, MoreHorizontal, Star, Tag, Terminal } from "lucide-react";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { useResolveSubjects } from "@/api/hooks/compare";
import { useBaselineSuggestions, usePatchRun, useRun, useRunSuites, useTaskResults } from "@/api/hooks/runs";
import { AppLink, buildHref } from "@/components/AppLink";
import { SubjectLabel } from "@/components/cells";
import { Badge, Button, CopyText, cx, displayStatus, ErrorPanel, LinkOut, Menu, ModelDot, Skeleton, StatusBadge, Tabs, Tip, uiStyles as ui } from "@/components/primitives";
import { openPalette } from "@/components/shell/registry";
import { toast } from "@/components/toast";
import { copyText } from "@/lib/csv";
import { formatCount, formatDuration, formatRuntime, shortHash } from "@/lib/format";
import { useHotkeys } from "@/lib/keyboard";
import { modelLabel, runSubject } from "@/lib/subjects";
import { absoluteLocal } from "@/lib/time";
import { oneOf } from "@/lib/url";
import { currentSubjectStore, useBaseline, useSearchParams } from "@/state/nav";
import { addToTray, pushRecent } from "@/state/prefs";
import { NotFoundPage } from "../NotFoundPage";
import { OverviewTab } from "./OverviewTab";
import s from "./run.module.css";
import type { RunSearch } from "./types";

const TasksTab = lazy(() => import("./TasksTab").then((m) => ({ default: m.TasksTab })));
const InstancesTab = lazy(() => import("./InstancesTab").then((m) => ({ default: m.InstancesTab })));
const InferenceTab = lazy(() => import("./InferenceTab").then((m) => ({ default: m.InferenceTab })));
const ConfigTab = lazy(() => import("./ConfigTab").then((m) => ({ default: m.ConfigTab })));
const ArtifactsTab = lazy(() => import("./ArtifactsTab").then((m) => ({ default: m.ArtifactsTab })));
const RawTab = lazy(() => import("./RawTab").then((m) => ({ default: m.RawTab })));

export const RUN_TABS = ["overview", "tasks", "instances", "inference", "config", "artifacts", "raw"] as const;
export type RunTab = (typeof RUN_TABS)[number];

function gpuLabel(run: RunDetail): string | null {
  const count = run.environment.gpu_count ?? run.beaker?.gpu_count;
  const type = run.environment.gpu_type?.replace(/^NVIDIA\s+/, "").replace(/\s+80GB.*$/, "");
  if (!count && !type) return null;
  return `${count ?? "?"}× ${type ?? "GPU"}`;
}

function TagEditor({ run }: { run: RunDetail }) {
  const patch = usePatchRun(run.run_id);
  const [draft, setDraft] = useState("");
  return (
    <div style={{ padding: 10, width: 280, display: "flex", flexDirection: "column", gap: 8 }}>
      <span className="t-overline">Tags</span>
      <div className="row-wrap" style={{ gap: 4 }}>
        {run.tags.length === 0 && <span className="t-caption">No tags</span>}
        {run.tags.map((t) => (
          <span key={t} className={ui.chip} style={{ height: 22 }}>
            <span>{t}</span>
            <button type="button" className={ui.chipRemove} aria-label={`Remove ${t}`} onClick={() => patch.mutate({ tags: run.tags.filter((x) => x !== t) })}>
              ×
            </button>
          </span>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const tag = draft.trim().replace(/\s+/g, "-");
          if (tag && !run.tags.includes(tag)) patch.mutate({ tags: [...run.tags, tag] });
          setDraft("");
        }}
        className="row"
      >
        <input className={ui.input} style={{ flex: 1 }} placeholder="Add a tag" value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus />
        <Button size="sm" type="submit">
          Add
        </Button>
      </form>
    </div>
  );
}

function RunHeader({ run, baseline }: { run: RunDetail; baseline?: SubjectKey }) {
  const [, setBaseline] = useBaseline();
  const subject = runSubject(run.run_id);
  const isBase = baseline === subject;
  const gpu = gpuLabel(run);
  const vllm = run.environment.packages.vllm;
  return (
    <div className={s.head}>
      <nav className="t-caption" aria-label="Breadcrumb">
        <AppLink href="/runs" className="link" style={{ textDecorationColor: "transparent" }}>
          Runs
        </AppLink>
        <span className="faint"> / </span>
        {run.experiment_group ? (
          <AppLink href={`/groups/${encodeURIComponent(run.experiment_group)}`} className="link" style={{ textDecorationColor: "transparent" }}>
            {run.experiment_group}
          </AppLink>
        ) : (
          <span>ungrouped</span>
        )}
        <span className="faint"> / </span>
        <span className="mono">{run.run_id}</span>
      </nav>
      <div className={s.titleRow}>
        {isBase && <ModelDot baseline />}
        <h1 className={s.title}>
          <AppLink href={`/models/${encodeURIComponent(run.model.series)}`} title={`Model page for ${run.model.series}`}>
            {run.model.series_label}
          </AppLink>
        </h1>
        {run.model.step != null && <span className={s.stepChip}>step {formatCount(run.model.step)}</span>}
        <StatusBadge status={displayStatus(run)} />
        {isBase && <Badge tone="base">BASE</Badge>}
        {run.tags.map((t) => (
          <Badge key={t} tone="muted">
            {t}
          </Badge>
        ))}
        <span className="spacer" />
        <div className="row-wrap" style={{ gap: 6 }}>
          <Button size="sm" icon={<ArrowRightLeft />} onClick={() => (addToTray({ key: subject, label: modelLabel(run.model) }), toast("Added to compare"))}>
            Compare
          </Button>
          <Button size="sm" icon={<Star fill={isBase ? "currentColor" : "none"} />} onClick={() => setBaseline(isBase ? null : subject, modelLabel(run.model))}>
            {isBase ? "Baseline" : "Set baseline"}
          </Button>
          <LinkOut href={run.links.beaker_experiment}>Beaker</LinkOut>
          <LinkOut href={run.links.gcs_console}>GCS</LinkOut>
          <Popover.Root>
            <Popover.Trigger asChild>
              <Button size="sm" variant="ghost" icon={<Tag />} aria-label="Edit tags" />
            </Popover.Trigger>
            <Popover.Portal>
              <Popover.Content className={ui.popover} align="end" sideOffset={4}>
                <TagEditor run={run} />
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>
          <Menu
            trigger={<Button size="sm" variant="ghost" icon={<MoreHorizontal />} aria-label="More actions" />}
            items={[
              { label: "Copy link to this view", icon: <Link2 />, hint: "y", onSelect: () => copyText(window.location.href).then((ok) => ok && toast("Link copied")) },
              { label: "Copy run ID", icon: <Copy />, onSelect: () => copyText(run.run_id).then(() => toast("Copied run ID")) },
              { label: "Copy gs:// prefix", icon: <Copy />, onSelect: () => copyText(run.gcs_prefix).then(() => toast("Copied GCS prefix")) },
              ...(run.reproduce_command
                ? [{ label: "Copy reproduce command", icon: <Terminal />, onSelect: () => copyText(run.reproduce_command!).then(() => toast("Copied command")) }]
                : []),
              { separator: true },
              { label: "Beaker experiment", icon: <ExternalLink />, href: run.links.beaker_experiment ?? undefined, disabled: !run.links.beaker_experiment },
              ...(run.beaker?.job_id ? [{ label: "Beaker job", icon: <ExternalLink />, href: `https://beaker.org/job/${run.beaker.job_id}` }] : []),
              { label: "Beaker result dataset", icon: <ExternalLink />, href: run.links.beaker_result_dataset ?? undefined, disabled: !run.links.beaker_result_dataset },
              { label: "Beaker workspace", icon: <ExternalLink />, href: run.links.beaker_workspace ?? undefined, disabled: !run.links.beaker_workspace },
              { separator: true },
              { label: "Open raw JSON", icon: <Braces />, onSelect: () => window.open(`/api/runs/${run.run_id}`, "_blank") },
            ]}
          />
        </div>
      </div>
      <div className={s.metaLine}>
        {run.model.family && (
          <span>
            <span className={s.metaKey}>family</span> {run.model.family}
          </span>
        )}
        <span>
          <span className={s.metaKey}>hash</span>
          <CopyText text={run.model.model_hash} display={<span className="mono">{shortHash(run.model.model_hash, 8)}</span>} />
        </span>
        <span>
          {run.model_detail.provider_kind}
          {vllm ? ` ${vllm}` : ""}
        </span>
        {gpu && <span>{gpu}</span>}
        {run.launch_id && (
          <span>
            <span className={s.metaKey}>launch</span>
            {run.siblings.length ? (
              <Popover.Root>
                <Popover.Trigger asChild>
                  <button type="button" className={ui.copy} style={{ textDecoration: "underline dotted", textUnderlineOffset: 3 }}>
                    <span className="mono">{run.launch_id}</span>
                  </button>
                </Popover.Trigger>
                <Popover.Portal>
                  <Popover.Content className={ui.popover} sideOffset={4} align="start" style={{ padding: 8, minWidth: 260 }}>
                    <div className="t-overline" style={{ marginBottom: 6 }}>
                      Other runs in this launch
                    </div>
                    {run.siblings.map((sib) => (
                      <AppLink key={sib.run_id} href={`/runs/${sib.run_id}`} className={ui.menuItem}>
                        <StatusBadge status={displayStatus(sib)} iconOnly />
                        <span className="truncate">{modelLabel(sib.model)}</span>
                        <span className="t-caption" style={{ marginLeft: "auto" }}>
                          {sib.num_tasks} tasks
                        </span>
                      </AppLink>
                    ))}
                  </Popover.Content>
                </Popover.Portal>
              </Popover.Root>
            ) : (
              <span className="mono">{run.launch_id}</span>
            )}
          </span>
        )}
      </div>
      <div className={s.metaLine}>
        <span>
          <span className={s.metaKey}>by</span> {run.author ?? run.uploaded_by}
        </span>
        <span title={run.created_at}>{absoluteLocal(run.finished_at ?? run.created_at)}</span>
        <span>
          <span className={s.metaKey}>took</span> {formatDuration(run.duration_seconds)}
          {(run.startup_seconds != null || run.processing_seconds != null) && (
            <Tip
              content={
                <span style={{ display: "block", maxWidth: 280 }}>
                  Startup is model load and server start. Processing runs from when every worker was ready until the last task finished. The total also covers result upload; Beaker queue time is never included.
                </span>
              }
            >
              <span className="muted" style={{ marginLeft: 6, textDecoration: "underline dotted", textUnderlineOffset: 3 }}>
                {run.startup_seconds != null && <>startup {formatRuntime(run.startup_seconds)}</>}
                {run.startup_seconds != null && run.processing_seconds != null && " · "}
                {run.processing_seconds != null && <>processing {formatRuntime(run.processing_seconds)}</>}
              </span>
            </Tip>
          )}
        </span>
        {run.git.commit && (
          <span>
            <span className={s.metaKey}>commit</span>
            <LinkOut href={run.links.github_commit}>
              <span className="mono">{shortHash(run.git.commit)}</span>
            </LinkOut>
            {run.git.branch && <span className="muted">({run.git.branch})</span>}
            {run.git.dirty && (
              <Tip content="The working tree had uncommitted changes when this run started.">
                <span className={s.dirty}>dirty</span>
              </Tip>
            )}
          </span>
        )}
        <span>
          <span className={s.metaKey}>{formatCount(run.num_instances)}</span> instances
        </span>
      </div>
      {run.notes && (
        <div className="t-caption" style={{ color: "var(--fg-2)", fontSize: 12.5 }}>
          <span className={s.metaKey}>Note:</span> {run.notes}
        </div>
      )}
    </div>
  );
}

const REASON_LABEL: Record<string, string> = {
  previous_checkpoint: "previous checkpoint",
  same_group: "same group",
  previous_run_same_tasks: "earlier run",
};

function BaselineBar({ run, baseline, baselineInfo }: { run: RunDetail; baseline?: SubjectKey; baselineInfo?: SubjectInfo | null }) {
  const suggestions = useBaselineSuggestions(run.run_id);
  const [, setBaseline] = useBaseline();
  const self = baseline === runSubject(run.run_id);
  return (
    <div className={s.baselineBar}>
      <span className={s.baselineBarLabel}>Compare against</span>
      {baseline && !self ? (
        <>
          <span className="row" style={{ gap: 6 }}>
            <SubjectLabel label={baselineInfo?.label ?? baseline} baseline />
          </span>
          {baselineInfo?.run && (
            <AppLink href={`/runs/${baselineInfo.run.run_id}`} className="t-caption link">
              open
            </AppLink>
          )}
          <Button size="sm" variant="ghost" onClick={() => openPalette("baseline")}>
            Change
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setBaseline(null)}>
            Clear
          </Button>
        </>
      ) : (
        <span className="muted">{self ? "This run is the baseline. Pick another to see deltas." : "No baseline. Pick one to see deltas, paired CIs and gained or lost instances."}</span>
      )}
      <span className="spacer" />
      <span className={cx(s.suggestions, baseline && !self && s.suggestionsSecondary)}>
        {(suggestions.data?.items ?? [])
          // Hide the current baseline, including its model-subject twin.
          .filter((sg) => sg.subject !== baseline && (!baselineInfo || sg.subject !== `m:${baselineInfo.model.model_id}`))
          .slice(0, 4)
          .map((sg) => (
            <button key={sg.subject} type="button" className={s.suggestion} onClick={() => setBaseline(sg.subject, sg.label)} title={`Use ${sg.label} as the baseline`}>
              <ModelDot baseline />
              <span>{sg.label}</span>
              <span className={s.suggestionReason}>{REASON_LABEL[sg.reason]}</span>
            </button>
          ))}
        {suggestions.isLoading && <Skeleton width={180} height={20} />}
      </span>
    </div>
  );
}

export function RunPage() {
  const { runId } = useParams({ strict: false }) as { runId: string };
  const [search, setSearch] = useSearchParams<RunSearch>();
  const [baseline] = useBaseline();
  const run = useRun(runId);
  const effectiveBaseline = baseline && baseline !== runSubject(runId) ? baseline : undefined;
  // Fetched in parallel with the run; a bad link costs two extra 404s, a good one saves a round trip.
  const tasks = useTaskResults(runId, effectiveBaseline);
  const suites = useRunSuites(runId, effectiveBaseline);
  const resolved = useResolveSubjects(effectiveBaseline ? [effectiveBaseline] : []);
  const baselineInfo = resolved.data?.items[0] ?? tasks.data?.baseline ?? null;
  const tab = oneOf<RunTab>(search.tab, RUN_TABS, "overview");
  const sentinel = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    if (!run.data) return;
    const label = modelLabel(run.data.model);
    currentSubjectStore.set({ key: runSubject(runId), label });
    pushRecent({ type: "run", key: runId, label, href: `/runs/${runId}`, secondary: run.data.experiment_group ?? undefined });
    return () => currentSubjectStore.set(null);
  }, [run.data, runId]);

  useEffect(() => {
    const el = sentinel.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => setStuck(!entry.isIntersecting), { rootMargin: "-49px 0px 0px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [run.data]);

  const setTab = (t: RunTab) => setSearch({ tab: t === "overview" ? undefined : t });
  useHotkeys(Object.fromEntries(RUN_TABS.map((t, i) => [String(i + 1), () => setTab(t)])));

  if (run.error) {
    const status = (run.error as { status?: number }).status;
    if (status === 404) return <NotFoundPage what={`Run ${runId}`} />;
    return (
      <div className="page">
        <ErrorPanel error={run.error} onRetry={() => run.refetch()} />
      </div>
    );
  }
  if (!run.data) {
    return (
      <div className="page">
        <Skeleton width={260} height={28} />
        <Skeleton width="60%" height={14} />
        <Skeleton width="40%" height={14} />
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 10, marginTop: 12 }}>
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} height={92} />
          ))}
        </div>
        <Skeleton height={360} />
      </div>
    );
  }
  const r = run.data;
  const nTasks = tasks.data?.items.length ?? r.num_tasks;
  const ctx = { run: r, baseline: effectiveBaseline, baselineInfo, tasks, suites, search, setSearch };
  return (
    <div className="page">
      <RunHeader run={r} baseline={baseline} />
      <BaselineBar run={r} baseline={baseline} baselineInfo={baselineInfo} />
      <div ref={sentinel} style={{ height: 1, marginBottom: -1 }} />
      <div className={`${s.tabsBar} ${stuck ? s.stuck : ""}`}>
        <Tabs
          label="Run sections"
          value={tab}
          onChange={setTab}
          showKeys
          tabs={[
            { value: "overview", label: "Overview" },
            { value: "tasks", label: "Tasks", count: nTasks },
            { value: "instances", label: "Instances" },
            { value: "inference", label: "Inference" },
            { value: "config", label: "Config" },
            { value: "artifacts", label: "Artifacts" },
            { value: "raw", label: "Raw" },
          ]}
        />
        <span className={s.stickyTitle}>
          {r.model.series_label}
          {r.model.step != null && <span className="mono muted">@{formatCount(r.model.step)}</span>}
          <StatusBadge status={displayStatus(r)} iconOnly />
        </span>
      </div>
      <Suspense fallback={<Skeleton height={300} />}>
        {tab === "overview" && <OverviewTab {...ctx} />}
        {tab === "tasks" && <TasksTab {...ctx} />}
        {tab === "instances" && <InstancesTab {...ctx} />}
        {tab === "inference" && <InferenceTab {...ctx} />}
        {tab === "config" && <ConfigTab {...ctx} />}
        {tab === "artifacts" && <ArtifactsTab {...ctx} />}
        {tab === "raw" && <RawTab {...ctx} />}
      </Suspense>
    </div>
  );
}

export function compareHref(subjects: string[]): string {
  return buildHref("/compare", { subjects: subjects.join(",") });
}
