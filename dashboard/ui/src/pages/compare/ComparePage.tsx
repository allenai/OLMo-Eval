import type { SubjectKey } from "@contract/api-types";
import * as Popover from "@radix-ui/react-popover";
import { ArrowRightLeft, GripVertical, Info, Plus, Star, X } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { useGroupDetail, useGroups, useSearch, useSuites } from "@/api/hooks/catalog";
import { useMatrix, useResolveSubjects } from "@/api/hooks/compare";
import { useRunsList } from "@/api/hooks/runs";
import { PageHeader } from "@/components/PageHeader";
import { Banner, Button, Checkbox, cx, EmptyState, ErrorPanel, ModelDot, SearchInput, Segmented, Select, Skeleton, Tabs, Tip, uiStyles as ui } from "@/components/primitives";
import { formatDelta, metricLabel } from "@/lib/format";
import { runtimeMetricMode } from "@/lib/runtime";
import { useHotkeys } from "@/lib/keyboard";
import { parseSubjects } from "@/lib/subjects";
import { oneOf, parseBool, parseNumber } from "@/lib/url";
import { useSubjectSlots } from "@/state/colors";
import { useBaseline, useSearchParams } from "@/state/nav";
import { pushRecent, trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { HeatmapView } from "./HeatmapView";
import c from "./compare.module.css";
import { type CompareCtx, type CompareSearch, type CompareView, distinctLabels, VIEWS } from "./types";
import { pressable } from "@/lib/pressable";

const PairwiseView = lazy(() => import("./PairwiseView").then((m) => ({ default: m.PairwiseView })));
const ScatterView = lazy(() => import("./ScatterView").then((m) => ({ default: m.ScatterView })));
const ProfilesView = lazy(() => import("./ProfilesView").then((m) => ({ default: m.ProfilesView })));
const ProgressionView = lazy(() => import("./ProgressionView").then((m) => ({ default: m.ProgressionView })));
const DisagreeView = lazy(() => import("./DisagreeView").then((m) => ({ default: m.DisagreeView })));

function AddSubject({ onAdd }: { onAdd: (key: SubjectKey, label: string) => void }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const search = useSearch(q);
  const tray = useStore(trayStore);
  const results = (search.data?.groups ?? []).flatMap((g) => g.items).filter((i) => i.subject);
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button type="button" className={ui.addChip}>
          <Plus /> Add
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={ui.popover} align="start" sideOffset={4} collisionPadding={8}>
          <div className={ui.facet} style={{ width: 340 }}>
            <div className={ui.facetHead}>
              <span className="t-overline">Add a run or model</span>
              <SearchInput autoFocus placeholder="Search runs and models" value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            <div className={ui.facetList}>
              {!q &&
                tray.map((t) => (
                  <div key={t.key} className={ui.facetItem} {...pressable(() => (onAdd(t.key, t.label), setOpen(false)))}>
                    <ArrowRightLeft size={13} />
                    <span className={ui.facetValue}>{t.label}</span>
                    <span className={ui.facetCount}>tray</span>
                  </div>
                ))}
              {!q && !tray.length && <div className="t-caption" style={{ padding: 8 }}>Type to search runs and models.</div>}
              {results.map((r) => (
                <div key={`${r.type}:${r.key}`} className={ui.facetItem} {...pressable(() => (onAdd(r.subject!, r.label), setOpen(false)))}>
                  <span className={ui.facetValue}>
                    {r.label}
                    <span className="t-caption" style={{ display: "block" }}>
                      {r.secondary}
                    </span>
                  </span>
                  <span className={ui.facetCount}>{r.type}</span>
                </div>
              ))}
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function useSubjectKeys(search: CompareSearch, baseline: SubjectKey | undefined) {
  const explicit = parseSubjects(search.subjects);
  const merge = search.merge === "none" ? "none" : "latest";
  const group = useGroupDetail(explicit.length ? "" : (search.group ?? ""));
  const groupRuns = useRunsList({ group: search.group, limit: 100, sort: "step" }, { enabled: !explicit.length && !!search.group && merge === "none" });
  let keys: SubjectKey[] = explicit;
  if (!explicit.length && search.group) {
    keys =
      merge === "none"
        ? (groupRuns.data?.items ?? []).filter((r) => r.num_tasks > 0).map((r) => `r:${r.run_id}` as SubjectKey)
        : (group.data?.subjects ?? []).map((sub) => sub.key);
  }
  if (baseline && keys.length && !keys.includes(baseline)) keys = [baseline, ...keys];
  return { keys, loading: !explicit.length && !!search.group && (group.isLoading || groupRuns.isLoading) };
}

export function ComparePage({ presetGroup }: { presetGroup?: string } = {}) {
  const [rawSearch, setSearch] = useSearchParams<CompareSearch>();
  const search = useMemo(() => (presetGroup ? { ...rawSearch, group: presetGroup } : rawSearch), [rawSearch, presetGroup]);
  const [baseline, setBaseline] = useBaseline();
  const { keys: requestedKeys, loading } = useSubjectKeys(search, baseline);
  const resolved = useResolveSubjects(requestedKeys, search.group);
  const resolvedFresh = !!resolved.data && !resolved.isPlaceholderData;
  const missingKey = resolvedFresh ? resolved.data.missing.join(",") : "";
  const suites = useSuites();
  const groups = useGroups({ limit: 100 });
  const view = oneOf<CompareView>(search.view, VIEWS, "heatmap");
  const alpha = parseNumber(search.alpha, 0.05);
  const shared = parseBool(search.shared, false);
  const scope = search.scope ?? "all";
  const metric = search.metric ?? "primary";
  const infoByKey = useMemo(() => new Map((resolved.data?.items ?? []).map((i) => [i.key, i])), [resolved.data]);
  const explicitKey = search.subjects ?? "";
  const keys = useMemo(() => {
    // Unknown subjects (a deleted run in a shared link, a stale baseline) are left out so the
    // rest of the page still loads.
    const missing = new Set(missingKey.split(",").filter(Boolean));
    let out = requestedKeys.filter((k) => !missing.has(k));
    // A run baseline prepended to a group's model subjects duplicates the model subject it
    // belongs to, so that model subject is dropped.
    if (baseline?.startsWith("r:") && !parseSubjects(explicitKey).length) {
      const runId = baseline.slice(2);
      out = out.filter((k) => k === baseline || !infoByKey.get(k)?.run_ids.includes(runId) || infoByKey.get(k)?.kind !== "model");
    }
    return out;
  }, [requestedKeys, missingKey, baseline, explicitKey, infoByKey]);
  const missingKeys = missingKey ? missingKey.split(",") : [];
  const removeMissing = () => {
    const explicit = parseSubjects(search.subjects).filter((k) => !missingKeys.includes(k));
    setSearch({
      subjects: explicit.join(",") || undefined,
      baseline: baseline && missingKeys.includes(baseline) ? undefined : baseline,
    });
  };
  const subjects = keys.map((k) => infoByKey.get(k)).filter((x): x is NonNullable<typeof x> => !!x);
  const slots = useSubjectSlots(keys.filter((k) => k !== baseline));
  const matrixBody = useMemo(
    () => ({ subjects: keys, group: search.group ?? null, scope, metric, alpha, baseline: baseline ?? null, shared_only: shared }),
    [keys, search.group, scope, metric, alpha, baseline, shared],
  );
  const matrix = useMatrix(matrixBody, keys.length > 0 && (resolvedFresh || resolved.isError));
  const [drag, setDrag] = useState<number | null>(null);
  // Phones show the first three subjects until the list is expanded.
  const [allSubjects, setAllSubjects] = useState(false);

  const subjectSignature = keys.join(",");
  const subjectCount = keys.length;
  useEffect(() => {
    if (subjectCount < 2) return;
    const label = search.group ? `Compare ${search.group}` : `Compare ${subjectCount} subjects`;
    pushRecent({ type: "compare", key: window.location.search, label, href: `/compare${window.location.search}` });
  }, [subjectSignature, subjectCount, search.group]);

  useHotkeys(Object.fromEntries(VIEWS.map((v, i) => [String(i + 1), () => setSearch({ view: v === "heatmap" ? undefined : v })])));

  const setSubjects = (next: SubjectKey[]) => setSearch({ subjects: next.join(",") || undefined, group: search.group });
  const labelOf = useMemo(() => {
    const labels = distinctLabels(keys, infoByKey);
    return (key: string) => labels.get(key) ?? key;
  }, [keys, infoByKey]);
  const explicitList = keys.filter((k) => k !== baseline || parseSubjects(search.subjects).includes(k));

  const coverage = matrix.data?.coverage;
  const metricOptions = useMemo(() => {
    const set = new Set<string>();
    for (const row of matrix.data?.rows ?? []) if (row.metric_key && !runtimeMetricMode(row.metric_key)) set.add(row.metric_key);
    return [
      { value: "primary", label: "primary (per task)", group: "Score" },
      ...[...set].sort().map((m) => ({ value: m, label: metricLabel(m), group: "Score" })),
      { value: "runtime:inference", label: "Runtime: inference", hint: "seconds", group: "Runtime (lower is better)" },
      { value: "runtime:with_startup", label: "Runtime: with startup", hint: "seconds", group: "Runtime (lower is better)" },
    ];
  }, [matrix.data]);

  const runtimeMode = runtimeMetricMode(metric);
  // Runtime has no instance-level pairing and no score scale, so only the heatmap shows it.
  const runtimeBlocked = runtimeMode != null && view !== "heatmap";
  const ctx: CompareCtx = { keys, subjects, baseline, group: search.group, scope, metric, alpha, shared, matrix, slots, search, setSearch, labelOf };

  const scopeOptions = [
    { value: "all", label: "All tasks", group: "Scope" },
    { value: "shared", label: "Shared tasks only", group: "Scope" },
    ...(suites.data?.items ?? []).map((su) => ({ value: `suite:${su.suite_name}`, label: su.suite_name, hint: `${su.n_tasks} tasks`, group: "Suites" })),
  ];
  if (scope.startsWith("task:")) scopeOptions.push({ value: scope, label: scope.slice(5), group: "Task" });

  return (
    <div className={presetGroup ? "col" : "page"} style={presetGroup ? { gap: 16 } : undefined}>
      <PageHeader
        title={presetGroup ? <span className="t-caption">Preset: every model in this group, latest result per task</span> : "Compare"}
        titleExtra={!presetGroup && search.group && <span className="muted" style={{ fontSize: 14, marginTop: 4 }}>group {search.group}</span>}
        actions={
          <>
            {!presetGroup && <Select
              size="sm"
              label="Group"
              value={search.group ?? ""}
              onChange={(v) => setSearch({ group: v || undefined, subjects: undefined })}
              options={[{ value: "", label: "none" }, ...(groups.data?.items ?? []).map((g) => ({ value: g.name, label: g.name, hint: `${g.n_models}` }))]}
              width={250}
            />}
            <Segmented
              value={search.merge === "none" ? "none" : "latest"}
              onChange={(v) => setSearch({ merge: v === "latest" ? undefined : v })}
              label="Merge"
              options={[
                { value: "latest", label: "Latest per model", title: "Model subjects merge the most recent result per task across runs" },
                { value: "none", label: "Runs as-is" },
              ]}
            />
          </>
        }
      />
      <div className={c.scope}>
        <div className={cx(c.subjects, !allSubjects && c.subjectsCollapsed)}>
          <span className="t-overline">Subjects</span>
          {keys.map((k, i) => {
            const info = infoByKey.get(k);
            const isBase = k === baseline;
            const move = (from: number, to: number) => {
              if (from === to || to < 0 || to >= keys.length || keys[to] === baseline) return;
              const next = [...keys];
              const [m] = next.splice(from, 1);
              next.splice(to, 0, m);
              setSubjects(next.filter((x) => x !== baseline || parseSubjects(search.subjects).includes(x)));
            };
            return (
              <span
                key={k}
                className={cx(c.subject, isBase && c.subjectBase, drag === i && c.dragging)}
                data-overflow={i >= 3 || undefined}
                draggable={!isBase}
                tabIndex={isBase ? undefined : 0}
                aria-label={isBase ? undefined : `${info?.label ?? k}, column ${i + 1} of ${keys.length}. Alt+Left or Alt+Right moves it.`}
                onKeyDown={(e) => {
                  if (isBase || !e.altKey || e.target !== e.currentTarget) return;
                  const to = e.key === "ArrowLeft" ? i - 1 : e.key === "ArrowRight" ? i + 1 : null;
                  if (to == null) return;
                  e.preventDefault();
                  move(i, to);
                }}
                onDragStart={() => setDrag(i)}
                onDragOver={(e) => e.preventDefault()}
                onDrop={() => {
                  if (drag == null || drag === i) return;
                  move(drag, i);
                  setDrag(null);
                }}
                title={info?.label ?? k}
              >
                {!isBase && <GripVertical size={12} className={c.grip} />}
                <ModelDot slot={slots[k]} baseline={isBase} />
                <span className={c.subjectLabel}>{info ? labelOf(k) : <Skeleton width={90} height={10} />}</span>
                {isBase && <span className={ui.badgeBase + " " + ui.badge}>BASE</span>}
                <button type="button" className={ui.chipRemove} aria-label={isBase ? "Clear baseline" : "Set as baseline"} title={isBase ? "Clear baseline" : "Set as baseline"} onClick={() => setBaseline(isBase ? null : k, info?.label)}>
                  <Star fill={isBase ? "currentColor" : "none"} />
                </button>
                <button
                  type="button"
                  className={ui.chipRemove}
                  aria-label="Remove"
                  onClick={() => {
                    if (isBase) setBaseline(null);
                    else setSubjects(explicitList.filter((x) => x !== k));
                  }}
                >
                  <X />
                </button>
              </span>
            );
          })}
          {keys.length > 3 && (
            <button type="button" className={c.moreSubjects} onClick={() => setAllSubjects((v) => !v)} aria-expanded={allSubjects}>
              {allSubjects ? "Show fewer" : `+${keys.length - 3} more`}
            </button>
          )}
          <AddSubject onAdd={(key) => setSubjects([...explicitList, key].filter((v, i, a) => a.indexOf(v) === i))} />
          {missingKeys.length ? (
            <span className="t-caption" title={missingKeys.join(", ")}>
              {missingKeys.length} unknown subject{missingKeys.length === 1 ? "" : "s"} ignored ·{" "}
              <button type="button" className="link" onClick={removeMissing}>
                remove from link
              </button>
            </span>
          ) : null}
        </div>
        <div className={c.controls}>
          <Select size="sm" label="Scope" value={scope} onChange={(v) => setSearch({ scope: v === "all" ? undefined : v })} options={scopeOptions} width={240} />
          <Select size="sm" label="Metric" value={metric} onChange={(v) => setSearch({ metric: v === "primary" ? undefined : v })} options={metricOptions} width={210} />
          {!runtimeMode && (<>
          <span className="row" style={{ gap: 6 }}>
            <span className="t-caption">alpha</span>
            <Segmented
              value={String(alpha)}
              onChange={(v) => setSearch({ alpha: v === "0.05" ? undefined : v })}
              label="Significance level"
              options={[
                { value: "0.01", label: "0.01" },
                { value: "0.05", label: "0.05" },
                { value: "0.1", label: "0.1" },
              ]}
            />
          </span>
          <Checkbox checked={shared} onChange={(on) => setSearch({ shared: on ? "1" : undefined })} label="Shared instances only" />
          </>)}
          <span className="spacer" />
          {coverage && (
            <span className={c.coverage}>
              <span className="mono">{coverage.n_subjects}</span> subjects × <span className="mono">{coverage.n_tasks}</span> tasks ·{" "}
              <span className="mono">{coverage.complete}</span> complete
              {coverage.missing > 0 && (
                <>
                  {" · "}
                  <span className="mono" style={{ color: "var(--warn-fg)" }}>{coverage.missing}</span> missing
                </>
              )}
              {coverage.failed > 0 && (
                <>
                  {" · "}
                  <span className="mono" style={{ color: "var(--danger-fg)" }}>{coverage.failed}</span> failed
                </>
              )}
              {coverage.hash_mismatch > 0 && (
                <>
                  {" · "}
                  <span className="mono">{coverage.hash_mismatch}</span> config mismatches
                </>
              )}
            </span>
          )}
          {matrix.data?.mde80 != null && (
            <Tip
              content={
                <span style={{ maxWidth: 280, display: "block" }}>
                  Minimum detectable effect at 80% power, from the median paired standard error across cells. Deltas smaller than this are unlikely to reach significance at alpha {alpha}.
                </span>
              }
            >
              <span className={c.coverage}>
                MDE80 <strong className="mono">{formatDelta(matrix.data.mde80, "percent").replace("+", "")} pp</strong> <Info size={12} />
              </span>
            </Tip>
          )}
        </div>
      </div>
      {loading || (requestedKeys.length > 0 && resolved.isLoading) ? (
        <Skeleton height={420} />
      ) : keys.length < 2 ? (
        <EmptyState
          icon={<ArrowRightLeft />}
          title="Add at least two runs or models to compare"
          actions={
            <>
              {trayStore.get().length > 0 && (
                <Button variant="primary" onClick={() => setSubjects(trayStore.get().map((t) => t.key))}>
                  Use the compare tray ({trayStore.get().length})
                </Button>
              )}
              <Button onClick={() => setSearch({ group: groups.data?.items[0]?.name })} disabled={!groups.data?.items.length}>
                Try the most recent group
              </Button>
            </>
          }
        >
          Pick a group above, use Add, or select rows on the Runs page with <code>x</code>.
        </EmptyState>
      ) : (
        <>
          <Tabs
            label="Compare views"
            value={view}
            onChange={(v) => setSearch({ view: v === "heatmap" ? undefined : v })}
            showKeys
            tabs={[
              { value: "heatmap", label: "Heatmap" },
              { value: "pairwise", label: "Pairwise" },
              { value: "scatter", label: "Scatter" },
              { value: "profiles", label: "Profiles" },
              { value: "progression", label: "Progression" },
              { value: "disagree", label: "Disagreements" },
            ]}
          />
          {coverage && coverage.missing > 0 && view === "heatmap" && (
            <Banner>
              {coverage.missing} subject–task cells have no result; they show as hatched cells and are never silently dropped.
            </Banner>
          )}
          {matrix.error ? (
            <ErrorPanel error={matrix.error} onRetry={() => matrix.refetch()} />
          ) : (
            <Suspense fallback={<Skeleton height={400} />}>
              {runtimeBlocked && (
                <EmptyState
                  title={view === "pairwise" || view === "disagree" ? "Not available for runtime" : "This view shows scores only"}
                  actions={
                    <>
                      <Button variant="primary" onClick={() => setSearch({ view: undefined })}>
                        Show runtime in the heatmap
                      </Button>
                      <Button onClick={() => setSearch({ metric: undefined })}>Back to scores</Button>
                    </>
                  }
                >
                  {view === "pairwise" || view === "disagree"
                    ? "Runtime is one number per task result, with no per-instance values to pair, so wins, losses and disagreements cannot be computed."
                    : "Runtime is in seconds, not on a score scale, so this view cannot show it."}
                </EmptyState>
              )}
              {view === "heatmap" && <HeatmapView {...ctx} />}
              {view === "pairwise" && !runtimeBlocked && <PairwiseView {...ctx} />}
              {view === "scatter" && !runtimeBlocked && <ScatterView {...ctx} />}
              {view === "profiles" && !runtimeBlocked && <ProfilesView {...ctx} />}
              {view === "progression" && !runtimeBlocked && <ProgressionView {...ctx} />}
              {view === "disagree" && !runtimeBlocked && <DisagreeView {...ctx} />}
            </Suspense>
          )}
        </>
      )}
    </div>
  );
}

