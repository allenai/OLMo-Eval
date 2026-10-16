/**
 * Mock implementations of every /api endpoint over the synthetic world. Each function takes the
 * parsed query or body and returns a response typed by the contract.
 */
import type * as T from "@contract/api-types";
import {
  completionTokens,
  DAY,
  finishReason,
  getWorld,
  hasScoringError,
  indexOfNative,
  instanceData,
  instanceMetrics,
  ME,
  type MockRun,
  type MockTaskResult,
  nativeId,
  NOW,
  parentSuite,
  RESULTS_BUCKET,
  runLookup,
  SUITE_BY_NAME,
  SUITES,
  suiteLeaves,
  suiteScore,
  suitesFor,
  TASK_BY_NAME,
  TASKS,
  taskScore,
} from "./fixtures";
import { hexId, rngFor } from "./random";
import {
  combineDeltas,
  contingency,
  histogram,
  isCorrect,
  median,
  pairedDelta,
  pairedSummary,
  quantile,
  unpairedDelta,
  zFor,
} from "./stats";
import { instanceRecords, instanceText } from "./texts";

export class MockHttpError extends Error {
  status: number;
  code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export type Query = { get: (key: string) => string | null; getAll: (key: string) => string[] };

const nowIso = () => new Date(NOW).toISOString();
const visible = (r: MockRun) => r.summary.upload_state === "complete" || r.summary.status === "running";

function tr(id: number): MockTaskResult {
  const found = getWorld().taskResults.get(id);
  if (!found) throw new MockHttpError(404, "not_found", `Task result ${id} not found.`);
  return found;
}

function getRun(runId: string): MockRun {
  const run = getWorld().runById.get(runId);
  if (!run) throw new MockHttpError(404, "not_found", `Run ${runId} not found.`);
  return run;
}

function runTRs(run: MockRun): MockTaskResult[] {
  return run.taskResultIds.map(tr);
}

/** The contract's TaskRuntime for a mock task result. */
function taskRuntime(t: MockTaskResult): T.TaskRuntime {
  const run = getRun(t.runId);
  const rt = t.runtime;
  const recorded = rt.basis !== "not_recorded";
  return {
    inference_seconds: rt.inferenceS,
    basis: rt.basis,
    startup_seconds: run.startupS,
    with_startup_seconds: rt.inferenceS != null && run.startupS != null ? Math.round((rt.inferenceS + run.startupS) * 10) / 10 : null,
    token_share: rt.tokenShare,
    prompt_tokens_total: recorded ? rt.promptTokens : null,
    completion_tokens_total: recorded && rt.completionTokens ? rt.completionTokens : null,
    seconds_per_1k_instances: rt.inferenceS != null && t.n ? (rt.inferenceS / t.n) * 1000 : null,
    span_start_s: rt.spanStart,
    span_end_s: rt.spanEnd,
  };
}

function paginate<X>(items: X[], q: Query, defLimit = 50, maxLimit = 500) {
  const offset = Number(q.get("cursor") ?? 0) || 0;
  const limit = Math.min(maxLimit, Math.max(1, Number(q.get("limit") ?? defLimit) || defLimit));
  const page = items.slice(offset, offset + limit);
  const next = offset + limit < items.length ? String(offset + limit) : null;
  return { page, next_cursor: next, total: items.length };
}

function globMatch(value: string | null | undefined, pattern: string): boolean {
  if (!value) return false;
  const re = new RegExp(`^${pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*")}$`, "i");
  return re.test(value);
}

// ------------------------------------------------------------------------------------- subjects

interface Resolved {
  info: T.SubjectInfo;
  lookup: Map<string, MockTaskResult>;
}

function modelRef(run: MockRun): T.ModelRef {
  return run.summary.model;
}

function baseLabel(model: T.ModelRef): string {
  return model.step != null ? `${model.series_label} @ step ${model.step.toLocaleString("en-US")}` : model.series_label;
}

export function resolveSubject(key: string, group?: string | null): Resolved | null {
  const w = getWorld();
  if (key.startsWith("r:")) {
    const run = w.runById.get(key.slice(2));
    if (!run) return null;
    const lookup = new Map<string, MockTaskResult>();
    for (const t of runTRs(run)) lookup.set(t.task.name, t);
    return {
      info: {
        key,
        kind: "run",
        label: baseLabel(run.summary.model),
        model: modelRef(run),
        run: run.summary,
        run_ids: [run.summary.run_id],
        latest_at: run.summary.finished_at ?? run.summary.created_at,
      },
      lookup,
    };
  }
  if (key.startsWith("m:")) {
    const modelId = key.slice(2);
    const runs = w.runs
      .filter((r) => r.model.model_id === modelId && visible(r) && (!group || r.summary.experiment_group === group))
      .sort((a, b) => b.summary.created_at.localeCompare(a.summary.created_at));
    if (!runs.length) return null;
    const lookup = new Map<string, MockTaskResult>();
    const fallback = new Map<string, MockTaskResult>();
    const contributing = new Set<string>();
    for (const run of runs) {
      for (const t of runTRs(run)) {
        if (t.error) {
          if (!fallback.has(t.task.name)) fallback.set(t.task.name, t);
        } else if (!lookup.has(t.task.name)) {
          lookup.set(t.task.name, t);
          contributing.add(run.summary.run_id);
        }
      }
    }
    for (const [name, t] of fallback) {
      if (!lookup.has(name)) {
        lookup.set(name, t);
        contributing.add(t.runId);
      }
    }
    return {
      info: {
        key,
        kind: "model",
        label: baseLabel(runs[0].summary.model),
        model: modelRef(runs[0]),
        run: null,
        run_ids: [...contributing],
        latest_at: runs[0].summary.finished_at ?? runs[0].summary.created_at,
      },
      lookup,
    };
  }
  return null;
}

function resolveMany(keys: string[], group?: string | null): Resolved[] {
  const out: Resolved[] = [];
  for (const key of keys) {
    const r = resolveSubject(key, group);
    if (!r) throw new MockHttpError(400, "unknown_subject", `Unknown subject ${key}.`);
    out.push(r);
  }
  // Disambiguate identical labels for run subjects.
  const counts = new Map<string, number>();
  out.forEach((r) => counts.set(r.info.label, (counts.get(r.info.label) ?? 0) + 1));
  out.forEach((r) => {
    if ((counts.get(r.info.label) ?? 0) > 1 && r.info.run) r.info.label = `${r.info.label} · ${r.info.run.experiment_name ?? r.info.run.run_id}`;
  });
  return out;
}

export function subjectsResolve(body: T.ResolveSubjectsRequest): T.ResolveSubjectsResponse {
  const items: T.SubjectInfo[] = [];
  const missing: string[] = [];
  for (const key of body.subjects) {
    const r = resolveSubject(key, body.group);
    if (r) items.push(r.info);
    else missing.push(key);
  }
  return { items, missing };
}

export function subjectTaskResult(q: Query): T.SubjectTaskResultResponse {
  const subject = q.get("subject") ?? "";
  const task = q.get("task") ?? "";
  const r = resolveSubject(subject, q.get("group"));
  if (!r) throw new MockHttpError(404, "not_found", `Unknown subject ${subject}.`);
  const t = r.lookup.get(task);
  return { subject, task_name: task, task_result_id: t?.id ?? null, run_id: t?.runId ?? null, task_hash: t?.taskHash ?? null };
}

// ------------------------------------------------------------------------------------- basics

export function me(): T.MeResponse {
  return { email: `${ME}@allenai.org`, username: ME, dev_mode: true };
}

export function health(): T.HealthResponse {
  return { status: "ok", mode: "dashboard", git_sha: null, skiff_env: "local", db: "ok" };
}

// ------------------------------------------------------------------------------------- runs list

function filterRuns(q: Query, skip?: string): MockRun[] {
  const w = getWorld();
  const all = (key: string) => (key === skip ? [] : q.getAll(key).flatMap((v) => v.split(",")).filter(Boolean));
  const one = (key: string) => (key === skip ? null : q.get(key));
  const models = all("model");
  const series = all("series");
  const families = all("family");
  const tasks = all("task");
  const suites = all("suite");
  const users = all("user").map((u) => (u === "me" ? ME : u));
  const groups = all("group");
  const launches = all("launch");
  const workspaces = all("workspace");
  const tags = all("tag");
  const notTags = all("not_tag");
  const statuses = all("status");
  const runIds = all("run_id");
  const after = one("after");
  const before = one("before");
  const date = one("date");
  const commit = one("commit");
  const stepMin = one("step_min");
  const stepMax = one("step_max");
  const hasFailures = one("has_failures") === "true";
  const text = one("q");
  const coverage = q.get("suite_coverage") ?? "full";
  const dateMs: Record<string, number> = { "24h": DAY, "7d": 7 * DAY, "30d": 30 * DAY, "90d": 90 * DAY };
  return w.runs.filter((run) => {
    const s = run.summary;
    if (!visible(run)) return false;
    if (models.length && !models.some((m) => globMatch(s.model.name, m))) return false;
    if (series.length && !series.includes(s.model.series)) return false;
    if (families.length && !families.includes(s.model.family ?? "")) return false;
    const names = new Set(runTRs(run).map((t) => t.task.name));
    if (tasks.length && !tasks.some((pattern) => [...names].some((n) => globMatch(n, pattern)))) return false;
    if (suites.length) {
      const ok = suites.some((suite) => {
        const leaves = suiteLeaves(suite);
        return coverage === "any" ? leaves.some((l) => names.has(l)) : leaves.length > 0 && leaves.every((l) => names.has(l));
      });
      if (!ok) return false;
    }
    if (users.length && !users.includes(s.author ?? "") && !users.includes(s.uploaded_by.split("@")[0])) return false;
    if (groups.length && !groups.includes(s.experiment_group ?? "")) return false;
    if (launches.length && !launches.includes(s.launch_id ?? "")) return false;
    if (workspaces.length && !workspaces.includes(s.beaker_workspace ?? "")) return false;
    if (tags.length && !tags.some((t) => s.tags.includes(t))) return false;
    if (notTags.length && notTags.some((t) => s.tags.includes(t))) return false;
    if (statuses.length) {
      const st = s.status === "running" ? "running" : s.upload_state === "uploading" ? "uploading" : s.status;
      if (!statuses.includes(st) && !statuses.includes(s.status)) return false;
    }
    if (runIds.length && !runIds.includes(s.run_id)) return false;
    if (after && s.created_at < new Date(after).toISOString()) return false;
    if (before && s.created_at >= new Date(before).toISOString()) return false;
    if (date && dateMs[date] && new Date(s.created_at).getTime() < NOW - dateMs[date]) return false;
    if (commit && !(s.git_commit ?? "").startsWith(commit)) return false;
    if (stepMin && (s.model.step == null || s.model.step < Number(stepMin))) return false;
    if (stepMax && (s.model.step == null || s.model.step > Number(stepMax))) return false;
    if (hasFailures && s.num_failed_tasks === 0) return false;
    if (text) {
      const hay = [s.model.name, s.experiment_name, s.experiment_group, s.author, ...s.tags, s.run_id].join(" ").toLowerCase();
      if (!text.toLowerCase().split(/\s+/).every((tok) => hay.includes(tok))) return false;
    }
    return true;
  });
}

function scoreColumns(cols: string[]): T.ScoreColumnMeta[] {
  return cols.slice(0, 20).map((key) => {
    const [kind, ...rest] = key.split(":");
    const name = rest.join(":");
    if (kind === "suite") return { key, kind: "suite", name, display_format: "percent", higher_is_better: true };
    const task = TASK_BY_NAME.get(name);
    const meta = task?.meta[task.primary];
    return {
      key,
      kind: "task",
      name,
      display_format: meta?.display_format ?? "percent",
      higher_is_better: meta?.higher_is_better ?? true,
    };
  });
}

function columnValue(run: MockRun, col: T.ScoreColumnMeta): { score: number | null; stderr: number | null; n: number | null; id: number | null } {
  const lookup = runLookup(run);
  if (col.kind === "suite") {
    const s = suiteScore(col.name, lookup);
    if (!run.taskResultIds.length || s.missing.length === suiteLeaves(col.name).length) return { score: null, stderr: null, n: null, id: null };
    return { score: s.score, stderr: s.stderr, n: s.nInstances, id: null };
  }
  const t = lookup(col.name);
  if (!t) return { score: null, stderr: null, n: null, id: null };
  const s = taskScore(t);
  return { score: s.score, stderr: s.stderr, n: s.n, id: t.id };
}

export function runsList(q: Query): T.RunsListResponse {
  let runs = filterRuns(q);
  const cols = scoreColumns((q.get("cols") ?? "").split(",").filter(Boolean));
  const baselineKey = q.get("baseline");
  const baseline = baselineKey ? resolveSubject(baselineKey) : null;
  const sort = q.get("sort") ?? "-created_at";
  const desc = sort.startsWith("-");
  const key = desc ? sort.slice(1) : sort;
  const getter = (r: MockRun): string | number | null => {
    const s = r.summary;
    switch (key) {
      case "created_at":
        return s.created_at;
      case "model":
        return s.model.name.toLowerCase();
      case "step":
        return s.model.step;
      case "group":
        return s.experiment_group;
      case "user":
        return s.author;
      case "duration":
        return s.duration_seconds;
      case "num_tasks":
        return s.num_tasks;
      default:
        if (key.startsWith("score:")) {
          const col = cols.find((c) => c.key === key.slice(6)) ?? scoreColumns([key.slice(6)])[0];
          return columnValue(r, col).score;
        }
        return s.created_at;
    }
  };
  runs = [...runs].sort((a, b) => {
    const va = getter(a);
    const vb = getter(b);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const cmp = va < vb ? -1 : va > vb ? 1 : 0;
    return desc ? -cmp : cmp;
  });
  const { page, next_cursor, total } = paginate(runs, q);
  const items: T.RunRow[] = page.map((run) => {
    const scores: Record<string, T.ScoreColumnValue> = {};
    for (const col of cols) {
      const v = columnValue(run, col);
      let delta: T.DeltaStats | null = null;
      if (baseline && v.score != null) {
        let b: { score: number | null; stderr: number | null } = { score: null, stderr: null };
        if (col.kind === "suite") {
          const s = suiteScore(col.name, (t) => baseline.lookup.get(t));
          b = { score: s.score, stderr: s.stderr };
        } else {
          const bt = baseline.lookup.get(col.name);
          if (bt) b = taskScore(bt);
        }
        delta = unpairedDelta(v, b, col.higher_is_better);
      }
      scores[col.key] = { score: v.score, stderr: v.stderr, n: v.n, task_result_id: v.id, delta };
    }
    return { ...run.summary, scores };
  });
  return { items, next_cursor, total, columns: cols, baseline: baseline?.info ?? null };
}

export function runsFacets(q: Query): T.RunsFacetsResponse {
  const count = (skip: string, getter: (r: MockRun) => (string | null)[]) => {
    const counts = new Map<string, number>();
    for (const run of filterRuns(q, skip)) {
      for (const v of getter(run)) if (v) counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 50)
      .map(([value, c]) => ({ value, count: c }));
  };
  return {
    user: count("user", (r) => [r.summary.author]),
    family: count("family", (r) => [r.summary.model.family]),
    group: count("group", (r) => [r.summary.experiment_group]),
    workspace: count("workspace", (r) => [r.summary.beaker_workspace]),
    tag: count("tag", (r) => r.summary.tags),
    status: count("status", (r) => [
      r.summary.status === "running" ? "running" : r.summary.upload_state === "uploading" ? "uploading" : r.summary.status,
    ]),
    model: count("model", (r) => [r.summary.model.name]),
  };
}

export function bulkTag(body: T.BulkTagRequest): T.BulkTagResponse {
  let updated = 0;
  for (const id of body.run_ids) {
    const run = getWorld().runById.get(id);
    if (!run) continue;
    const tags = new Set(run.summary.tags);
    body.add.forEach((t) => tags.add(t));
    body.remove.forEach((t) => tags.delete(t));
    run.summary.tags = [...tags];
    updated += 1;
  }
  return { updated };
}

// ------------------------------------------------------------------------------------- run detail

export function runDetail(runId: string): T.RunDetail {
  const run = getRun(runId);
  const names = new Set(runTRs(run).map((t) => t.task.name));
  const siblings = getWorld()
    .runs.filter((r) => r.summary.launch_id === run.summary.launch_id && r.summary.run_id !== runId)
    .map((r) => r.summary);
  return {
    ...run.summary,
    model_detail: run.model,
    git: run.git,
    beaker: run.beaker,
    environment: run.environment,
    olmo_eval_version: "0.9.3",
    argv: run.argv,
    reproduce_command: run.argv.map((a) => (/[\s"']/.test(a) ? `'${a}'` : a)).join(" "),
    task_specs: run.taskSpecs,
    output_dir: "/results",
    notes: run.notes,
    errors: run.errors,
    provider_init_seconds: { [`${run.model.series_label.slice(0, 18)}...-w0`]: 74.2 + (run.model.step ?? 0) / 1000 },
    startup_seconds: run.startupS,
    processing_started_at: run.processingStartedAt,
    processing_seconds: run.processingS,
    suites_used: suitesFor(names),
    siblings,
    updated_at: run.updatedAt,
  };
}

export function patchRun(runId: string, body: T.RunPatchRequest): T.RunDetail {
  const run = getRun(runId);
  if (body.tags) run.summary.tags = body.tags;
  if (body.notes !== undefined) run.notes = body.notes;
  return runDetail(runId);
}

function taskRow(t: MockTaskResult, baseline: Resolved | null, alpha: number, threshold: number): T.TaskResultRow {
  const s = taskScore(t);
  const data = t.error ? null : instanceData(t);
  let tokens = 0;
  let tokenCount = 0;
  const finish: Record<string, number> = {};
  if (data) {
    for (let i = 0; i < t.n; i++) {
      const c = completionTokens(t, i);
      if (c != null) {
        tokens += c;
        tokenCount += 1;
      }
      const f = finishReason(t, i);
      if (f) finish[f] = (finish[f] ?? 0) + 1;
    }
  }
  const bt = baseline?.lookup.get(t.task.name);
  let baselineCell: T.BaselineCell | null = null;
  if (bt && bt.id !== t.id) {
    const bs = taskScore(bt);
    baselineCell = {
      task_result_id: bt.id,
      run_id: bt.runId,
      task_hash: bt.taskHash,
      score: bs.score,
      stderr: bs.stderr,
      n: bs.n,
      metrics: bs.metrics,
      delta: pairedDelta(t, bt, alpha),
      contingency: contingency(t, bt, threshold),
    };
  }
  const failed = data ? Math.round(t.n * (t.task.type === "code" ? 0.012 : 0)) : null;
  const prefix = `gs://${RESULTS_BUCKET}/runs/${t.runId}`;
  const file = t.task.name.replace(/[:/ ]/g, "_");
  return {
    task_result_id: t.id,
    run_id: t.runId,
    task_name: t.task.name,
    task_hash: t.taskHash,
    base_task: t.task.base,
    primary_metric: t.task.primary,
    metric_meta: t.task.meta,
    score: s.score,
    stderr: s.stderr,
    score_is_mean: true,
    instance_scale: t.task.scale,
    n: s.n,
    instances_processed: t.error ? 0 : t.n,
    instances_failed: failed,
    instances_stored: t.error ? 0 : t.n,
    metrics: s.metrics,
    error: t.error,
    error_summary: t.error ? { type: t.error.split(":")[0], count: 1 } : failed ? { SandboxTimeoutError: failed } : null,
    duration_seconds: t.durationS,
    runtime: taskRuntime(t),
    completion_tokens_total: tokenCount ? tokens : null,
    mean_completion_tokens: tokenCount ? tokens / tokenCount : null,
    truncation_rate: tokenCount ? (finish.length ?? 0) / tokenCount : null,
    finish_reason_counts: finish,
    num_fewshot: t.task.fewshot,
    limit: t.limit,
    split: t.task.split,
    suites: SUITES.filter((su) => suiteLeaves(su.name).includes(t.task.name)).map((su) => su.name),
    predictions_uri: t.error ? null : `${prefix}/predictions/${file}_${t.taskHash.slice(-6)}-predictions.jsonl`,
    requests_uri: t.error ? null : `${prefix}/requests/${file}_${t.taskHash.slice(-6)}-requests.jsonl`,
    baseline: baselineCell,
  };
}

export function runTaskResults(runId: string, q: Query): T.RunTaskResultsResponse {
  const run = getRun(runId);
  const baselineKey = q.get("baseline");
  const baseline = baselineKey ? resolveSubject(baselineKey) : null;
  const alpha = Number(q.get("alpha") ?? 0.05);
  const threshold = Number(q.get("threshold") ?? 0.5);
  return {
    run_id: runId,
    baseline: baseline?.info ?? null,
    items: runTRs(run).map((t) => taskRow(t, baseline, alpha, threshold)),
    computed_at: nowIso(),
  };
}

function suiteDelta(name: string, a: Map<string, MockTaskResult>, b: Map<string, MockTaskResult>, alpha: number): T.DeltaStats {
  const def = SUITE_BY_NAME.get(name)!;
  const parts: { stats: T.DeltaStats; se: number; w: number }[] = [];
  const k = def.children.length;
  for (const child of def.children) {
    if (child.type === "task") {
      const ta = a.get(child.name);
      const tb = b.get(child.name);
      if (!ta || !tb) continue;
      const stats = pairedDelta(ta, tb, alpha);
      parts.push({ stats, se: pairedSummary(ta, tb).se, w: 1 / k });
    } else {
      const stats = suiteDelta(child.name, a, b, alpha);
      if (stats.delta == null || stats.ci_high == null) continue;
      parts.push({ stats, se: (stats.ci_high - stats.delta) / zFor(alpha), w: 1 / k });
    }
  }
  return combineDeltas(parts, true, alpha);
}

export function runSuites(runId: string, q: Query): T.RunSuitesResponse {
  const run = getRun(runId);
  const baselineKey = q.get("baseline");
  const baseline = baselineKey ? resolveSubject(baselineKey) : null;
  const alpha = Number(q.get("alpha") ?? 0.05);
  const trs = runTRs(run);
  const names = new Set(trs.map((t) => t.task.name));
  const mine = new Map(trs.map((t) => [t.task.name, t]));
  const used = suitesFor(names);
  const usedNames = new Set(used.map((s) => s.name));
  const items: T.SuiteResultRow[] = [];
  const visit = (name: string, parent: string | null, depth: number) => {
    const def = SUITE_BY_NAME.get(name)!;
    const s = suiteScore(name, (t) => mine.get(t));
    let baseCell: T.SuiteBaselineCell | null = null;
    if (baseline) {
      const bs = suiteScore(name, (t) => baseline.lookup.get(t));
      baseCell = { score: bs.score, stderr: bs.stderr, delta: suiteDelta(name, mine, baseline.lookup, alpha) };
    }
    items.push({
      name,
      parent,
      depth,
      aggregation: def.aggregation,
      description: def.description,
      children: def.children,
      score: s.score,
      stderr: s.stderr,
      num_tasks: s.numTasks,
      n_instances: s.nInstances,
      display_format: "percent",
      higher_is_better: true,
      tasks_missing: s.missing,
      baseline: baseCell,
    });
    for (const child of def.children) if (child.type === "suite" && usedNames.has(child.name)) visit(child.name, name, depth + 1);
  };
  for (const s of used) if (!parentSuite(s.name) || !usedNames.has(parentSuite(s.name)!)) visit(s.name, null, 0);
  const suited = new Set(used.flatMap((s) => suiteLeaves(s.name)));
  return {
    run_id: runId,
    baseline: baseline?.info ?? null,
    items,
    unsuited_tasks: [...names].filter((n) => !suited.has(n)),
    computed_at: nowIso(),
  };
}

// ------------------------------------------------------------------------------------- instances

function instanceRow(t: MockTaskResult, i: number, bt: MockTaskResult | null, threshold: number): T.InstanceRow {
  const data = instanceData(t);
  const text = instanceText(t, i);
  const v = data.primary[i];
  const bv = bt && i < bt.n ? instanceData(bt).primary[i] : null;
  return {
    native_id: nativeId(t.task, i),
    doc_id: i,
    primary_score: v,
    correct: isCorrect(t, v, threshold),
    metrics: instanceMetrics(t, i),
    label: text.label,
    extracted_answer: text.extracted,
    finish_reason: finishReason(t, i),
    completion_tokens: completionTokens(t, i),
    prompt_tokens: Math.round(text.prompt.length / 3.8) + (t.task.fewshot ?? 0) * 120,
    num_outputs: t.task.type === "mc" ? 4 : 1,
    prompt_preview: text.prompt.slice(0, 300),
    output_preview: text.output ? text.output.replace(/<think>[\s\S]*?<\/think>\s*/, "").slice(0, 300) : null,
    judge_verdict: text.judgeVerdict,
    has_scoring_error: hasScoringError(t, i),
    has_execution_result: t.task.type === "code",
    has_judge_result: !!text.judgeVerdict,
    has_trajectory: t.task.type === "agent",
    baseline_score: bv,
    baseline_correct: bt && bv != null ? isCorrect(bt, bv, threshold) : null,
    delta: bv != null ? v - bv : null,
  };
}

export function instances(runId: string, trId: number, q: Query): T.InstancesResponse {
  const t = tr(trId);
  if (t.runId !== runId) throw new MockHttpError(404, "not_found", "Task result does not belong to this run.");
  const threshold = Number(q.get("threshold") ?? 0.5);
  const baselineKey = q.get("baseline");
  const baseline = baselineKey ? resolveSubject(baselineKey) : null;
  const bt = baseline?.lookup.get(t.task.name) ?? null;
  const btUse = bt && bt.id !== t.id && !bt.error ? bt : null;
  const correct = q.get("correct");
  if (correct && t.task.kind === "unbounded") throw new MockHttpError(400, "bad_request", "Correctness filters do not apply to unbounded metrics.");
  const vs = q.get("vs");
  const finishes = q.getAll("finish_reason").flatMap((v) => v.split(","));
  const lenMin = q.get("len_min");
  const lenMax = q.get("len_max");
  const scoreMin = q.get("score_min");
  const scoreMax = q.get("score_max");
  const has = q.getAll("has").flatMap((v) => v.split(","));
  const text = q.get("q")?.toLowerCase();
  let rows: T.InstanceRow[] = [];
  if (!t.error) {
    for (let i = 0; i < t.n; i++) {
      const row = instanceRow(t, i, btUse, threshold);
      const v = row.primary_score ?? 0;
      if (correct === "correct" && !row.correct) continue;
      if (correct === "incorrect" && row.correct !== false) continue;
      if (correct === "partial" && !(v > 0 && v < 1)) continue;
      if (vs && btUse) {
        if (vs === "gained" && !(row.correct && row.baseline_correct === false)) continue;
        if (vs === "lost" && !(row.correct === false && row.baseline_correct)) continue;
        if (vs === "both_right" && !(row.correct && row.baseline_correct)) continue;
        if (vs === "both_wrong" && !(row.correct === false && row.baseline_correct === false)) continue;
        if (vs === "changed" && !(row.delta != null && row.delta !== 0)) continue;
      }
      if (finishes.length && !finishes.includes(row.finish_reason ?? "")) continue;
      if (lenMin && (row.completion_tokens ?? 0) < Number(lenMin)) continue;
      if (lenMax && (row.completion_tokens ?? 0) > Number(lenMax)) continue;
      if (scoreMin && v < Number(scoreMin)) continue;
      if (scoreMax && v > Number(scoreMax)) continue;
      if (has.includes("scoring_error") && !row.has_scoring_error) continue;
      if (has.includes("execution_result") && !row.has_execution_result) continue;
      if (has.includes("judge_result") && !row.has_judge_result) continue;
      if (has.includes("trajectory") && !row.has_trajectory) continue;
      if (text) {
        const hay = `${row.prompt_preview} ${row.output_preview} ${row.label} ${row.native_id}`.toLowerCase();
        if (!hay.includes(text)) continue;
      }
      rows.push(row);
    }
  }
  const sort = q.get("sort") ?? "doc_id";
  const desc = sort.startsWith("-");
  const key = desc ? sort.slice(1) : sort;
  const get = (r: T.InstanceRow): number | string =>
    key === "score" ? r.primary_score ?? -Infinity : key === "delta" ? r.delta ?? 0 : key === "length" ? r.completion_tokens ?? 0 : key === "native_id" ? r.native_id : r.doc_id ?? 0;
  rows = rows.sort((a, b) => {
    const va = get(a);
    const vb = get(b);
    const cmp = va < vb ? -1 : va > vb ? 1 : 0;
    return desc ? -cmp : cmp;
  });
  const { page, next_cursor, total } = paginate(rows, q, 200, 1000);
  return {
    task_result_id: t.id,
    baseline_task_result_id: btUse?.id ?? null,
    kind: t.task.kind,
    threshold,
    items: page,
    next_cursor,
    total,
  };
}

export function instanceDetail(q: Query): T.InstanceDetailResponse {
  const t = tr(Number(q.get("task_result_id")));
  const native = q.get("native_id") ?? "";
  const i = indexOfNative(t.task, native);
  if (i < 0 || i >= t.n) throw new MockHttpError(404, "not_found", `Instance ${native} not found.`);
  const records = instanceRecords(t, i);
  const row = taskRow(t, null, 0.05, 0.5);
  return {
    task_result_id: t.id,
    run_id: t.runId,
    task_name: t.task.name,
    native_id: native,
    doc_id: i,
    primary_score: instanceData(t).primary[i],
    metrics: instanceMetrics(t, i),
    prediction: records.prediction,
    request: records.request,
    predictions_uri: row.predictions_uri,
    requests_uri: row.requests_uri,
    unavailable_reason: null,
  };
}

export function histograms(_runId: string, trId: number, q: Query): T.HistogramsResponse {
  const t = tr(trId);
  const bins = Math.max(5, Math.min(100, Number(q.get("bins") ?? 30)));
  const threshold = Number(q.get("threshold") ?? 0.5);
  const baselineKey = q.get("baseline");
  const baseline = baselineKey ? resolveSubject(baselineKey) : null;
  const bt = baseline?.lookup.get(t.task.name);
  const data = instanceData(t);
  const values = Array.from(data.primary.slice(0, t.n));
  const binary = t.task.kind === "binary";
  const range = t.task.kind === "unbounded" ? undefined : ([0, 1] as const);
  const lengthsCorrect: number[] = [];
  const lengthsWrong: number[] = [];
  const finishes: Record<string, number> = {};
  for (let i = 0; i < t.n; i++) {
    const c = completionTokens(t, i);
    const f = finishReason(t, i);
    if (f) finishes[f] = (finishes[f] ?? 0) + 1;
    if (c == null) continue;
    if (isCorrect(t, data.primary[i], threshold)) lengthsCorrect.push(c);
    else lengthsWrong.push(c);
  }
  const allLengths = [...lengthsCorrect, ...lengthsWrong];
  const maxLen = allLengths.length ? Math.max(...allLengths) : 1;
  let delta: T.Histogram | null = null;
  let baseScore: T.Histogram | null = null;
  if (bt && bt.id !== t.id && !bt.error) {
    const bd = instanceData(bt).primary;
    const n = Math.min(t.n, bt.n);
    const diffs: number[] = [];
    for (let i = 0; i < n; i++) diffs.push(data.primary[i] - bd[i]);
    if (!binary) {
      delta = histogram(diffs, bins);
      baseScore = histogram(Array.from(bd.slice(0, bt.n)), bins, range?.[0], range?.[1]);
    }
  }
  return {
    task_result_id: t.id,
    metric: t.task.primary,
    kind: t.task.kind,
    score: binary ? null : histogram(values, bins, range?.[0], range?.[1]),
    baseline_score: baseScore,
    delta,
    completion_tokens: allLengths.length
      ? {
          correct: histogram(lengthsCorrect, bins, 0, maxLen),
          incorrect: histogram(lengthsWrong, bins, 0, maxLen),
          all: histogram(allLengths, bins, 0, maxLen),
        }
      : null,
    finish_reasons: finishes,
    max_tokens: t.maxTokens,
  };
}

// ------------------------------------------------------------------------------------- inference

function inferenceKpis(run: MockRun): { kpis: T.InferenceKpis; batches: T.InferenceBatchOut[]; per: T.InferenceTaskRow[] } {
  const trs = runTRs(run).filter((t) => !t.error);
  const batches: T.InferenceBatchOut[] = [];
  const per: T.InferenceTaskRow[] = [];
  let t0 = 0;
  let seq = 0;
  const speedBase = run.model.step != null && run.model.series.includes("32b") ? 1600 : run.model.series.includes("1b") ? 9000 : 4200;
  for (const t of trs) {
    let total = 0;
    let prompt = 0;
    const lengths: number[] = [];
    let trunc = 0;
    for (let i = 0; i < t.n; i++) {
      const c = completionTokens(t, i) ?? 1;
      lengths.push(c);
      total += c;
      prompt += 180 + (t.task.fewshot ?? 0) * 140;
      if (finishReason(t, i) === "length") trunc += 1;
    }
    const nb = Math.max(1, Math.ceil(t.n / 64));
    for (let b = 0; b < nb; b++) {
      const rng = rngFor("batch", run.summary.run_id, t.task.name, b);
      const reqs = Math.min(64, t.n - b * 64);
      const comp = Math.round((total / t.n) * reqs);
      const tps = speedBase * (t.task.meanTokens ? 1 : 0.15) * (0.75 + 0.5 * rng());
      const wall = Math.max(0.4, comp / tps);
      batches.push({
        seq: seq++,
        t: t0,
        task_name: t.task.name,
        total_requests: reqs,
        failed_requests: rng() < 0.02 ? 1 : 0,
        total_prompt_tokens: Math.round((prompt / t.n) * reqs),
        total_completion_tokens: comp,
        wall_clock_time_s: wall,
        output_tokens_per_second: comp / wall,
        mean_latency_s: wall * (0.55 + 0.3 * rng()),
      });
      t0 += wall + 0.3;
    }
    const sorted = [...lengths].sort((a, b) => a - b);
    per.push({
      task_name: t.task.name,
      task_result_id: t.id,
      instances: t.n,
      completion_tokens: t.task.meanTokens ? total : null,
      prompt_tokens: prompt,
      mean_completion_tokens: t.task.meanTokens ? total / t.n : null,
      truncation_rate: t.task.meanTokens ? trunc / t.n : null,
      duration_seconds: t.durationS,
      completion_tokens_box: t.task.meanTokens
        ? {
            p5: quantile(sorted, 0.05),
            p25: quantile(sorted, 0.25),
            p50: quantile(sorted, 0.5),
            p75: quantile(sorted, 0.75),
            p95: quantile(sorted, 0.95),
            mean: total / t.n,
            n: t.n,
          }
        : null,
    });
  }
  const totalComp = batches.reduce((a, b) => a + b.total_completion_tokens, 0);
  const totalReq = batches.reduce((a, b) => a + b.total_requests, 0);
  const wall = t0;
  const kpis: T.InferenceKpis = {
    wall_clock_s: wall,
    total_requests: totalReq,
    failed_requests: batches.reduce((a, b) => a + b.failed_requests, 0),
    prompt_tokens: batches.reduce((a, b) => a + b.total_prompt_tokens, 0),
    completion_tokens: totalComp,
    output_tokens_per_second: wall ? totalComp / wall : null,
    mean_latency_s: totalReq ? batches.reduce((a, b) => a + b.mean_latency_s * b.total_requests, 0) / totalReq : null,
    ttft_p50_s: run.thinking ? 0.21 : null,
    ttft_p95_s: run.thinking ? 0.84 : null,
    gpu_utilization_mean_pct: 71 + (run.summary.run_id.charCodeAt(0) % 15),
    gpu_memory_max_mb: 75267.9,
    gpu_hours: run.summary.duration_seconds ? (run.summary.duration_seconds / 3600) * (run.beaker?.gpu_count ?? 1) : null,
  };
  return { kpis, batches, per };
}

function box(values: number[]): T.BoxStats | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return {
    p5: quantile(sorted, 0.05),
    p25: quantile(sorted, 0.25),
    p50: quantile(sorted, 0.5),
    p75: quantile(sorted, 0.75),
    p95: quantile(sorted, 0.95),
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    n: values.length,
  };
}

export function inference(runId: string, q: Query): T.InferenceResponse {
  const run = getRun(runId);
  const { kpis, batches, per } = inferenceKpis(run);
  const baselineKey = q.get("baseline");
  let baselineKpis: T.InferenceKpis | null = null;
  if (baselineKey) {
    const b = resolveSubject(baselineKey);
    const bRunId = b?.info.run?.run_id ?? b?.info.run_ids[0];
    if (bRunId && bRunId !== runId) baselineKpis = inferenceKpis(getRun(bRunId)).kpis;
  }
  if (!batches.length) {
    return {
      run_id: runId,
      available: false,
      kpis,
      baseline_kpis: baselineKpis,
      per_task: [],
      batch_latency: null,
      batch_latency_by_task: [],
      request_latency: null,
      batches: [],
      series: [],
      gpu_devices: [],
      started_at: run.summary.started_at,
    };
  }
  const end = batches[batches.length - 1].t + batches[batches.length - 1].wall_clock_time_s;
  const steps = 240;
  const series = (name: string, unit: string, f: (x: number, b: T.InferenceBatchOut | undefined) => number): T.SeriesOut => {
    const points: [number, number][] = [];
    let bi = 0;
    for (let s = 0; s <= steps; s++) {
      const x = (end * s) / steps;
      while (bi < batches.length - 1 && batches[bi + 1].t <= x) bi += 1;
      points.push([Math.round(x * 10) / 10, Math.round(f(x, batches[bi]) * 100) / 100]);
    }
    return { name, unit, device: null, points };
  };
  const rng = rngFor("series", runId);
  const jitter = () => (rng() - 0.5) * 2;
  const byTask = new Map<string, number[]>();
  batches.forEach((b) => {
    if (!b.task_name) return;
    byTask.set(b.task_name, [...(byTask.get(b.task_name) ?? []), b.mean_latency_s]);
  });
  return {
    run_id: runId,
    available: true,
    kpis,
    baseline_kpis: baselineKpis,
    per_task: per,
    batch_latency: box(batches.map((b) => b.mean_latency_s)),
    batch_latency_by_task: [...byTask.entries()].map(([task_name, v]) => ({ task_name, box: box(v)! })),
    request_latency: run.thinking
      ? {
          end_to_end_s: { p5: 0.9, p25: 3.1, p50: 7.4, p75: 15.2, p95: 38.5, mean: 11.3, n: kpis.total_requests ?? 0 },
          ttft_s: { p5: 0.05, p25: 0.11, p50: 0.21, p75: 0.38, p95: 0.84, mean: 0.29, n: kpis.total_requests ?? 0 },
          tpot_s: { p5: 0.009, p25: 0.011, p50: 0.013, p75: 0.016, p95: 0.022, mean: 0.014, n: kpis.total_requests ?? 0 },
        }
      : null,
    batches: batches.slice(0, 2000),
    series: [
      series("output_tokens_per_second", "tok/s", (_x, b) => (b ? b.output_tokens_per_second * (1 + 0.08 * jitter()) : 0)),
      series("vllm_num_requests_running", "requests", (_x, b) => (b ? Math.min(256, b.total_requests * 3.6 + 20 * jitter()) : 0)),
      series("vllm_num_requests_waiting", "requests", (x) => Math.max(0, 40 * Math.sin(x / 90) + 18 * jitter())),
      series("vllm_kv_cache_usage_pct", "%", (_x, b) => Math.min(99, 35 + (b?.total_completion_tokens ?? 0) / 400 + 6 * jitter())),
      series("gpu_utilization_pct", "%", (_x, b) => Math.min(100, (b?.output_tokens_per_second ?? 0) > 500 ? 88 + 8 * jitter() : 35 + 15 * jitter())),
      series("gpu_memory_used_mb", "MB", () => 75200 + 60 * jitter()),
    ],
    gpu_devices: Array.from({ length: run.beaker?.gpu_count ?? 1 }, (_, i) => ({
      device_id: i,
      name: run.environment.gpu_type ?? "GPU",
      memory_total_mb: 81559,
    })),
    started_at: run.summary.started_at,
  };
}

// ------------------------------------------------------------------------------------- configs, artifacts

export function taskConfig(t: MockTaskResult): T.JsonObject {
  return {
    task_name: t.task.name,
    task_hash: t.taskHash,
    version: 2,
    split: t.task.split,
    num_fewshot: t.task.fewshot,
    fewshot_seed: 1234,
    limit: t.limit,
    primary_metric: t.task.primary,
    metrics: t.task.metrics.map((m) => ({ name: m.split(":")[0], scorer: m.split(":")[1] })),
    generation_kwargs: t.task.maxTokens
      ? { max_gen_toks: t.maxTokens, temperature: 0.0, do_sample: false, stop_sequences: ["Question:", "\n\n"] }
      : null,
    chat_format: t.thinking || ["safety", "judge", "ifeval", "omni", "agent"].includes(t.task.type),
    request_type: ["mc", "bpb"].includes(t.task.type) ? "loglikelihood" : "generate_until",
    ...(t.task.type === "code" ? { sandbox: { image: "olmo-eval-sandbox:py312", timeout_s: 60, workers: 16 } } : {}),
    ...(t.task.type === "judge" || t.task.type === "omni" ? { judge: { model: "gpt-4.1-2025-04-14", temperature: 0 } } : {}),
  };
}

export function runConfigs(runId: string): T.RunConfigsResponse {
  const run = getRun(runId);
  return {
    run_id: runId,
    model_config: run.model.provider_config,
    harness_config: { batch_size: "auto", save_predictions: true, save_requests: true, inference_metrics: { enabled: true, reporter: "file" } },
    environment: run.environment,
    argv: run.argv,
    task_configs: runTRs(run).map((t) => ({ task_result_id: t.id, task_name: t.task.name, task_hash: t.taskHash, config: taskConfig(t) })),
  };
}

export function artifacts(runId: string): T.ArtifactsResponse {
  const run = getRun(runId);
  const prefix = run.summary.gcs_prefix;
  const items: T.ArtifactRow[] = [];
  const add = (path: string, kind: T.ArtifactKind, size: number, task: string | null = null) =>
    items.push({
      path,
      kind,
      task_name: task,
      size_bytes: size,
      uploaded: run.summary.upload_state === "complete" || kind !== "predictions",
      updated_at: run.summary.finished_at ?? run.summary.created_at,
      gs_uri: `${prefix}${path}`,
      console_url: `https://console.cloud.google.com/storage/browser/_details/${RESULTS_BUCKET}/runs/${runId}/${path}?project=ai2-skiff2-olmo-eval`,
    });
  add("metrics.json", "metrics", 48_211);
  add("manifest.json", "manifest", 6_310);
  for (const t of runTRs(run)) {
    if (t.error) continue;
    const file = t.task.name.replace(/[:/ ]/g, "_");
    const model = run.model.name.replace(/[/:]/g, "_");
    const size = t.n * (t.task.meanTokens ? t.task.meanTokens * 9 : 1400);
    add(`predictions/${model}/${file}_${t.taskHash.slice(-6)}-predictions.jsonl`, "predictions", size, t.task.name);
    add(`requests/${model}/${file}_${t.taskHash.slice(-6)}-requests.jsonl`, "requests", Math.round(size * 1.6), t.task.name);
  }
  add(`metrics/vllm_${run.model.series_label}-inference.jsonl`, "inference_metrics", 1_204_118);
  add(`logs/vllm_server_${run.model.series_label}/vllm_server_8000.log`, "logs", 3_841_002);
  return {
    run_id: runId,
    gcs_prefix: prefix,
    gcs_console: run.summary.links.gcs_console ?? "",
    items,
    beaker: run.beaker,
    links: run.summary.links,
  };
}

export function signDownload(body: T.SignDownloadRequest): T.SignDownloadResponse {
  if (!body.gs_uri.startsWith(`gs://${RESULTS_BUCKET}/`)) throw new MockHttpError(400, "bad_request", "Only results objects can be signed.");
  const path = body.gs_uri.replace("gs://", "");
  return {
    url: `https://storage.googleapis.com/${path}?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=mock`,
    expires_at: new Date(NOW + 15 * 60_000).toISOString(),
  };
}

export function baselineSuggestions(runId: string): T.BaselineSuggestionsResponse {
  const run = getRun(runId);
  const w = getWorld();
  const items: T.BaselineSuggestion[] = [];
  const step = run.model.step;
  if (step != null) {
    const prev = w.runs
      .filter((r) => r.model.series === run.model.series && r.model.settings_hash === run.model.settings_hash && r.model.step != null && r.model.step < step && visible(r))
      .sort((a, b) => (b.model.step ?? 0) - (a.model.step ?? 0))[0];
    if (prev) {
      items.push({
        subject: `m:${prev.model.model_id}`,
        reason: "previous_checkpoint",
        label: baseLabel(prev.summary.model),
        model: prev.summary.model,
        run: null,
      });
    }
  }
  if (run.summary.experiment_group) {
    w.runs
      .filter((r) => r.summary.experiment_group === run.summary.experiment_group && r.summary.run_id !== runId && visible(r) && r.model.model_id !== run.model.model_id)
      .filter((r) => r.summary.tags.includes("reference") || r.model.step !== step)
      .sort((a, b) => Number(b.summary.tags.includes("reference")) - Number(a.summary.tags.includes("reference")))
      .slice(0, 3)
      .forEach((r) =>
        items.push({ subject: `r:${r.summary.run_id}`, reason: "same_group", label: baseLabel(r.summary.model), model: r.summary.model, run: r.summary }),
      );
  }
  const prevSame = w.runs.find(
    (r) => r.model.model_id === run.model.model_id && r.summary.run_id !== runId && r.summary.created_at < run.summary.created_at && visible(r),
  );
  if (prevSame) {
    items.push({ subject: `r:${prevSame.summary.run_id}`, reason: "previous_run_same_tasks", label: baseLabel(prevSame.summary.model), model: prevSame.summary.model, run: prevSame.summary });
  }
  return { items };
}

// ------------------------------------------------------------------------------------- compare

interface ScopeRow {
  key: string;
  kind: "task" | "suite";
  name: string;
  depth: number;
  parent: string | null;
  aggregation: string | null;
}

function suiteRows(name: string, depth: number, parent: string | null, out: ScopeRow[]) {
  const def = SUITE_BY_NAME.get(name);
  if (!def) return;
  out.push({ key: `suite:${name}`, kind: "suite", name, depth, parent, aggregation: def.aggregation });
  for (const child of def.children) {
    if (child.type === "suite") suiteRows(child.name, depth + 1, name, out);
    else out.push({ key: `task:${child.name}`, kind: "task", name: child.name, depth: depth + 1, parent: name, aggregation: null });
  }
}

function scopeRows(scope: string, subjects: Resolved[]): ScopeRow[] {
  const union = new Set<string>();
  subjects.forEach((s) => s.lookup.forEach((_v, k) => union.add(k)));
  if (scope.startsWith("suite:")) {
    const out: ScopeRow[] = [];
    suiteRows(scope.slice(6), 0, null, out);
    if (!out.length) throw new MockHttpError(400, "bad_request", `Unknown suite ${scope.slice(6)}.`);
    return out;
  }
  if (scope.startsWith("task:")) {
    const name = scope.slice(5);
    return [{ key: `task:${name}`, kind: "task", name, depth: 0, parent: null, aggregation: null }];
  }
  let names = [...union];
  if (scope === "shared") names = names.filter((n) => subjects.every((s) => s.lookup.has(n)));
  const nameSet = new Set(names);
  const out: ScopeRow[] = [];
  const covered = new Set<string>();
  for (const s of SUITES) {
    if (parentSuite(s.name)) continue;
    const leaves = suiteLeaves(s.name);
    if (!leaves.some((l) => nameSet.has(l))) continue;
    const rows: ScopeRow[] = [];
    suiteRows(s.name, 0, null, rows);
    out.push(...rows.filter((r) => r.kind === "suite" || nameSet.has(r.name)));
    leaves.forEach((l) => covered.add(l));
  }
  for (const n of names.sort()) {
    if (!covered.has(n)) out.push({ key: `task:${n}`, kind: "task", name: n, depth: 0, parent: null, aggregation: null });
  }
  return out;
}

function metricValue(t: MockTaskResult, metric: string): number | null {
  const s = taskScore(t);
  if (metric === "primary") return s.score;
  return s.metrics[metric] ?? null;
}

const RUNTIME_META: T.MetricMeta = { kind: "unbounded", higher_is_better: false, display_format: "raw", unit: "s" };

function runtimeMode(metric: string): T.RuntimeMode | null {
  if (metric === "runtime:inference") return "inference";
  if (metric === "runtime:with_startup") return "with_startup";
  return null;
}

function rejectRuntime(metric: string) {
  if (runtimeMode(metric)) {
    throw new MockHttpError(400, "bad_request", "Runtime metrics have no instance-level pairing; use the heatmap instead.");
  }
}

function runtimeValue(t: MockTaskResult, mode: T.RuntimeMode): number | null {
  const rt = taskRuntime(t);
  return mode === "inference" ? rt.inference_seconds : rt.with_startup_seconds;
}

/** Matrix of task runtimes: seconds, lower is better, suites sum their tasks. */
function runtimeMatrix(body: T.MatrixRequest, mode: T.RuntimeMode): T.MatrixResponse {
  const subjects = resolveMany(body.subjects, body.group);
  const alpha = body.alpha ?? 0.05;
  const baselineIdx = body.baseline ? body.subjects.indexOf(body.baseline) : -1;
  const baseline = body.baseline ? (baselineIdx >= 0 ? subjects[baselineIdx] : resolveSubject(body.baseline, body.group)) : null;
  const rows = scopeRows(body.scope, subjects);
  const coverage: T.Coverage = { n_subjects: subjects.length, n_tasks: 0, complete: 0, missing: 0, failed: 0, hash_mismatch: 0 };
  const delta = (a: number | null, b: number | null) => unpairedDelta({ score: a, stderr: null }, { score: b, stderr: null }, false, alpha);
  const out: T.MatrixRow[] = rows.map((row) => {
    if (row.kind === "task") {
      coverage.n_tasks += 1;
      const hashes = new Set<string>();
      const cells: T.MatrixCell[] = subjects.map((s) => {
        const t = s.lookup.get(row.name);
        if (!t) {
          coverage.missing += 1;
          return { status: "missing", score: null, stderr: null, n: null, task_result_id: null, run_id: null, task_hash: null, delta: null, children_missing: 0 };
        }
        if (t.error) {
          coverage.failed += 1;
          return { status: "failed", score: null, stderr: null, n: 0, task_result_id: t.id, run_id: t.runId, task_hash: t.taskHash, delta: null, children_missing: 0 };
        }
        coverage.complete += 1;
        hashes.add(t.taskHash);
        const value = runtimeValue(t, mode);
        const bt = baseline?.lookup.get(row.name);
        const d = bt && !bt.error && s.info.key !== baseline?.info.key ? delta(value, runtimeValue(bt, mode)) : null;
        return { status: "ok", score: value, stderr: null, n: t.n, task_result_id: t.id, run_id: t.runId, task_hash: t.taskHash, delta: d, children_missing: 0 };
      });
      if (hashes.size > 1) coverage.hash_mismatch += 1;
      return { key: row.key, kind: "task", name: row.name, depth: row.depth, parent: row.parent, aggregation: null, metric_key: body.metric, meta: RUNTIME_META, cells };
    }
    const leaves = suiteLeaves(row.name);
    const total = (s: Resolved) => {
      let sum = 0;
      let missing = 0;
      for (const leaf of leaves) {
        const t = s.lookup.get(leaf);
        const v = t && !t.error ? runtimeValue(t, mode) : null;
        if (v == null) missing += 1;
        else sum += v;
      }
      return { sum: missing < leaves.length ? sum : null, missing };
    };
    const base = baseline ? total(baseline) : null;
    const cells: T.MatrixCell[] = subjects.map((s) => {
      const { sum, missing } = total(s);
      const d = base && s.info.key !== baseline?.info.key && sum != null ? delta(sum, base.sum) : null;
      return { status: sum == null ? "missing" : missing ? "partial" : "ok", score: sum, stderr: null, n: null, task_result_id: null, run_id: null, task_hash: null, delta: d, children_missing: missing };
    });
    return { key: row.key, kind: "suite", name: row.name, depth: row.depth, parent: row.parent, aggregation: "sum", metric_key: null, meta: RUNTIME_META, cells };
  });
  return {
    subjects: subjects.map((s) => s.info),
    baseline: body.baseline ?? null,
    scope: body.scope,
    metric: body.metric,
    rows: out,
    coverage,
    mde80: null,
    alpha,
    computed_at: nowIso(),
  };
}

export function compareMatrix(body: T.MatrixRequest): T.MatrixResponse {
  const mode = runtimeMode(body.metric);
  if (mode) return runtimeMatrix(body, mode);
  const subjects = resolveMany(body.subjects, body.group);
  const alpha = body.alpha ?? 0.05;
  const baselineIdx = body.baseline ? body.subjects.indexOf(body.baseline) : -1;
  const baseline = body.baseline ? (baselineIdx >= 0 ? subjects[baselineIdx] : resolveSubject(body.baseline, body.group)) : null;
  const rows = scopeRows(body.scope, subjects);
  const coverage: T.Coverage = { n_subjects: subjects.length, n_tasks: 0, complete: 0, missing: 0, failed: 0, hash_mismatch: 0 };
  const paired: number[] = [];
  const out: T.MatrixRow[] = rows.map((row) => {
    if (row.kind === "task") {
      coverage.n_tasks += 1;
      const task = TASK_BY_NAME.get(row.name);
      const meta = task ? task.meta[body.metric === "primary" ? task.primary : body.metric] ?? null : null;
      const hashes = new Set<string>();
      const cells: T.MatrixCell[] = subjects.map((s) => {
        const t = s.lookup.get(row.name);
        if (!t) {
          coverage.missing += 1;
          return { status: "missing", score: null, stderr: null, n: null, task_result_id: null, run_id: null, task_hash: null, delta: null, children_missing: 0 };
        }
        if (t.error) {
          coverage.failed += 1;
          return { status: "failed", score: null, stderr: null, n: 0, task_result_id: t.id, run_id: t.runId, task_hash: t.taskHash, delta: null, children_missing: 0 };
        }
        coverage.complete += 1;
        hashes.add(t.taskHash);
        const ts = taskScore(t);
        const bt = baseline?.lookup.get(row.name);
        let delta: T.DeltaStats | null = null;
        if (bt && s.info.key !== baseline?.info.key) {
          delta = body.metric === "primary" ? pairedDelta(t, bt, alpha) : unpairedDelta({ score: metricValue(t, body.metric), stderr: null }, { score: metricValue(bt, body.metric), stderr: null }, meta?.higher_is_better ?? true, alpha);
          if (delta.method === "paired_bootstrap") paired.push(pairedSummary(t, bt).se);
        }
        return {
          status: "ok",
          score: metricValue(t, body.metric),
          stderr: body.metric === "primary" ? ts.stderr : null,
          n: ts.n,
          task_result_id: t.id,
          run_id: t.runId,
          task_hash: t.taskHash,
          delta,
          children_missing: 0,
        };
      });
      if (hashes.size > 1) coverage.hash_mismatch += 1;
      return {
        key: row.key,
        kind: "task",
        name: row.name,
        depth: row.depth,
        parent: row.parent,
        aggregation: null,
        metric_key: task ? (body.metric === "primary" ? task.primary : body.metric) : null,
        meta,
        cells,
      };
    }
    const cells: T.MatrixCell[] = subjects.map((s) => {
      const sc = suiteScore(row.name, (t) => s.lookup.get(t));
      const leaves = suiteLeaves(row.name);
      let delta: T.DeltaStats | null = null;
      if (baseline && s.info.key !== baseline.info.key && sc.score != null) delta = suiteDelta(row.name, s.lookup, baseline.lookup, alpha);
      const status: T.CellStatus = sc.score == null ? "missing" : sc.missing.length ? "partial" : "ok";
      return { status, score: sc.score, stderr: sc.stderr, n: sc.nInstances, task_result_id: null, run_id: null, task_hash: null, delta, children_missing: Math.min(leaves.length, sc.missing.length) };
    });
    return {
      key: row.key,
      kind: "suite",
      name: row.name,
      depth: row.depth,
      parent: row.parent,
      aggregation: row.aggregation,
      metric_key: null,
      meta: { kind: "bounded", higher_is_better: true, display_format: "percent", unit: null },
      cells,
    };
  });
  const med = median(paired);
  return {
    subjects: subjects.map((s) => s.info),
    baseline: body.baseline ?? null,
    scope: body.scope,
    metric: body.metric,
    rows: out,
    coverage,
    mde80: med != null ? 2.8016 * med : null,
    alpha,
    computed_at: nowIso(),
  };
}

function scopeTasks(scope: string, subjects: Resolved[]): string[] {
  return scopeRows(scope, subjects)
    .filter((r) => r.kind === "task")
    .map((r) => r.name);
}

export function comparePairwise(body: T.PairwiseRequest): T.PairwiseResponse {
  rejectRuntime(body.metric);
  const subjects = resolveMany(body.subjects, body.group);
  const alpha = body.alpha ?? 0.05;
  const margin = body.margin ?? 0;
  const z = zFor(alpha);
  const tasks = scopeTasks(body.scope, subjects).filter((name) => {
    const task = TASK_BY_NAME.get(name);
    return task && task.kind !== "unbounded" && subjects.filter((s) => s.lookup.get(name) && !s.lookup.get(name)!.error).length >= 2;
  });
  const pairs: T.PairStats[] = [];
  const ses: number[] = [];
  for (const a of subjects) {
    for (const b of subjects) {
      if (a === b) continue;
      let wins = 0;
      let losses = 0;
      let ties = 0;
      let sum = 0;
      let sumSq = 0;
      let n = 0;
      for (const name of tasks) {
        const ta = a.lookup.get(name);
        const tb = b.lookup.get(name);
        if (!ta || !tb || ta.error || tb.error) continue;
        const m = Math.min(ta.n, tb.n);
        const da = instanceData(ta).primary;
        const db = instanceData(tb).primary;
        for (let i = 0; i < m; i++) {
          const d = da[i] - db[i];
          if (d > margin) wins += 1;
          else if (d < -margin) losses += 1;
          else ties += 1;
          sum += d;
          sumSq += d * d;
          n += 1;
        }
      }
      const mean = n ? sum / n : 0;
      const se = n > 1 ? Math.sqrt(Math.max(0, sumSq / n - mean * mean) / (n - 1)) : 0;
      if (n) ses.push(se);
      const contested = wins + losses;
      const signZ = contested ? (wins - contested / 2) / Math.sqrt(contested / 4) : 0;
      const pSign = contested ? Math.min(1, 2 * (1 - phiLocal(Math.abs(signZ)))) : 1;
      pairs.push({
        row: a.info.key,
        col: b.info.key,
        n_shared: n,
        n_contested: contested,
        wins,
        losses,
        ties,
        win_rate: contested ? wins / contested : 0.5,
        delta: n ? mean : null,
        ci_low: n >= 20 ? mean - z * se : null,
        ci_high: n >= 20 ? mean + z * se : null,
        p_bootstrap: n >= 20 && se > 0 ? Math.min(1, 2 * (1 - phiLocal(Math.abs(mean / se)))) : null,
        p_sign: pSign,
        prob_row_better: n >= 20 && se > 0 ? phiLocal(mean / se) : null,
        significant: n >= 20 && Math.abs(mean) > z * se,
        method: n >= 20 ? "paired_bootstrap" : n ? "insufficient" : "none",
      });
    }
  }
  const rows: T.PairwiseRowSummary[] = subjects.map((s) => {
    const mine = pairs.filter((p) => p.row === s.info.key);
    const wr = mine.map((p) => p.win_rate);
    const mean = wr.reduce((a, b) => a + b, 0) / Math.max(1, wr.length);
    const sd = Math.sqrt(wr.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, wr.length));
    const deltas = mine.map((p) => p.delta ?? 0);
    return {
      subject: s.info.key,
      mean_win_rate: mean,
      ci_low: Math.max(0, mean - (z * sd) / Math.sqrt(Math.max(1, wr.length))),
      ci_high: Math.min(1, mean + (z * sd) / Math.sqrt(Math.max(1, wr.length))),
      mean_delta: deltas.length ? deltas.reduce((a, b) => a + b, 0) / deltas.length : null,
    };
  });
  rows.sort((a, b) => b.mean_win_rate - a.mean_win_rate);
  const order = rows.map((r) => r.subject);
  const med = median(ses);
  const warnings: string[] = [];
  if (body.scope.startsWith("suite:") || body.scope === "all" || body.scope === "shared") {
    warnings.push("Pairs pool instances across tasks in scope; pooling unrelated capabilities can hide task-level differences.");
  }
  const skipped = scopeTasks(body.scope, subjects).filter((n) => TASK_BY_NAME.get(n)?.kind === "unbounded");
  if (skipped.length) warnings.push(`Skipped ${skipped.length} unbounded task(s): ${skipped.join(", ")}.`);
  return {
    subjects: order.map((k) => subjects.find((s) => s.info.key === k)!.info),
    pairs,
    rows,
    tasks_used: tasks,
    mde80: med != null ? 2.8016 * med : null,
    alpha,
    margin,
    higher_is_better: true,
    display_format: "percent",
    warnings,
    computed_at: nowIso(),
  };
}

function phiLocal(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

export function compareContingency(body: T.ContingencyRequest): T.ContingencyResponse {
  rejectRuntime(body.metric);
  const [a, b] = resolveMany([body.a, body.b], body.group);
  const threshold = body.threshold ?? 0.5;
  const tasks = scopeTasks(body.scope, [a, b]);
  const totals: T.Contingency = { both_right: 0, only_a: 0, only_b: 0, both_wrong: 0, n_shared: 0, net: 0, threshold };
  const items: T.ContingencyTaskRow[] = tasks.map((name) => {
    const ta = a.lookup.get(name);
    const tb = b.lookup.get(name);
    const task = TASK_BY_NAME.get(name)!;
    const c = ta && tb ? contingency(ta, tb, threshold) : null;
    if (c) {
      totals.both_right += c.both_right;
      totals.only_a += c.only_a;
      totals.only_b += c.only_b;
      totals.both_wrong += c.both_wrong;
      totals.n_shared += c.n_shared;
    }
    const sa = ta ? taskScore(ta).score : null;
    const sb = tb ? taskScore(tb).score : null;
    return {
      task_name: name,
      kind: task.kind,
      task_hash_a: ta?.taskHash ?? null,
      task_hash_b: tb?.taskHash ?? null,
      task_result_id_a: ta?.id ?? null,
      task_result_id_b: tb?.id ?? null,
      contingency: c,
      delta: sa != null && sb != null ? sa - sb : null,
    };
  });
  totals.net = totals.only_a - totals.only_b;
  items.sort((x, y) => (x.contingency?.net ?? 0) - (y.contingency?.net ?? 0));
  return { a: body.a, b: body.b, items, totals };
}

export function compareInstances(body: T.CompareInstancesRequest): T.CompareInstancesResponse {
  rejectRuntime(body.metric);
  const subjects = resolveMany(body.subjects, body.group);
  const threshold = body.threshold ?? 0.5;
  const trs = subjects.map((s) => {
    const t = s.lookup.get(body.task_name);
    return t && !t.error ? t : null;
  });
  const task = TASK_BY_NAME.get(body.task_name);
  if (!task) throw new MockHttpError(400, "bad_request", `Unknown task ${body.task_name}.`);
  const maxN = Math.max(0, ...trs.map((t) => t?.n ?? 0));
  const aIdx = body.a ? body.subjects.indexOf(body.a) : 0;
  const bIdx = body.b ? body.subjects.indexOf(body.b) : 1;
  const baseIdx = body.baseline ? body.subjects.indexOf(body.baseline) : -1;
  let rows: T.CompareInstanceRow[] = [];
  for (let i = 0; i < maxN; i++) {
    const scores = trs.map((t) => (t && i < t.n ? instanceData(t).primary[i] : null));
    const correct = trs.map((t, k) => (t && scores[k] != null ? isCorrect(t, scores[k]!, threshold) : null));
    const nPresent = scores.filter((s) => s != null).length;
    const nCorrect = correct.filter((c) => c === true).length;
    switch (body.filter) {
      case "disagree":
        if (nCorrect === 0 || nCorrect === nPresent) continue;
        break;
      case "all_wrong":
        if (nCorrect !== 0) continue;
        break;
      case "baseline_wrong":
        if (baseIdx < 0 || correct[baseIdx] !== false) continue;
        break;
      case "cell": {
        const ca = correct[aIdx];
        const cb = correct[bIdx];
        if (ca == null || cb == null) continue;
        const cell = ca && cb ? "both_right" : ca ? "only_a" : cb ? "only_b" : "both_wrong";
        if (cell !== body.cell) continue;
        break;
      }
    }
    const first = trs.find((t) => t && i < t.n)!;
    rows.push({
      native_id: nativeId(task, i),
      scores,
      correct,
      n_correct: nCorrect,
      n_present: nPresent,
      prompt_preview: body.include_previews ? instanceText(first, i).prompt.slice(0, 300) : null,
      output_previews: body.include_previews ? trs.map((t) => (t && i < t.n ? instanceText(t, i).output?.replace(/<think>[\s\S]*?<\/think>\s*/, "").slice(0, 300) ?? null : null)) : null,
    });
  }
  const counts = new Array(subjects.length + 1).fill(0);
  rows.forEach((r) => (counts[r.n_correct] += 1));
  const sort = body.sort ?? "disagreement";
  if (sort === "disagreement") rows.sort((x, y) => Math.abs(x.n_correct - x.n_present / 2) - Math.abs(y.n_correct - y.n_present / 2) || y.n_correct - x.n_correct);
  else if (sort === "n_correct") rows.sort((x, y) => y.n_correct - x.n_correct);
  const offset = Number(body.cursor ?? 0) || 0;
  const limit = body.limit ?? (body.include_previews ? 200 : 5000);
  const total = rows.length;
  rows = rows.slice(offset, offset + limit);
  return {
    subjects: subjects.map((s, k) => ({ key: s.info.key, task_result_id: trs[k]?.id ?? null, run_id: trs[k]?.runId ?? null })),
    kind: task.kind,
    threshold,
    items: rows,
    counts_by_n_correct: counts,
    total,
    next_cursor: offset + limit < total ? String(offset + limit) : null,
  };
}

// ------------------------------------------------------------------------------------- distributions, models

function latestPerModel(taskName: string, hash?: string | null, keep: (run: MockRun) => boolean = () => true): MockTaskResult[] {
  const w = getWorld();
  const seen = new Set<string>();
  const out: MockTaskResult[] = [];
  for (const run of w.runs) {
    if (!visible(run) || run.summary.status === "running" || !keep(run)) continue;
    if (seen.has(run.model.model_id)) continue;
    const t = runTRs(run).find((x) => x.task.name === taskName && !x.error && (!hash || x.taskHash === hash));
    if (!t) continue;
    seen.add(run.model.model_id);
    out.push(t);
  }
  return out;
}

export function distributions(body: T.DistributionsRequest): T.DistributionsResponse {
  const w = getWorld();
  return {
    items: body.items.map((key) => {
      const trs = body.per_model === "all"
        ? [...w.taskResults.values()].filter((t) => t.task.name === key.task_name && !t.error && (!key.task_hash || t.taskHash === key.task_hash))
        : latestPerModel(key.task_name, key.task_hash);
      const vals = trs.map((t) => ({ t, v: metricValue(t, key.metric) })).filter((x): x is { t: MockTaskResult; v: number } => x.v != null);
      const task = TASK_BY_NAME.get(key.task_name);
      const hib = task?.meta[key.metric === "primary" ? task.primary : key.metric]?.higher_is_better ?? true;
      const best = [...vals].sort((a, b) => (hib === false ? a.v - b.v : b.v - a.v))[0];
      return {
        task_name: key.task_name,
        task_hash: key.task_hash,
        metric: key.metric,
        n: vals.length,
        scores: vals.map((x) => x.v).sort((a, b) => a - b),
        is_quantiles: false,
        higher_is_better: hib,
        top: best ? { run_id: best.t.runId, model: getRun(best.t.runId).summary.model, score: best.v } : null,
      };
    }),
  };
}

export function modelsList(q: Query): T.ModelsListResponse {
  const w = getWorld();
  const bySeries = new Map<string, MockRun[]>();
  for (const run of w.runs) {
    if (!visible(run)) continue;
    bySeries.set(run.model.series, [...(bySeries.get(run.model.series) ?? []), run]);
  }
  const text = q.get("q")?.toLowerCase();
  const families = q.getAll("family").flatMap((v) => v.split(","));
  let rows: T.ModelSeriesRow[] = [...bySeries.entries()].map(([series, runs]) => {
    const latest = runs[0];
    const variants = new Map<string, number>();
    runs.forEach((r) => variants.set(r.model.settings_hash, (variants.get(r.model.settings_hash) ?? 0) + 1));
    const steps = runs.map((r) => r.model.step).filter((s): s is number => s != null);
    return {
      series,
      series_label: latest.model.series_label,
      family: latest.model.family,
      n_checkpoints: new Set(runs.map((r) => r.model.model_id)).size,
      n_runs: runs.length,
      latest_step: steps.length ? Math.max(...steps) : null,
      last_run_at: latest.summary.created_at,
      latest_run: latest.summary,
      variants: [...variants.entries()].map(([settings_hash, n_runs]) => ({ settings_hash, n_runs })),
    };
  });
  if (text) rows = rows.filter((r) => `${r.series} ${r.family}`.toLowerCase().includes(text));
  if (families.length) rows = rows.filter((r) => families.includes(r.family ?? ""));
  const sort = q.get("sort") ?? "-last_run_at";
  rows.sort((a, b) =>
    sort === "series" ? a.series.localeCompare(b.series) : sort === "-n_runs" ? b.n_runs - a.n_runs : (b.last_run_at ?? "").localeCompare(a.last_run_at ?? ""),
  );
  const { page, next_cursor, total } = paginate(rows, q);
  return { items: page, next_cursor, total };
}

export function modelSeries(q: Query): T.ModelSeriesDetailResponse {
  const series = q.get("series") ?? "";
  const runs = getWorld().runs.filter((r) => r.model.series === series && visible(r));
  if (!runs.length) throw new MockHttpError(404, "not_found", `Model series ${series} not found.`);
  const byModel = new Map<string, MockRun[]>();
  runs.forEach((r) => byModel.set(r.model.model_id, [...(byModel.get(r.model.model_id) ?? []), r]));
  const checkpoints: T.CheckpointRow[] = [...byModel.values()].map((rs) => ({
    model: rs[0].summary.model,
    settings_hash: rs[0].model.settings_hash,
    run_ids: rs.map((r) => r.summary.run_id),
    latest_run_at: rs[0].summary.created_at,
  }));
  checkpoints.sort((a, b) => (a.model.step ?? Infinity) - (b.model.step ?? Infinity));
  const variants = new Map<string, MockRun[]>();
  runs.forEach((r) => variants.set(r.model.settings_hash, [...(variants.get(r.model.settings_hash) ?? []), r]));
  const sorted = [...variants.entries()].sort((a, b) => b[1].length - a[1].length);
  const common = sorted[0][1][0].model.provider_config as Record<string, unknown>;
  return {
    series,
    series_label: runs[0].model.series_label,
    family: runs[0].model.family,
    n_runs: runs.length,
    checkpoints,
    variants: sorted.map(([hash, rs]) => {
      const cfg = rs[0].model.provider_config as Record<string, unknown>;
      const differs: Record<string, unknown> = {};
      for (const k of Object.keys(cfg)) if (k !== "model" && k !== "revision" && JSON.stringify(cfg[k]) !== JSON.stringify(common[k])) differs[k] = cfg[k];
      return { settings_hash: hash, n_runs: rs.length, differs };
    }),
  };
}

export function progression(q: Query): T.ProgressionResponse {
  const series = q.get("series") ?? "";
  const settings = q.get("settings_hash");
  const metric = q.get("metric") ?? "primary";
  const merge = q.get("merge") ?? "latest";
  const w = getWorld();
  const runs = w.runs.filter((r) => r.model.series === series && visible(r) && (!settings || r.model.settings_hash === settings));
  if (!runs.length) throw new MockHttpError(404, "not_found", `Model series ${series} not found.`);
  const scope = q.get("scope");
  const taskList = (q.get("tasks") ?? "").split(",").filter(Boolean);
  let panels: { key: string; kind: "task" | "suite"; name: string }[];
  if (taskList.length) panels = taskList.map((t) => ({ key: `task:${t}`, kind: "task", name: t }));
  else if (scope?.startsWith("suite:")) {
    const def = SUITE_BY_NAME.get(scope.slice(6));
    panels = (def?.children ?? []).map((c) => ({ key: `${c.type}:${c.name}`, kind: c.type, name: c.name }));
  } else if (scope?.startsWith("task:")) panels = [{ key: scope, kind: "task", name: scope.slice(5) }];
  else {
    // The most complete of the recent runs (reruns often cover only a few tasks).
    const latest = [...runs].sort((x, y) => y.taskResultIds.length - x.taskResultIds.length)[0];
    const names = new Set(runTRs(latest).map((t) => t.task.name));
    const tops = suitesFor(names).filter((s) => !parentSuite(s.name));
    const extras = suitesFor(names).filter((s) => parentSuite(s.name));
    panels = [...tops, ...extras].map((s) => ({ key: `suite:${s.name}`, kind: "suite", name: s.name }));
    const unsuited = [...names].filter((n) => !TASK_BY_NAME.get(n)?.suite).slice(0, 6);
    panels.push(...unsuited.map((n) => ({ key: `task:${n}`, kind: "task" as const, name: n })));
  }
  const byModel = new Map<string, MockRun[]>();
  runs.forEach((r) => byModel.set(r.model.model_id, [...(byModel.get(r.model.model_id) ?? []), r]));
  const references = (q.get("references") ?? "").split(",").filter(Boolean).map((k) => resolveSubject(k)).filter((r): r is Resolved => !!r);
  return {
    series,
    settings_hash: settings,
    metric,
    panels: panels.map((p) => {
      const points: T.ProgressionPoint[] = [];
      for (const [modelId, rs] of byModel) {
        const groups = merge === "all" ? rs.map((r) => [r]) : [rs];
        for (const group of groups) {
          const lookup = new Map<string, MockTaskResult>();
          for (const r of group) for (const t of runTRs(r)) if (!t.error && !lookup.has(t.task.name)) lookup.set(t.task.name, t);
          let score: number | null;
          let stderr: number | null;
          let n: number | null;
          let trId: number | null = null;
          let runId = group[0].summary.run_id;
          if (p.kind === "task") {
            const t = lookup.get(p.name);
            if (!t) continue;
            const s = taskScore(t);
            score = metric === "primary" ? s.score : s.metrics[metric] ?? null;
            stderr = s.stderr;
            n = s.n;
            trId = t.id;
            runId = t.runId;
          } else {
            const s = suiteScore(p.name, (t) => lookup.get(t));
            if (s.score == null) continue;
            score = s.score;
            stderr = s.stderr;
            n = s.nInstances;
          }
          const m = group[0].summary.model;
          points.push({ model_id: modelId, run_id: runId, task_result_id: trId, step: m.step, tokens_seen: m.tokens_seen, date: group[0].summary.created_at, score, stderr, n });
        }
      }
      points.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
      const task = p.kind === "task" ? TASK_BY_NAME.get(p.name) : null;
      return {
        key: p.key,
        kind: p.kind,
        name: p.name,
        meta: task ? task.meta[metric === "primary" ? task.primary : metric] ?? null : { kind: "bounded", higher_is_better: true, display_format: "percent", unit: null },
        points,
        references: references.map((ref) => {
          let score: number | null = null;
          let stderr: number | null = null;
          if (p.kind === "task") {
            const t = ref.lookup.get(p.name);
            if (t && !t.error) ({ score, stderr } = taskScore(t));
          } else ({ score, stderr } = suiteScore(p.name, (t) => ref.lookup.get(t)));
          return { subject: ref.info.key, label: ref.info.label, score, stderr };
        }),
      };
    }),
  };
}

// ------------------------------------------------------------------------------------- tasks, suites

export function tasksList(q: Query): T.TasksListResponse {
  const w = getWorld();
  const text = q.get("q")?.toLowerCase();
  const suite = q.get("suite");
  let rows: T.TaskRow[] = TASKS.map((task) => {
    const trs = [...w.taskResults.values()].filter((t) => t.task.name === task.name);
    const runs = trs.map((t) => getRun(t.runId));
    return {
      task_name: task.name,
      base_task: task.base,
      n_variants: new Set(trs.map((t) => t.taskHash)).size,
      n_runs: trs.length,
      n_models: new Set(runs.map((r) => r.model.model_id)).size,
      suites: SUITES.filter((s) => suiteLeaves(s.name).includes(task.name)).map((s) => s.name),
      primary_metric: task.primary,
      last_run_at: runs.map((r) => r.summary.created_at).sort().pop() ?? null,
    };
  });
  if (text) {
    const toks = text.split(/[\s:]+/).filter(Boolean);
    rows = rows.filter((r) => toks.every((tok) => r.task_name.toLowerCase().includes(tok)));
  }
  if (suite) rows = rows.filter((r) => suiteLeaves(suite).includes(r.task_name));
  const sort = q.get("sort") ?? "-n_runs";
  rows.sort((a, b) =>
    sort === "task_name" ? a.task_name.localeCompare(b.task_name) : sort === "-last_run_at" ? (b.last_run_at ?? "").localeCompare(a.last_run_at ?? "") : b.n_runs - a.n_runs || a.task_name.localeCompare(b.task_name),
  );
  const { page, next_cursor, total } = paginate(rows, q);
  return { items: page, next_cursor, total };
}

export function taskDetail(q: Query): T.TaskDetailResponse {
  const name = q.get("task") ?? "";
  const task = TASK_BY_NAME.get(name);
  if (!task) throw new MockHttpError(404, "not_found", `Task ${name} not found.`);
  const trs = [...getWorld().taskResults.values()].filter((t) => t.task.name === name);
  const byHash = new Map<string, MockTaskResult[]>();
  trs.forEach((t) => byHash.set(t.taskHash, [...(byHash.get(t.taskHash) ?? []), t]));
  return {
    task_name: name,
    base_task: task.base,
    suites: SUITES.filter((s) => suiteLeaves(s.name).includes(name)).map((s) => s.name),
    variants: [...byHash.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .map(([hash, list]) => ({
        task_hash: hash,
        n_runs: list.length,
        primary_metric: task.primary,
        metric_meta: task.meta,
        num_fewshot: task.fewshot,
        limit: list[0].limit,
        split: task.split,
        n_instances: list[0].n,
        first_seen_at: list.map((t) => t.createdAt).sort()[0],
        last_seen_at: list.map((t) => t.createdAt).sort().pop()!,
        config: taskConfig(list[0]),
      })),
  };
}

function leaderboardFilter(q: Query) {
  const families = q.getAll("family").flatMap((v) => v.split(","));
  const groups = q.getAll("group").flatMap((v) => v.split(","));
  const users = q.getAll("user").flatMap((v) => v.split(","));
  const gpus = q.getAll("gpu_type").flatMap((v) => v.split(","));
  return (run: MockRun) =>
    (!gpus.length || gpus.includes(run.environment.gpu_type ?? "")) &&
    (!families.length || families.includes(run.model.family ?? "")) &&
    (!groups.length || groups.includes(run.summary.experiment_group ?? "")) &&
    (!users.length || users.includes(run.summary.author ?? ""));
}

export function taskLeaderboard(q: Query): T.LeaderboardResponse {
  const name = q.get("task") ?? "";
  const task = TASK_BY_NAME.get(name);
  if (!task) throw new MockHttpError(404, "not_found", `Task ${name} not found.`);
  const hash = q.get("hash");
  const metric = q.get("metric") ?? "primary";
  const perModel = q.get("per_model") ?? "latest";
  const keep = leaderboardFilter(q);
  const meta = task.meta[metric === "primary" ? task.primary : metric] ?? null;
  const hib = meta?.higher_is_better !== false;
  const trs = perModel === "all"
    ? [...getWorld().taskResults.values()].filter((t) => t.task.name === name && !t.error && (!hash || t.taskHash === hash))
    : latestPerModel(name, hash, keep);
  const z = 1.96;
  const rows = trs
    .map((t) => ({ t, run: getRun(t.runId), s: taskScore(t) }))
    .filter((x) => keep(x.run))
    .map((x) => ({ ...x, v: metricValue(x.t, metric) }))
    .filter((x) => x.v != null);
  rows.sort((a, b) => (hib ? b.v! - a.v! : a.v! - b.v!));
  const leader = rows[0];
  const leaderCi = leader ? [leader.v! - z * (leader.s.stderr ?? 0), leader.v! + z * (leader.s.stderr ?? 0)] : null;
  const items: T.LeaderboardRow[] = rows.map((x, i) => {
    const se = metric === "primary" ? x.s.stderr : null;
    const lo = se != null ? x.v! - z * se : null;
    const hi = se != null ? x.v! + z * se : null;
    return {
      rank: i + 1,
      subject: `m:${x.run.model.model_id}`,
      model: x.run.summary.model,
      run_id: x.run.summary.run_id,
      task_result_id: x.t.id,
      task_hash: x.t.taskHash,
      score: x.v,
      stderr: se,
      ci_low: lo,
      ci_high: hi,
      n: x.s.n,
      date: x.run.summary.created_at,
      experiment_group: x.run.summary.experiment_group,
      author: x.run.summary.author,
      tied_with_leader: i > 0 && leaderCi != null && lo != null && hi != null && (hib ? hi >= leaderCi[0] : lo <= leaderCi[1]),
      child_scores: null,
      runtime: taskRuntime(x.t),
      gpu_type: x.run.environment.gpu_type,
      gpu_count: x.run.environment.gpu_count,
    };
  });
  const { page, next_cursor, total } = paginate(items, q, 100, 1000);
  return { kind: "task", name, task_hash: hash, metric, meta, children: null, items: page, next_cursor, total };
}

function runtimeStats(values: (number | null)[]): T.RuntimeStats {
  const v = values.filter((x): x is number => x != null).sort((a, b) => a - b);
  return {
    n: v.length,
    median: v.length ? quantile(v, 0.5) : null,
    p90: v.length ? quantile(v, 0.9) : null,
    min: v.length ? v[0] : null,
    max: v.length ? v[v.length - 1] : null,
  };
}

export function taskRuntimeSummary(name: string, q: Query): T.TaskRuntimeResponse {
  const task = TASK_BY_NAME.get(name);
  if (!task) throw new MockHttpError(404, "not_found", `Task ${name} not found.`);
  const all = [...getWorld().taskResults.values()].filter((t) => t.task.name === name && !t.error);
  const counts = new Map<string, number>();
  all.forEach((t) => counts.set(t.taskHash, (counts.get(t.taskHash) ?? 0) + 1));
  const hash = q.get("hash") ?? [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  const keep = leaderboardFilter(q);
  const finalized = all
    .filter((t) => t.taskHash === hash)
    .map((t) => ({ t, run: getRun(t.runId) }))
    .filter((x) => x.run.summary.status !== "running" && x.run.summary.upload_state === "complete");
  const gpuCounts = new Map<string, number>();
  finalized.forEach((x) => x.run.environment.gpu_type && gpuCounts.set(x.run.environment.gpu_type, (gpuCounts.get(x.run.environment.gpu_type) ?? 0) + 1));
  const rows = finalized.filter((x) => keep(x.run)).sort((a, b) => b.run.summary.created_at.localeCompare(a.run.summary.created_at));
  const recorded = rows.filter((x) => x.t.runtime.basis !== "not_recorded");
  const groups = new Map<string, typeof recorded>();
  for (const x of recorded) {
    const key = `${x.run.model.model_id}|${x.run.environment.gpu_type}|${x.run.environment.gpu_count}`;
    groups.set(key, [...(groups.get(key) ?? []), x]);
  }
  const modelRows: T.TaskRuntimeModelRow[] = [...groups.values()].map((list) => {
    const rts = list.map((x) => taskRuntime(x.t));
    return {
      model: list[0].run.summary.model,
      gpu_type: list[0].run.environment.gpu_type,
      gpu_count: list[0].run.environment.gpu_count,
      runs: list.length,
      inference: runtimeStats(rts.map((r) => r.inference_seconds)),
      with_startup: runtimeStats(rts.map((r) => r.with_startup_seconds)),
      seconds_per_1k_instances: runtimeStats(rts.map((r) => r.seconds_per_1k_instances)),
      latest_score: taskScore(list[0].t).score,
    };
  });
  modelRows.sort((a, b) => (a.inference.median ?? Infinity) - (b.inference.median ?? Infinity));
  return {
    task_name: name,
    task_hash: hash,
    meta: task.meta[task.primary] ?? null,
    gpu_types: [...gpuCounts.entries()].sort((a, b) => b[1] - a[1]).map(([g]) => g),
    rows: modelRows,
    points: recorded.slice(0, 2000).map((x) => ({
      run_id: x.run.summary.run_id,
      task_result_id: x.t.id,
      model: x.run.summary.model,
      date: x.run.summary.created_at,
      gpu_type: x.run.environment.gpu_type,
      gpu_count: x.run.environment.gpu_count,
      score: taskScore(x.t).score,
      n: x.t.n,
      runtime: taskRuntime(x.t),
    })),
    not_recorded: rows.length - recorded.length,
  };
}

export function suitesList(q: Query): T.SuitesListResponse {
  const w = getWorld();
  const text = q.get("q")?.toLowerCase();
  let rows: T.SuiteRow[] = SUITES.map((s) => {
    const leaves = suiteLeaves(s.name);
    const runs = w.runs.filter((r) => visible(r) && runTRs(r).some((t) => leaves.includes(t.task.name)));
    return {
      suite_name: s.name,
      aggregation: s.aggregation,
      description: s.description,
      n_children: s.children.length,
      n_tasks: leaves.length,
      n_runs: runs.length,
      last_run_at: runs.map((r) => r.summary.created_at).sort().pop() ?? null,
    };
  });
  if (text) rows = rows.filter((r) => r.suite_name.toLowerCase().includes(text));
  const { page, next_cursor, total } = paginate(rows, q);
  return { items: page, next_cursor, total };
}

function tree(name: string): T.SuiteTreeNode {
  const def = SUITE_BY_NAME.get(name)!;
  return {
    type: "suite",
    name,
    aggregation: def.aggregation,
    children: def.children.map((c) => (c.type === "suite" ? tree(c.name) : { type: "task", name: c.name, aggregation: null, children: [] })),
  };
}

export function suiteDetail(q: Query): T.SuiteDetailResponse {
  const name = q.get("suite") ?? "";
  const def = SUITE_BY_NAME.get(name);
  if (!def) throw new MockHttpError(404, "not_found", `Suite ${name} not found.`);
  return {
    suite_name: name,
    aggregation: def.aggregation,
    description: def.description,
    definition_hash: def.definition_hash,
    definitions_seen: 1,
    tree: tree(name),
    n_runs: suitesList({ get: () => null, getAll: () => [] }).items.find((s) => s.suite_name === name)?.n_runs ?? 0,
  };
}

export function suiteLeaderboard(q: Query): T.LeaderboardResponse {
  const name = q.get("suite") ?? "";
  const def = SUITE_BY_NAME.get(name);
  if (!def) throw new MockHttpError(404, "not_found", `Suite ${name} not found.`);
  const keep = leaderboardFilter(q);
  const w = getWorld();
  const seen = new Set<string>();
  const rows: { key: string; run: MockRun; s: ReturnType<typeof suiteScore>; children: Record<string, number | null> }[] = [];
  for (const run of w.runs) {
    if (!visible(run) || run.summary.status === "running" || !keep(run)) continue;
    if (seen.has(run.model.model_id)) continue;
    const r = resolveSubject(`m:${run.model.model_id}`)!;
    const s = suiteScore(name, (t) => r.lookup.get(t));
    if (s.score == null) continue;
    seen.add(run.model.model_id);
    const children: Record<string, number | null> = {};
    for (const c of def.children) {
      if (c.type === "task") {
        const t = r.lookup.get(c.name);
        children[`task:${c.name}`] = t && !t.error ? taskScore(t).score : null;
      } else children[`suite:${c.name}`] = suiteScore(c.name, (t) => r.lookup.get(t)).score;
    }
    rows.push({ key: r.info.key, run, s, children });
  }
  rows.sort((a, b) => b.s.score! - a.s.score!);
  const leader = rows[0];
  const items: T.LeaderboardRow[] = rows.map((x, i) => ({
    rank: i + 1,
    subject: x.key,
    model: x.run.summary.model,
    run_id: x.run.summary.run_id,
    task_result_id: null,
    task_hash: null,
    score: x.s.score,
    stderr: x.s.stderr,
    ci_low: x.s.score! - 1.96 * (x.s.stderr ?? 0),
    ci_high: x.s.score! + 1.96 * (x.s.stderr ?? 0),
    n: x.s.nInstances,
    date: x.run.summary.created_at,
    experiment_group: x.run.summary.experiment_group,
    author: x.run.summary.author,
    tied_with_leader: i > 0 && x.s.score! + 1.96 * (x.s.stderr ?? 0) >= leader.s.score! - 1.96 * (leader.s.stderr ?? 0),
    child_scores: x.children,
    runtime: null,
    gpu_type: null,
    gpu_count: null,
  }));
  const { page, next_cursor, total } = paginate(items, q, 100, 1000);
  return {
    kind: "suite",
    name,
    task_hash: null,
    metric: "primary",
    meta: { kind: "bounded", higher_is_better: true, display_format: "percent", unit: null },
    children: def.children,
    items: page,
    next_cursor,
    total,
  };
}

// ------------------------------------------------------------------------------------- groups, home

function groupRuns(name: string): MockRun[] {
  return getWorld().runs.filter((r) => r.summary.experiment_group === name && visible(r));
}

function groupHeatmap(name: string): T.MiniHeatmap {
  const runs = groupRuns(name);
  const models: MockRun[] = [];
  const seen = new Set<string>();
  for (const r of runs) {
    if (seen.has(r.model.model_id) || !r.taskResultIds.length) continue;
    seen.add(r.model.model_id);
    models.push(r);
  }
  const names = new Set(runs.flatMap((r) => runTRs(r).map((t) => t.task.name)));
  const cols = suitesFor(names).slice(0, 8);
  const rows = models.slice(0, 8);
  return {
    rows: rows.map((r) => ({ key: `m:${r.model.model_id}`, label: baseLabel(r.summary.model) })),
    columns: cols.map((s) => ({ key: `suite:${s.name}`, label: s.name })),
    values: rows.map((r) => cols.map((c) => suiteScore(c.name, runLookup(r)).score)),
  };
}

function groupRow(name: string, withHeatmap: boolean): T.GroupRow {
  const runs = groupRuns(name);
  const dates = runs.map((r) => r.summary.created_at).sort();
  return {
    name,
    n_runs: runs.length,
    n_models: new Set(runs.map((r) => r.model.model_id)).size,
    n_tasks: new Set(runs.flatMap((r) => runTRs(r).map((t) => t.task.name))).size,
    users: [...new Set(runs.map((r) => r.summary.author ?? ""))].filter(Boolean),
    first_run_at: dates[0],
    last_run_at: dates[dates.length - 1],
    heatmap: withHeatmap ? groupHeatmap(name) : null,
  };
}

export function groupsList(q: Query): T.GroupsListResponse {
  const names = [...new Set(getWorld().runs.map((r) => r.summary.experiment_group).filter((g): g is string => !!g))];
  const text = q.get("q")?.toLowerCase();
  const activeDays = q.get("active_days");
  let rows = names.map((n) => groupRow(n, q.get("include_heatmap") === "true"));
  if (text) rows = rows.filter((r) => r.name.toLowerCase().includes(text));
  if (activeDays) rows = rows.filter((r) => new Date(r.last_run_at).getTime() >= NOW - Number(activeDays) * DAY);
  rows.sort((a, b) => b.last_run_at.localeCompare(a.last_run_at));
  const { page, next_cursor, total } = paginate(rows, q);
  return { items: page, next_cursor, total };
}

export function groupDetail(q: Query): T.GroupDetailResponse {
  const name = q.get("group") ?? "";
  const runs = groupRuns(name);
  if (!runs.length) throw new MockHttpError(404, "not_found", `Group ${name} not found.`);
  const row = groupRow(name, false);
  const modelIds: string[] = [];
  for (const r of [...runs].sort((a, b) => (a.model.step ?? 1e12) - (b.model.step ?? 1e12))) {
    if (!modelIds.includes(r.model.model_id) && r.taskResultIds.length) modelIds.push(r.model.model_id);
  }
  const subjects = modelIds.map((id) => resolveSubject(`m:${id}`, name)!);
  const tasks = [...new Set(runs.flatMap((r) => runTRs(r).map((t) => t.task.name)))];
  const order = scopeRows("all", subjects).filter((r) => r.kind === "task").map((r) => r.name);
  const sortedTasks = order.filter((t) => tasks.includes(t));
  const coverage: T.CoverageCell[] = [];
  for (const s of subjects) {
    for (const t of sortedTasks) {
      const found = s.lookup.get(t);
      coverage.push({
        subject: s.info.key,
        task_name: t,
        status: !found ? "missing" : found.error ? "failed" : "ok",
        task_result_id: found?.id ?? null,
        run_id: found?.runId ?? null,
      });
    }
  }
  return {
    name,
    n_runs: row.n_runs,
    n_models: row.n_models,
    n_tasks: row.n_tasks,
    users: row.users,
    first_run_at: row.first_run_at,
    last_run_at: row.last_run_at,
    subjects: subjects.map((s) => s.info),
    tasks: sortedTasks,
    coverage,
  };
}

export function activity(q: Query): T.ActivityResponse {
  const user = q.get("user");
  const groups = new Map<string, MockRun[]>();
  for (const run of getWorld().runs) {
    if (user && run.summary.author !== (user === "me" ? ME : user)) continue;
    const key = run.summary.launch_id ?? `run:${run.summary.run_id}`;
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  const rank: Record<string, number> = { failed: 3, partial: 2, running: 1, complete: 0 };
  const items: T.ActivityItem[] = [...groups.entries()].map(([key, runs]) => {
    const worst = runs.map((r) => r.summary.status).sort((a, b) => rank[b] - rank[a])[0];
    const err = runs.flatMap((r) => r.errors)[0];
    return {
      key,
      launch_id: runs[0].summary.launch_id,
      user: runs[0].summary.author,
      created_at: runs.map((r) => r.summary.created_at).sort().pop()!,
      runs: runs.map((r) => r.summary),
      n_tasks: runs.reduce((a, r) => a + r.summary.num_tasks, 0),
      n_failed_tasks: runs.reduce((a, r) => a + r.summary.num_failed_tasks, 0),
      status: worst,
      failure_reason: err ? `${err.task ? `${err.task}: ` : ""}${err.error}` : null,
    };
  });
  items.sort((a, b) => b.created_at.localeCompare(a.created_at));
  const { page, next_cursor } = paginate(items, q, 20, 100);
  return { items: page, next_cursor };
}

export function statsSummary(q: Query): T.StatsSummaryResponse {
  const days = Number(q.get("window_days") ?? 7);
  const daily: T.DailyStat[] = [];
  let runsTotal = 0;
  let instancesTotal = 0;
  const modelsAll = new Set<string>();
  for (let d = days - 1; d >= 0; d--) {
    const start = NOW - (d + 1) * DAY;
    const end = NOW - d * DAY;
    const runs = getWorld().runs.filter((r) => {
      const t = new Date(r.summary.created_at).getTime();
      return t >= start && t < end;
    });
    const models = new Set(runs.map((r) => r.model.model_id));
    models.forEach((m) => modelsAll.add(m));
    const instances = runs.reduce((a, r) => a + r.summary.num_instances, 0);
    runsTotal += runs.length;
    instancesTotal += instances;
    daily.push({ date: new Date(end).toISOString().slice(0, 10), runs: runs.length, models: models.size, instances });
  }
  return { window_days: days, runs: runsTotal, models: modelsAll.size, instances: instancesTotal, daily };
}

// ------------------------------------------------------------------------------------- views

const views: T.SavedView[] = [
  { id: "v1", name: "7B midtrain, OLMES + GSM8K", owner_email: "chrisg@allenai.org", shared: true, page: "runs", query: "series=olmo3-7b-midtrain&cols=suite:olmes:base,task:gsm8k:cot:olmo3&sort=-step", created_at: new Date(NOW - 5 * DAY).toISOString(), updated_at: new Date(NOW - 2 * DAY).toISOString() },
  { id: "v2", name: "My partial runs", owner_email: "chrisg@allenai.org", shared: false, page: "runs", query: "user=me&status=partial,failed", created_at: new Date(NOW - 9 * DAY).toISOString(), updated_at: new Date(NOW - 9 * DAY).toISOString() },
  { id: "v3", name: "Safety sweep leaderboard", owner_email: "maliam@allenai.org", shared: true, page: "runs", query: "group=tulu4-dpo-safety&cols=suite:safety:olmo3,task:ifeval&sort=-score:suite:safety:olmo3", created_at: new Date(NOW - 4 * DAY).toISOString(), updated_at: new Date(NOW - 1 * DAY).toISOString() },
  { id: "v4", name: "32B SFT math", owner_email: "raj@allenai.org", shared: true, page: "runs", query: "series=olmo3-32b-sft&cols=suite:math:olmo3,task:aime24", created_at: new Date(NOW - 6 * DAY).toISOString(), updated_at: new Date(NOW - 6 * DAY).toISOString() },
];

export function listViews(q: Query): T.SavedViewsResponse {
  const page = q.get("page");
  const mine = views.filter((v) => v.owner_email === `${ME}@allenai.org` && (!page || v.page === page));
  const shared = views.filter((v) => v.shared && v.owner_email !== `${ME}@allenai.org` && (!page || v.page === page));
  return { mine, shared };
}

export function createView(body: T.CreateViewRequest): T.SavedView {
  const view: T.SavedView = {
    id: hexId(8, "view", body.name, views.length),
    name: body.name,
    owner_email: `${ME}@allenai.org`,
    shared: body.shared,
    page: body.page,
    query: body.query,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  views.unshift(view);
  return view;
}

export function updateView(id: string, body: T.UpdateViewRequest): T.SavedView {
  const view = views.find((v) => v.id === id);
  if (!view) throw new MockHttpError(404, "not_found", "View not found.");
  if (view.owner_email !== `${ME}@allenai.org`) throw new MockHttpError(403, "forbidden", "Only the owner can edit this view.");
  Object.assign(view, body, { updated_at: new Date().toISOString() });
  return view;
}

export function deleteView(id: string): void {
  const idx = views.findIndex((v) => v.id === id);
  if (idx < 0) throw new MockHttpError(404, "not_found", "View not found.");
  if (views[idx].owner_email !== `${ME}@allenai.org`) throw new MockHttpError(403, "forbidden", "Only the owner can delete this view.");
  views.splice(idx, 1);
}

// ------------------------------------------------------------------------------------- search

export function search(q: Query): T.SearchResponse {
  const query = (q.get("q") ?? "").trim();
  const limit = Math.min(20, Number(q.get("limit") ?? 5));
  const tokens = query.toLowerCase().split(/[\s:]+/).filter(Boolean);
  const match = (text: string) => tokens.every((t) => text.toLowerCase().includes(t));
  const w = getWorld();
  const groups: T.SearchGroup[] = [];
  const ids: T.SearchResult[] = [];
  if (/^[0-9A-Z]{26}$/i.test(query)) {
    const run = w.runs.find((r) => [r.beaker?.experiment_id, r.beaker?.job_id, r.beaker?.result_dataset_id].includes(query.toUpperCase()));
    if (run) ids.push({ type: "id", key: run.summary.run_id, label: baseLabel(run.summary.model), secondary: `Beaker ID → run ${run.summary.run_id}`, href: `/runs/${run.summary.run_id}`, subject: `r:${run.summary.run_id}` });
  }
  if (/^[a-z0-9]{12}$/.test(query) && w.runById.has(query)) {
    const run = w.runById.get(query)!;
    ids.push({ type: "id", key: query, label: baseLabel(run.summary.model), secondary: "Run ID", href: `/runs/${query}`, subject: `r:${query}` });
  }
  if (query.startsWith("gs://")) {
    const run = w.runs.find((r) => query.startsWith(r.summary.gcs_prefix) || r.summary.gcs_prefix.startsWith(query));
    if (run) ids.push({ type: "id", key: run.summary.run_id, label: baseLabel(run.summary.model), secondary: "GCS prefix", href: `/runs/${run.summary.run_id}`, subject: `r:${run.summary.run_id}` });
  }
  if (/^[0-9a-f]{7,40}$/.test(query)) {
    const run = w.runs.find((r) => (r.summary.git_commit ?? "").startsWith(query));
    if (run) ids.push({ type: "id", key: run.summary.run_id, label: baseLabel(run.summary.model), secondary: `Commit ${query.slice(0, 7)}`, href: `/runs/${run.summary.run_id}`, subject: `r:${run.summary.run_id}` });
  }
  if (ids.length) groups.push({ type: "id", items: ids.slice(0, limit) });
  const runs = w.runs
    .filter((r) => visible(r) && match([r.summary.model.name, r.summary.experiment_name, r.summary.experiment_group, ...r.summary.tags].join(" ")))
    .slice(0, limit)
    .map((r): T.SearchResult => ({
      type: "run",
      key: r.summary.run_id,
      label: baseLabel(r.summary.model),
      secondary: [r.summary.experiment_group, r.summary.author, new Date(r.summary.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric" })].filter(Boolean).join(" · "),
      href: `/runs/${r.summary.run_id}`,
      subject: `r:${r.summary.run_id}`,
    }));
  if (runs.length) groups.push({ type: "run", items: runs });
  const series = modelsList({ get: (k) => (k === "limit" ? "500" : null), getAll: () => [] }).items.filter((m) => match(`${m.series} ${m.family}`)).slice(0, limit);
  if (series.length) {
    groups.push({
      type: "model",
      items: series.map((m) => ({
        type: "model",
        key: m.series,
        label: m.series_label,
        secondary: `${m.n_runs} runs${m.latest_step != null ? ` · latest step ${m.latest_step.toLocaleString("en-US")}` : ""}`,
        href: `/models/${encodeURIComponent(m.series)}`,
        subject: m.latest_run ? `m:${m.latest_run.model.model_id}` : null,
      })),
    });
  }
  const tasks = TASKS.filter((t) => match(t.name)).slice(0, limit);
  if (tasks.length) {
    groups.push({
      type: "task",
      items: tasks.map((t) => ({ type: "task", key: t.name, label: t.name, secondary: t.suite ?? "No suite", href: `/tasks/${encodeURIComponent(t.name)}`, subject: null })),
    });
  }
  const suites = SUITES.filter((s) => match(s.name)).slice(0, limit);
  if (suites.length) {
    groups.push({
      type: "suite",
      items: suites.map((s) => ({ type: "suite", key: s.name, label: s.name, secondary: `${suiteLeaves(s.name).length} tasks · ${s.aggregation.replace(/_/g, " ")}`, href: `/suites/${encodeURIComponent(s.name)}`, subject: null })),
    });
  }
  const groupNames = [...new Set(w.runs.map((r) => r.summary.experiment_group).filter((g): g is string => !!g))].filter(match).slice(0, limit);
  if (groupNames.length) {
    groups.push({
      type: "group",
      items: groupNames.map((g) => {
        const row = groupRow(g, false);
        return { type: "group", key: g, label: g, secondary: `${row.n_runs} runs · ${row.n_models} models`, href: `/groups/${encodeURIComponent(g)}`, subject: null };
      }),
    });
  }
  const users = w.users.filter(match).slice(0, limit);
  if (users.length) {
    groups.push({
      type: "user",
      items: users.map((u) => ({ type: "user", key: u, label: u, secondary: `${w.runs.filter((r) => r.summary.author === u).length} runs`, href: `/runs?user=${u}`, subject: null })),
    });
  }
  return { query, groups };
}
