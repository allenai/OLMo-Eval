/**
 * The mock API must produce exactly what the real API promises, so every mock response is
 * validated against the generated contract schema (dashboard/contract/api-v1.schema.json).
 */
import Ajv from "ajv";
import addFormats from "ajv-formats";
import { describe, expect, it } from "vitest";
import schema from "../../../contract/api-v1.schema.json";
import * as api from "./api";
import { getWorld } from "./fixtures";

const ajv = new Ajv({ strict: false, allErrors: true });
addFormats(ajv);
ajv.addSchema(schema, "api");

function check(definition: string, value: unknown) {
  const validate = ajv.getSchema(`api#/definitions/${definition}`);
  if (!validate) throw new Error(`no definition ${definition}`);
  const ok = validate(JSON.parse(JSON.stringify(value)));
  if (!ok) throw new Error(`${definition}: ${ajv.errorsText(validate.errors, { separator: "\n" })}`);
  expect(ok).toBe(true);
}

const q = (params: Record<string, string | string[]> = {}): api.Query => ({
  get: (k) => {
    const v = params[k];
    return v == null ? null : Array.isArray(v) ? v[0] : v;
  },
  getAll: (k) => {
    const v = params[k];
    return v == null ? [] : Array.isArray(v) ? v : [v];
  },
});

describe("mock API matches the contract", () => {
  const world = getWorld();
  const run = world.runs.find((r) => r.summary.status === "partial")!;
  const runId = run.summary.run_id;
  const other = world.runs.find((r) => r.summary.experiment_group === run.summary.experiment_group && r.summary.run_id !== runId && r.taskResultIds.length > 30)!;
  const baseline = `r:${other.summary.run_id}`;
  const tr = world.taskResults.get(run.taskResultIds.find((id) => !world.taskResults.get(id)!.error)!)!;
  const group = run.summary.experiment_group!;
  const subjects = api.groupDetail(q({ group })).subjects.map((s) => s.key).slice(0, 4);

  it("basics, search and lists", () => {
    check("MeResponse", api.me());
    check("HealthResponse", api.health());
    check("SearchResponse", api.search(q({ q: "olmo3" })));
    check("SearchResponse", api.search(q({ q: runId })));
    check("RunsListResponse", api.runsList(q({ cols: "suite:olmes:base,task:gsm8k:cot:olmo3", baseline, model: "olmo3*" })));
    check("RunsFacetsResponse", api.runsFacets(q({ user: "me" })));
    check("ModelsListResponse", api.modelsList(q()));
    check("TasksListResponse", api.tasksList(q()));
    check("SuitesListResponse", api.suitesList(q()));
    check("GroupsListResponse", api.groupsList(q({ include_heatmap: "true" })));
    check("ActivityResponse", api.activity(q()));
    check("StatsSummaryResponse", api.statsSummary(q()));
    check("SavedViewsResponse", api.listViews(q({ page: "runs" })));
  });

  it("run detail endpoints", () => {
    check("RunDetail", api.runDetail(runId));
    check("RunTaskResultsResponse", api.runTaskResults(runId, q({ baseline })));
    check("RunSuitesResponse", api.runSuites(runId, q({ baseline })));
    check("InstancesResponse", api.instances(runId, tr.id, q({ baseline, sort: "-length" })));
    const native = api.instances(runId, tr.id, q()).items[0].native_id;
    check("InstanceDetailResponse", api.instanceDetail(q({ task_result_id: String(tr.id), native_id: native })));
    check("HistogramsResponse", api.histograms(runId, tr.id, q({ baseline })));
    check("InferenceResponse", api.inference(runId, q({ baseline })));
    check("RunConfigsResponse", api.runConfigs(runId));
    check("ArtifactsResponse", api.artifacts(runId));
    check("BaselineSuggestionsResponse", api.baselineSuggestions(runId));
    check("SignDownloadResponse", api.signDownload({ gs_uri: `${run.summary.gcs_prefix}metrics.json` }));
  });

  it("instance details for every task type", () => {
    for (const t of world.runs.find((r) => r.model.name.includes("Qwen"))!.taskResultIds.map((id) => world.taskResults.get(id)!)) {
      const native = api.instances(t.runId, t.id, q({ limit: "1" })).items[0]?.native_id;
      if (native) check("InstanceDetailResponse", api.instanceDetail(q({ task_result_id: String(t.id), native_id: native })));
    }
  });

  it("compare endpoints", () => {
    check("ResolveSubjectsResponse", api.subjectsResolve({ subjects: [...subjects, "r:nonexistent"] }));
    check("SubjectTaskResultResponse", api.subjectTaskResult(q({ subject: subjects[0], task: "gsm8k:cot:olmo3" })));
    check("MatrixResponse", api.compareMatrix({ subjects, scope: "all", metric: "primary", baseline: subjects[0] }));
    check("MatrixResponse", api.compareMatrix({ subjects, scope: "suite:olmes:base", metric: "primary" }));
    check("MatrixResponse", api.compareMatrix({ subjects, scope: "suite:olmes:base", metric: "runtime:inference", baseline: subjects[0] }));
    check("MatrixResponse", api.compareMatrix({ subjects, scope: "all", metric: "runtime:with_startup" }));
    expect(() => api.comparePairwise({ subjects, scope: "all", metric: "runtime:inference" })).toThrow(api.MockHttpError);
    check("PairwiseResponse", api.comparePairwise({ subjects, scope: "suite:olmes:base", metric: "primary" }));
    check("ContingencyResponse", api.compareContingency({ a: subjects[0], b: subjects[1], scope: "all", metric: "primary" }));
    check(
      "CompareInstancesResponse",
      api.compareInstances({ subjects, task_name: "gsm8k:cot:olmo3", metric: "primary", filter: "disagree", include_previews: true, limit: 20 }),
    );
    check("DistributionsResponse", api.distributions({ items: [{ task_name: "gsm8k:cot:olmo3", task_hash: null, metric: "primary" }] }));
  });

  it("models, tasks, suites and groups", () => {
    const series = run.model.series;
    check("ModelSeriesDetailResponse", api.modelSeries(q({ series })));
    check("ProgressionResponse", api.progression(q({ series, references: baseline })));
    check("TaskDetailResponse", api.taskDetail(q({ task: "gsm8k:cot:olmo3" })));
    check("LeaderboardResponse", api.taskLeaderboard(q({ task: "gsm8k:cot:olmo3" })));
    check("LeaderboardResponse", api.taskLeaderboard(q({ task: "gsm8k:cot:olmo3", gpu_type: "NVIDIA A100-SXM4-80GB" })));
    check("TaskRuntimeResponse", api.taskRuntimeSummary("gsm8k:cot:olmo3", q()));
    check("TaskRuntimeResponse", api.taskRuntimeSummary("ifeval", q({ gpu_type: "NVIDIA H100 80GB HBM3" })));
    check("SuiteDetailResponse", api.suiteDetail(q({ suite: "olmes:base" })));
    check("LeaderboardResponse", api.suiteLeaderboard(q({ suite: "olmes:base" })));
    check("GroupDetailResponse", api.groupDetail(q({ group })));
  });

  it("returns typed errors for unknown entities", () => {
    expect(() => api.runDetail("doesnotexist")).toThrow(api.MockHttpError);
    expect(() => api.compareMatrix({ subjects: ["r:missing00"], scope: "all", metric: "primary" })).toThrow(/Unknown subject/);
  });
});
