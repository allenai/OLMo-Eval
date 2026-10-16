import { delay, http, HttpResponse, type HttpResponseResolver } from "msw";
import * as api from "./api";
import { MockHttpError, type Query } from "./api";

let requestCounter = 0;

function requestId(): string {
  requestCounter += 1;
  return `mock-${Date.now().toString(36)}-${requestCounter.toString(36).padStart(4, "0")}`;
}

/** Simulated network latency; zero in tests. */
const LATENCY = import.meta.env.MODE === "test" ? 0 : 140;

type Handler = (ctx: { query: Query; params: Record<string, string>; body: unknown }) => unknown;

function wrap(fn: Handler): HttpResponseResolver {
  return async ({ request, params }) => {
    const id = requestId();
    if (LATENCY) await delay(LATENCY + Math.random() * 120);
    const url = new URL(request.url);
    const query: Query = { get: (k) => url.searchParams.get(k), getAll: (k) => url.searchParams.getAll(k) };
    let body: unknown = null;
    if (request.method !== "GET" && request.method !== "DELETE") {
      try {
        body = await request.json();
      } catch {
        body = null;
      }
    }
    try {
      const result = fn({ query, params: params as Record<string, string>, body });
      if (result === undefined) return new HttpResponse(null, { status: 204, headers: { "X-Request-Id": id } });
      return HttpResponse.json(result as Record<string, unknown>, { headers: { "X-Request-Id": id } });
    } catch (err) {
      const status = err instanceof MockHttpError ? err.status : 500;
      const code = err instanceof MockHttpError ? err.code : "internal";
      const message = err instanceof Error ? err.message : "Internal error";
      if (!(err instanceof MockHttpError)) console.error(err);
      return HttpResponse.json({ error: { code, message, request_id: id } }, { status, headers: { "X-Request-Id": id } });
    }
  };
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export const handlers = [
  http.get("/api/health", wrap(() => api.health())),
  http.get("/api/me", wrap(() => api.me())),
  http.get("/api/search", wrap(({ query }) => api.search(query))),
  http.get("/api/runs", wrap(({ query }) => api.runsList(query))),
  http.get("/api/runs/facets", wrap(({ query }) => api.runsFacets(query))),
  http.post("/api/runs/tags", wrap(({ body }) => api.bulkTag(body as any))),
  http.get("/api/runs/:runId", wrap(({ params }) => api.runDetail(params.runId))),
  http.patch("/api/runs/:runId", wrap(({ params, body }) => api.patchRun(params.runId, body as any))),
  http.get("/api/runs/:runId/task-results", wrap(({ params, query }) => api.runTaskResults(params.runId, query))),
  http.get("/api/runs/:runId/suites", wrap(({ params, query }) => api.runSuites(params.runId, query))),
  http.get(
    "/api/runs/:runId/task-results/:trId/instances",
    wrap(({ params, query }) => api.instances(params.runId, Number(params.trId), query)),
  ),
  http.get(
    "/api/runs/:runId/task-results/:trId/histograms",
    wrap(({ params, query }) => api.histograms(params.runId, Number(params.trId), query)),
  ),
  http.get("/api/instances/detail", wrap(({ query }) => api.instanceDetail(query))),
  http.get("/api/runs/:runId/inference", wrap(({ params, query }) => api.inference(params.runId, query))),
  http.get("/api/runs/:runId/configs", wrap(({ params }) => api.runConfigs(params.runId))),
  http.get("/api/runs/:runId/artifacts", wrap(({ params }) => api.artifacts(params.runId))),
  http.get("/api/runs/:runId/baseline-suggestions", wrap(({ params }) => api.baselineSuggestions(params.runId))),
  http.post("/api/artifacts/sign", wrap(({ body }) => api.signDownload(body as any))),
  http.post("/api/subjects/resolve", wrap(({ body }) => api.subjectsResolve(body as any))),
  http.get("/api/subjects/task-result", wrap(({ query }) => api.subjectTaskResult(query))),
  http.post("/api/compare/matrix", wrap(({ body }) => api.compareMatrix(body as any))),
  http.post("/api/compare/pairwise", wrap(({ body }) => api.comparePairwise(body as any))),
  http.post("/api/compare/contingency", wrap(({ body }) => api.compareContingency(body as any))),
  http.post("/api/compare/instances", wrap(({ body }) => api.compareInstances(body as any))),
  http.post("/api/tasks/distributions", wrap(({ body }) => api.distributions(body as any))),
  http.get("/api/models", wrap(({ query }) => api.modelsList(query))),
  http.get("/api/models/series", wrap(({ query }) => api.modelSeries(query))),
  http.get("/api/models/progression", wrap(({ query }) => api.progression(query))),
  http.get("/api/tasks", wrap(({ query }) => api.tasksList(query))),
  http.get("/api/tasks/detail", wrap(({ query }) => api.taskDetail(query))),
  http.get("/api/tasks/leaderboard", wrap(({ query }) => api.taskLeaderboard(query))),
  http.get("/api/tasks/:taskName/runtime", wrap(({ params, query }) => api.taskRuntimeSummary(decodeURIComponent(params.taskName), query))),
  http.get("/api/suites", wrap(({ query }) => api.suitesList(query))),
  http.get("/api/suites/detail", wrap(({ query }) => api.suiteDetail(query))),
  http.get("/api/suites/leaderboard", wrap(({ query }) => api.suiteLeaderboard(query))),
  http.get("/api/groups", wrap(({ query }) => api.groupsList(query))),
  http.get("/api/groups/detail", wrap(({ query }) => api.groupDetail(query))),
  http.get("/api/activity", wrap(({ query }) => api.activity(query))),
  http.get("/api/stats/summary", wrap(({ query }) => api.statsSummary(query))),
  http.get("/api/views", wrap(({ query }) => api.listViews(query))),
  http.post("/api/views", wrap(({ body }) => api.createView(body as any))),
  http.patch("/api/views/:viewId", wrap(({ params, body }) => api.updateView(params.viewId, body as any))),
  http.delete("/api/views/:viewId", wrap(({ params }) => api.deleteView(params.viewId))),
];
/* eslint-enable @typescript-eslint/no-explicit-any */
