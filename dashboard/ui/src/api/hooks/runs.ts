import type {
  ArtifactsResponse,
  BaselineSuggestionsResponse,
  BulkTagRequest,
  BulkTagResponse,
  HistogramsResponse,
  InferenceResponse,
  InstanceDetailResponse,
  InstancesResponse,
  RunConfigsResponse,
  RunDetail,
  RunPatchRequest,
  RunsFacetsResponse,
  RunsListResponse,
  RunSuitesResponse,
  RunTaskResultsResponse,
  SignDownloadResponse,
} from "@contract/api-types";
import {
  keepPreviousData,
  type QueryClient,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { apiGet, apiPatch, apiPost } from "../client";

type Params = Record<string, unknown>;

export function useRunsList(params: Params, options: { enabled?: boolean } = {}) {
  return useQuery({
    queryKey: ["runs", params],
    queryFn: ({ signal }) => apiGet<RunsListResponse>("/runs", params, signal),
    placeholderData: keepPreviousData,
    enabled: options.enabled ?? true,
  });
}

export function useRunsInfinite(params: Params) {
  return useInfiniteQuery({
    queryKey: ["runs-infinite", params],
    queryFn: ({ pageParam, signal }) =>
      apiGet<RunsListResponse>("/runs", { ...params, cursor: pageParam ?? undefined }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    placeholderData: keepPreviousData,
  });
}

export function useRunsFacets(params: Params) {
  return useQuery({
    queryKey: ["runs-facets", params],
    queryFn: ({ signal }) => apiGet<RunsFacetsResponse>("/runs/facets", params, signal),
    placeholderData: keepPreviousData,
  });
}

/** API path of a run. */
export function runPath(runId: string): string {
  return `/runs/${encodeURIComponent(runId)}`;
}

/** Every cached query that lists runs or counts them, so tag edits show up everywhere. */
export function invalidateRunLists(qc: QueryClient): void {
  for (const key of ["runs", "runs-infinite", "runs-facets"]) {
    qc.invalidateQueries({ queryKey: [key] });
  }
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: ["run", runId],
    queryFn: ({ signal }) => apiGet<RunDetail>(runPath(runId), undefined, signal),
  });
}

export function useTaskResults(runId: string, baseline: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ["task-results", runId, baseline ?? null],
    queryFn: ({ signal }) =>
      apiGet<RunTaskResultsResponse>(`${runPath(runId)}/task-results`, { baseline }, signal),
    placeholderData: keepPreviousData,
    enabled,
  });
}

export function useRunSuites(runId: string, baseline: string | undefined, enabled = true) {
  return useQuery({
    queryKey: ["run-suites", runId, baseline ?? null],
    queryFn: ({ signal }) => apiGet<RunSuitesResponse>(`${runPath(runId)}/suites`, { baseline }, signal),
    placeholderData: keepPreviousData,
    enabled,
  });
}

export function useInstances(runId: string, taskResultId: number | null, params: Params) {
  return useInfiniteQuery({
    queryKey: ["instances", runId, taskResultId, params],
    queryFn: ({ pageParam, signal }) =>
      apiGet<InstancesResponse>(
        `${runPath(runId)}/task-results/${taskResultId}/instances`,
        { ...params, cursor: pageParam ?? undefined },
        signal,
      ),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    enabled: taskResultId != null,
    placeholderData: keepPreviousData,
  });
}

export function useInstanceDetail(taskResultId: number | null | undefined, nativeId: string | undefined) {
  return useQuery({
    queryKey: ["instance-detail", taskResultId, nativeId],
    queryFn: ({ signal }) =>
      apiGet<InstanceDetailResponse>(
        "/instances/detail",
        { task_result_id: taskResultId, native_id: nativeId },
        signal,
      ),
    enabled: taskResultId != null && !!nativeId,
    staleTime: 5 * 60_000,
  });
}

export function useHistograms(runId: string, taskResultId: number | null, baseline: string | undefined) {
  return useQuery({
    queryKey: ["histograms", runId, taskResultId, baseline ?? null],
    queryFn: ({ signal }) =>
      apiGet<HistogramsResponse>(
        `${runPath(runId)}/task-results/${taskResultId}/histograms`,
        { baseline },
        signal,
      ),
    enabled: taskResultId != null,
  });
}

export function useInference(runId: string, baseline: string | undefined) {
  return useQuery({
    queryKey: ["inference", runId, baseline ?? null],
    queryFn: ({ signal }) => apiGet<InferenceResponse>(`${runPath(runId)}/inference`, { baseline }, signal),
  });
}

export function useRunConfigs(runId: string | null | undefined) {
  return useQuery({
    queryKey: ["configs", runId],
    queryFn: ({ signal }) => apiGet<RunConfigsResponse>(`${runPath(runId ?? "")}/configs`, undefined, signal),
    enabled: !!runId,
    staleTime: 5 * 60_000,
  });
}

export function useArtifacts(runId: string) {
  return useQuery({
    queryKey: ["artifacts", runId],
    queryFn: ({ signal }) => apiGet<ArtifactsResponse>(`${runPath(runId)}/artifacts`, undefined, signal),
  });
}

export function useBaselineSuggestions(runId: string) {
  return useQuery({
    queryKey: ["baseline-suggestions", runId],
    queryFn: ({ signal }) =>
      apiGet<BaselineSuggestionsResponse>(`${runPath(runId)}/baseline-suggestions`, undefined, signal),
    staleTime: 5 * 60_000,
  });
}

export function signArtifact(gsUri: string): Promise<SignDownloadResponse> {
  return apiPost<SignDownloadResponse>("/artifacts/sign", { gs_uri: gsUri });
}

export function usePatchRun(runId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RunPatchRequest) => apiPatch<RunDetail>(runPath(runId), body),
    onSuccess: (data) => {
      qc.setQueryData(["run", runId], data);
      invalidateRunLists(qc);
    },
  });
}

export function useBulkTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: BulkTagRequest) => apiPost<BulkTagResponse>("/runs/tags", body),
    onSuccess: () => {
      invalidateRunLists(qc);
      qc.invalidateQueries({ queryKey: ["run"] });
    },
  });
}
