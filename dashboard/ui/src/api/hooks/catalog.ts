import type {
  ActivityResponse,
  CreateViewRequest,
  GroupDetailResponse,
  GroupsListResponse,
  LeaderboardResponse,
  MeResponse,
  ModelSeriesDetailResponse,
  ModelsListResponse,
  ProgressionResponse,
  SavedView,
  SavedViewsResponse,
  SearchResponse,
  StatsSummaryResponse,
  SuiteDetailResponse,
  SuitesListResponse,
  TaskDetailResponse,
  TaskRuntimeResponse,
  TasksListResponse,
  UpdateViewRequest,
} from "@contract/api-types";
import {
  keepPreviousData,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { apiDelete, apiGet, apiPatch, apiPost } from "../client";

type Params = Record<string, unknown>;

/**
 * A keyset-paginated list endpoint for an index table: rows loaded so far, the exact total, and
 * a loader for the next page.
 */
export function usePagedList<T>(path: string, params: Params, pageSize: number) {
  const query = useInfiniteQuery({
    queryKey: ["paged", path, params, pageSize],
    queryFn: ({ pageParam, signal }) =>
      apiGet<{ items: T[]; next_cursor: string | null; total?: number }>(
        path,
        { ...params, limit: pageSize, cursor: pageParam ?? undefined },
        signal,
      ),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
    placeholderData: keepPreviousData,
  });
  const rows = query.data?.pages.flatMap((p) => p.items) ?? [];
  return {
    rows,
    total: query.data?.pages[0]?.total,
    hasMore: query.hasNextPage,
    loadMore: () => void query.fetchNextPage(),
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,
    refetch: () => void query.refetch(),
  };
}

export function useMe() {
  return useQuery({
    queryKey: ["me"],
    queryFn: ({ signal }) => apiGet<MeResponse>("/me", undefined, signal),
    staleTime: Infinity,
    retry: false,
  });
}

export function useSearch(q: string) {
  return useQuery({
    queryKey: ["search", q],
    queryFn: ({ signal }) => apiGet<SearchResponse>("/search", { q, limit: 5 }, signal),
    enabled: q.trim().length > 0,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
}

export function useModels(params: Params) {
  return useQuery({
    queryKey: ["models", params],
    queryFn: ({ signal }) => apiGet<ModelsListResponse>("/models", { limit: 200, ...params }, signal),
    placeholderData: keepPreviousData,
  });
}

export function useModelSeries(series: string) {
  return useQuery({
    queryKey: ["model-series", series],
    queryFn: ({ signal }) => apiGet<ModelSeriesDetailResponse>("/models/series", { series }, signal),
  });
}

export function useProgression(params: Params, enabled = true) {
  return useQuery({
    queryKey: ["progression", params],
    queryFn: ({ signal }) => apiGet<ProgressionResponse>("/models/progression", params, signal),
    placeholderData: keepPreviousData,
    enabled,
  });
}

export function useTasks(params: Params) {
  return useQuery({
    queryKey: ["tasks", params],
    queryFn: ({ signal }) => apiGet<TasksListResponse>("/tasks", { limit: 500, ...params }, signal),
    placeholderData: keepPreviousData,
  });
}

export function useTaskDetail(task: string) {
  return useQuery({
    queryKey: ["task-detail", task],
    queryFn: ({ signal }) => apiGet<TaskDetailResponse>("/tasks/detail", { task }, signal),
  });
}

export function useTaskLeaderboard(params: Params) {
  return useQuery({
    queryKey: ["task-leaderboard", params],
    queryFn: ({ signal }) => apiGet<LeaderboardResponse>("/tasks/leaderboard", params, signal),
    placeholderData: keepPreviousData,
  });
}

export function useTaskRuntime(task: string, params: Params) {
  return useQuery({
    queryKey: ["task-runtime", task, params],
    queryFn: ({ signal }) => apiGet<TaskRuntimeResponse>(`/tasks/${encodeURIComponent(task)}/runtime`, params, signal),
    placeholderData: keepPreviousData,
  });
}

export function useSuites(params: Params = {}) {
  return useQuery({
    queryKey: ["suites", params],
    queryFn: ({ signal }) => apiGet<SuitesListResponse>("/suites", { limit: 500, ...params }, signal),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function useSuiteDetail(suite: string) {
  return useQuery({
    queryKey: ["suite-detail", suite],
    queryFn: ({ signal }) => apiGet<SuiteDetailResponse>("/suites/detail", { suite }, signal),
  });
}

export function useSuiteLeaderboard(params: Params) {
  return useQuery({
    queryKey: ["suite-leaderboard", params],
    queryFn: ({ signal }) => apiGet<LeaderboardResponse>("/suites/leaderboard", params, signal),
    placeholderData: keepPreviousData,
  });
}

export function useGroups(params: Params) {
  return useQuery({
    queryKey: ["groups", params],
    queryFn: ({ signal }) => apiGet<GroupsListResponse>("/groups", params, signal),
    placeholderData: keepPreviousData,
  });
}

export function useGroupDetail(group: string) {
  return useQuery({
    queryKey: ["group-detail", group],
    queryFn: ({ signal }) => apiGet<GroupDetailResponse>("/groups/detail", { group }, signal),
    enabled: !!group,
  });
}

export function useActivity(params: Params = {}) {
  return useInfiniteQuery({
    queryKey: ["activity", params],
    queryFn: ({ pageParam, signal }) =>
      apiGet<ActivityResponse>("/activity", { ...params, cursor: pageParam ?? undefined }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next_cursor,
  });
}

export function useStatsSummary() {
  return useQuery({
    queryKey: ["stats-summary"],
    queryFn: ({ signal }) => apiGet<StatsSummaryResponse>("/stats/summary", { window_days: 7 }, signal),
  });
}

export function useSavedViews(page: string) {
  return useQuery({
    queryKey: ["views", page],
    queryFn: ({ signal }) => apiGet<SavedViewsResponse>("/views", { page }, signal),
  });
}

export function useViewMutations() {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries({ queryKey: ["views"] });
  const create = useMutation({
    mutationFn: (body: CreateViewRequest) => apiPost<SavedView>("/views", body),
    onSuccess: invalidate,
  });
  const update = useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateViewRequest }) =>
      apiPatch<SavedView>(`/views/${id}`, body),
    onSuccess: invalidate,
  });
  const remove = useMutation({
    mutationFn: (id: string) => apiDelete(`/views/${id}`),
    onSuccess: invalidate,
  });
  return { create, update, remove };
}
