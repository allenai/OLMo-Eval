import type {
  CompareInstancesRequest,
  CompareInstancesResponse,
  ContingencyRequest,
  ContingencyResponse,
  DistributionsRequest,
  DistributionsResponse,
  MatrixRequest,
  MatrixResponse,
  PairwiseRequest,
  PairwiseResponse,
  ResolveSubjectsResponse,
  SubjectTaskResultResponse,
} from "@contract/api-types";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { apiGet, apiPost } from "../client";

export function useResolveSubjects(subjects: string[], group?: string) {
  return useQuery({
    queryKey: ["resolve", subjects, group ?? null],
    queryFn: ({ signal }) =>
      apiPost<ResolveSubjectsResponse>("/subjects/resolve", { subjects, group: group ?? null }, signal),
    enabled: subjects.length > 0,
    staleTime: 5 * 60_000,
    placeholderData: keepPreviousData,
  });
}

export function useSubjectTaskResult(subject: string | undefined, task: string | undefined, group?: string) {
  return useQuery({
    queryKey: ["subject-task-result", subject, task, group ?? null],
    queryFn: ({ signal }) =>
      apiGet<SubjectTaskResultResponse>("/subjects/task-result", { subject, task, group }, signal),
    enabled: !!subject && !!task,
    staleTime: 5 * 60_000,
  });
}

export function useMatrix(body: MatrixRequest, enabled = true) {
  return useQuery({
    queryKey: ["matrix", body],
    queryFn: ({ signal }) => apiPost<MatrixResponse>("/compare/matrix", body, signal),
    enabled: enabled && body.subjects.length > 0,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function usePairwise(body: PairwiseRequest, enabled = true) {
  return useQuery({
    queryKey: ["pairwise", body],
    queryFn: ({ signal }) => apiPost<PairwiseResponse>("/compare/pairwise", body, signal),
    enabled: enabled && body.subjects.length > 1,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function useContingency(body: ContingencyRequest | null) {
  return useQuery({
    queryKey: ["contingency", body],
    queryFn: ({ signal }) => apiPost<ContingencyResponse>("/compare/contingency", body, signal),
    enabled: !!body,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function useCompareInstances(body: CompareInstancesRequest | null) {
  return useQuery({
    queryKey: ["compare-instances", body],
    queryFn: ({ signal }) => apiPost<CompareInstancesResponse>("/compare/instances", body, signal),
    enabled: !!body,
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });
}

export function useDistributions(body: DistributionsRequest | null) {
  return useQuery({
    queryKey: ["distributions", body],
    queryFn: ({ signal }) => apiPost<DistributionsResponse>("/tasks/distributions", body, signal),
    enabled: !!body && body.items.length > 0,
    staleTime: 5 * 60_000,
  });
}
