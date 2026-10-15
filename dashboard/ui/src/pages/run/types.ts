import type { RunDetail, RunSuitesResponse, RunTaskResultsResponse, SubjectInfo, SubjectKey, TaskResultRow } from "@contract/api-types";
import type { UseQueryResult } from "@tanstack/react-query";

export type RunSearch = {
  tab?: string;
  task?: string;
  hash?: string;
  inst?: string;
  cell?: string;
  correct?: string;
  thr?: string;
  fin?: string;
  len?: string;
  score?: string;
  has?: string;
  iq?: string;
  isort?: string;
  only?: string;
  side?: string;
  diff?: string;
  tq?: string;
  tf?: string;
  all?: string;
  sel?: string;
  /** Tasks tab sort. */
  ts?: string;
  /** Tasks tab runtime view: "with_startup", or unset for inference only. */
  rt?: string;
  baseline?: string;
};

export interface RunCtx {
  run: RunDetail;
  baseline?: SubjectKey;
  baselineInfo?: SubjectInfo | null;
  tasks: UseQueryResult<RunTaskResultsResponse>;
  suites: UseQueryResult<RunSuitesResponse>;
  search: RunSearch;
  setSearch: (patch: Partial<RunSearch>, options?: { replace?: boolean }) => void;
}

/** The primary metric's display metadata for a task result row. */
export function rowMeta(row: TaskResultRow) {
  const meta = row.primary_metric ? row.metric_meta[row.primary_metric] : undefined;
  return {
    format: meta?.display_format ?? "percent",
    higherIsBetter: meta?.higher_is_better ?? true,
    kind: meta?.kind ?? "bounded",
  } as const;
}

/** Display format for a task result's score: raw when scores are on a 100x scale. */
export function scoreFormat(row: TaskResultRow): "percent" | "raw" {
  const meta = rowMeta(row);
  return row.instance_scale !== 1 ? "raw" : meta.format;
}
