import type { ModelRef, SubjectKey } from "@contract/api-types";
import { formatStep } from "./format";

const SUBJECT_RE = /^(r:[a-z0-9]{6,32}|m:[0-9a-f]{12})$/;

export function isSubjectKey(value: string): value is SubjectKey {
  return SUBJECT_RE.test(value);
}

export function runSubject(runId: string): SubjectKey {
  return `r:${runId}`;
}

export function modelSubject(modelId: string): SubjectKey {
  return `m:${modelId}`;
}

export function parseSubject(key: string): { kind: "run" | "model"; id: string } | null {
  if (!isSubjectKey(key)) return null;
  return { kind: key.startsWith("r:") ? "run" : "model", id: key.slice(2) };
}

/** Parse a comma list of subject keys, dropping invalid and duplicate entries. */
export function parseSubjects(value: string | undefined): SubjectKey[] {
  if (!value) return [];
  const out: SubjectKey[] = [];
  for (const raw of value.split(",")) {
    const key = raw.trim();
    if (isSubjectKey(key) && !out.includes(key)) out.push(key);
  }
  return out;
}

export function modelLabel(model: Pick<ModelRef, "series_label" | "step">): string {
  return model.step != null ? `${model.series_label} @ ${formatStep(model.step)}` : model.series_label;
}

export function encodePath(value: string): string {
  return encodeURIComponent(value);
}
