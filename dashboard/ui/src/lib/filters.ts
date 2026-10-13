/**
 * Runs-list filters: URL params, the text grammar typed in the filter bar, and the API params.
 *
 * URL keys are short (`ws`, `ntag`, `smin`); the API uses the long names from spec 4.3.
 * The text grammar is `key:value[,value]` tokens (`model:olmo3*`, `-tag:debug`, `step:>=1000`,
 * `has:failures`), anything else is free text.
 */
import { joinList, parseList } from "./url";

export const RUN_FILTER_KEYS = [
  "model",
  "series",
  "family",
  "task",
  "suite",
  "sc",
  "user",
  "group",
  "launch",
  "ws",
  "tag",
  "ntag",
  "status",
  "after",
  "before",
  "date",
  "commit",
  "smin",
  "smax",
  "fail",
  "q",
] as const;

export type RunFilterKey = (typeof RUN_FILTER_KEYS)[number];
export type RunFilters = Partial<Record<RunFilterKey, string>>;

/** Keys whose value is a comma-separated OR list. */
const LIST_KEYS: RunFilterKey[] = [
  "model",
  "series",
  "family",
  "task",
  "suite",
  "user",
  "group",
  "launch",
  "ws",
  "tag",
  "ntag",
  "status",
];

/** Text-grammar key → URL key. */
const TEXT_KEYS: Record<string, RunFilterKey> = {
  model: "model",
  series: "series",
  family: "family",
  task: "task",
  suite: "suite",
  user: "user",
  group: "group",
  launch: "launch",
  ws: "ws",
  workspace: "ws",
  tag: "tag",
  status: "status",
  after: "after",
  before: "before",
  date: "date",
  commit: "commit",
};

export const FILTER_LABELS: Record<RunFilterKey, string> = {
  model: "Model",
  series: "Series",
  family: "Family",
  task: "Task",
  suite: "Suite",
  sc: "Suite coverage",
  user: "User",
  group: "Group",
  launch: "Launch",
  ws: "Workspace",
  tag: "Tag",
  ntag: "Not tag",
  status: "Status",
  after: "After",
  before: "Before",
  date: "Date",
  commit: "Commit",
  smin: "Step ≥",
  smax: "Step ≤",
  fail: "Has failures",
  q: "Text",
};

function addListValue(filters: RunFilters, key: RunFilterKey, value: string) {
  const existing = parseList(filters[key]);
  for (const v of parseList(value)) if (!existing.includes(v)) existing.push(v);
  filters[key] = joinList(existing);
}

/** Split on whitespace, keeping quoted phrases together. */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const re = /"([^"]*)"|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) tokens.push(m[1] ?? m[2]);
  return tokens;
}

export function parseFilterText(text: string): RunFilters {
  const filters: RunFilters = {};
  const free: string[] = [];
  for (const token of tokenize(text)) {
    const lower = token.toLowerCase();
    if (lower === "has:failures") {
      filters.fail = "1";
      continue;
    }
    const negated = token.startsWith("-") && token.includes(":");
    const body = negated ? token.slice(1) : token;
    const colon = body.indexOf(":");
    if (colon <= 0) {
      free.push(token);
      continue;
    }
    const rawKey = body.slice(0, colon).toLowerCase();
    const value = body.slice(colon + 1);
    if (!value) {
      free.push(token);
      continue;
    }
    if (rawKey === "step") {
      const m = /^(>=|<=|>|<|=)?(\d+)$/.exec(value);
      if (!m) {
        free.push(token);
        continue;
      }
      const n = Number(m[2]);
      const op = m[1] ?? "=";
      if (op === ">=" || op === ">") filters.smin = String(op === ">" ? n + 1 : n);
      else if (op === "<=" || op === "<") filters.smax = String(op === "<" ? n - 1 : n);
      else {
        filters.smin = String(n);
        filters.smax = String(n);
      }
      continue;
    }
    if (rawKey === "tag" && negated) {
      addListValue(filters, "ntag", value);
      continue;
    }
    const key = TEXT_KEYS[rawKey];
    if (!key) {
      free.push(token);
      continue;
    }
    if (LIST_KEYS.includes(key)) addListValue(filters, key, value);
    else filters[key] = value;
  }
  if (free.length) filters.q = free.join(" ");
  return filters;
}

function quote(value: string): string {
  return /\s/.test(value) ? `"${value}"` : value;
}

/** Inverse of parseFilterText (canonical order). */
export function filtersToText(filters: RunFilters): string {
  const parts: string[] = [];
  for (const key of RUN_FILTER_KEYS) {
    const value = filters[key];
    if (!value) continue;
    switch (key) {
      case "q":
        break;
      case "ntag":
        for (const v of parseList(value)) parts.push(`-tag:${v}`);
        break;
      case "fail":
        if (value === "1") parts.push("has:failures");
        break;
      case "sc":
        break;
      case "smin":
        if (filters.smax === value) parts.push(`step:${value}`);
        else parts.push(`step:>=${value}`);
        break;
      case "smax":
        if (filters.smin !== value) parts.push(`step:<=${value}`);
        break;
      default:
        parts.push(`${key}:${quote(value)}`);
    }
  }
  if (filters.q) parts.push(filters.q);
  return parts.join(" ");
}

/**
 * A date filter value as the full ISO datetime the API expects. A bare `YYYY-MM-DD` means
 * midnight in the viewer's time zone; anything else passes through unchanged.
 */
export function dateParam(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!m) return value;
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  return Number.isNaN(d.getTime()) ? value : d.toISOString();
}

/** URL filters → /api/runs query params (repeatable keys become arrays). */
export function filtersToApi(filters: RunFilters): Record<string, string | string[] | undefined> {
  const list = (key: RunFilterKey) => {
    const values = parseList(filters[key]);
    return values.length ? values : undefined;
  };
  return {
    model: list("model"),
    series: list("series"),
    family: list("family"),
    task: list("task"),
    suite: list("suite"),
    suite_coverage: filters.sc === "any" ? "any" : undefined,
    user: list("user"),
    group: list("group"),
    launch: list("launch"),
    workspace: list("ws"),
    tag: list("tag"),
    not_tag: list("ntag"),
    status: list("status"),
    after: dateParam(filters.after),
    before: dateParam(filters.before),
    date: filters.date,
    commit: filters.commit,
    step_min: filters.smin,
    step_max: filters.smax,
    has_failures: filters.fail === "1" ? "true" : undefined,
    q: filters.q,
  };
}

export function activeFilterCount(filters: RunFilters): number {
  return RUN_FILTER_KEYS.filter((k) => k !== "sc" && filters[k]).length;
}

export function removeFilterValue(filters: RunFilters, key: RunFilterKey, value?: string): RunFilters {
  const next = { ...filters };
  if (value === undefined || !LIST_KEYS.includes(key)) {
    delete next[key];
    if (key === "suite") delete next.sc;
    return next;
  }
  const remaining = parseList(next[key]).filter((v) => v !== value);
  if (remaining.length) next[key] = joinList(remaining);
  else delete next[key];
  return next;
}

export function isListKey(key: RunFilterKey): boolean {
  return LIST_KEYS.includes(key);
}
