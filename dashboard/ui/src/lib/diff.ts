import { diffWords } from "diff";

export interface TextSegment {
  text: string;
  kind: "same" | "ins" | "del";
}

/** Word-level diff from `a` (reference) to `b`. */
export function wordDiff(a: string, b: string): TextSegment[] {
  return diffWords(a, b).map((part) => ({
    text: part.value,
    kind: part.added ? "ins" : part.removed ? "del" : "same",
  }));
}

export type JsonDiffKind = "same" | "changed" | "added" | "removed";

export interface JsonDiffEntry {
  path: string;
  kind: JsonDiffKind;
  a?: unknown;
  b?: unknown;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function joinPath(base: string, key: string | number): string {
  if (typeof key === "number") return `${base}[${key}]`;
  return base ? `${base}.${key}` : key;
}

/** Structural diff of two JSON values, flattened to leaf paths in key order. */
export function jsonDiff(a: unknown, b: unknown, path = ""): JsonDiffEntry[] {
  if (isPlainObject(a) && isPlainObject(b)) {
    const keys = Array.from(new Set([...Object.keys(a), ...Object.keys(b)])).sort();
    return keys.flatMap((key) => {
      const p = joinPath(path, key);
      if (!(key in b)) return [{ path: p, kind: "removed" as const, a: a[key] }];
      if (!(key in a)) return [{ path: p, kind: "added" as const, b: b[key] }];
      return jsonDiff(a[key], b[key], p);
    });
  }
  if (Array.isArray(a) && Array.isArray(b) && a.every((v) => !isPlainObject(v))) {
    return JSON.stringify(a) === JSON.stringify(b)
      ? [{ path, kind: "same", a, b }]
      : [{ path, kind: "changed", a, b }];
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.max(a.length, b.length);
    const out: JsonDiffEntry[] = [];
    for (let i = 0; i < n; i++) {
      const p = joinPath(path, i);
      if (i >= b.length) out.push({ path: p, kind: "removed", a: a[i] });
      else if (i >= a.length) out.push({ path: p, kind: "added", b: b[i] });
      else out.push(...jsonDiff(a[i], b[i], p));
    }
    return out;
  }
  return JSON.stringify(a) === JSON.stringify(b)
    ? [{ path, kind: "same", a, b }]
    : [{ path, kind: "changed", a, b }];
}

export function diffSummary(entries: JsonDiffEntry[]): Record<JsonDiffKind, number> {
  const out: Record<JsonDiffKind, number> = { same: 0, changed: 0, added: 0, removed: 0 };
  for (const e of entries) out[e.kind] += 1;
  return out;
}
