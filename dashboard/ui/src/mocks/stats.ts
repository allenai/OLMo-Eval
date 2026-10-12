/**
 * Approximate statistics for the mock API. The real API runs a paired bootstrap; here a normal
 * approximation over paired per-instance differences gives similar CIs and is fast enough to run
 * in the browser.
 */
import type { Contingency, DeltaStats } from "@contract/api-types";
import { instanceData, type MockTaskResult, phi } from "./fixtures";

export function zFor(alpha: number): number {
  if (alpha <= 0.01) return 2.576;
  if (alpha >= 0.1) return 1.645;
  return 1.96;
}

export function emptyDelta(alpha: number, method: DeltaStats["method"], note: string | null): DeltaStats {
  return {
    delta: null,
    ci_low: null,
    ci_high: null,
    p_value: null,
    n_shared: 0,
    method,
    significant: false,
    improved: null,
    hash_mismatch: false,
    alpha,
    n_boot: null,
    note,
  };
}

function improvedOf(delta: number | null, hib: boolean | null): boolean | null {
  if (delta == null || delta === 0 || hib == null) return null;
  return hib ? delta > 0 : delta < 0;
}

export interface PairedSummary {
  n: number;
  mean: number;
  se: number;
}

/** Paired difference a - b over shared instances (instances are a shared prefix by index). */
export function pairedSummary(a: MockTaskResult, b: MockTaskResult): PairedSummary {
  const n = Math.min(a.n, b.n);
  if (n === 0) return { n: 0, mean: 0, se: 0 };
  const da = instanceData(a).primary;
  const db = instanceData(b).primary;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = da[i] - db[i];
    sum += d;
    sumSq += d * d;
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  const scale = a.task.scale;
  return { n, mean: mean * scale, se: (Math.sqrt(variance) / Math.sqrt(Math.max(1, n - 1))) * scale };
}

export function pairedDelta(a: MockTaskResult, b: MockTaskResult, alpha = 0.05, nBoot = 2000): DeltaStats {
  const hib = a.task.meta[a.task.primary]?.higher_is_better ?? true;
  const hashMismatch = a.taskHash !== b.taskHash;
  if (a.error || b.error) return { ...emptyDelta(alpha, "none", "One side failed on this task."), hash_mismatch: hashMismatch };
  const s = pairedSummary(a, b);
  if (s.n < 20) {
    return {
      ...emptyDelta(alpha, "insufficient", `Only ${s.n} shared instances (need 20).`),
      delta: s.n ? s.mean : null,
      n_shared: s.n,
      hash_mismatch: hashMismatch,
      improved: improvedOf(s.n ? s.mean : null, hib),
    };
  }
  const z = zFor(alpha);
  const half = z * s.se;
  const zStat = s.se > 0 ? s.mean / s.se : s.mean === 0 ? 0 : 99;
  const p = Math.min(1, 2 * (1 - phi(Math.abs(zStat))));
  const significant = s.se > 0 ? Math.abs(s.mean) > half : s.mean !== 0;
  return {
    delta: s.mean,
    ci_low: s.mean - half,
    ci_high: s.mean + half,
    p_value: p,
    n_shared: s.n,
    method: "paired_bootstrap",
    significant,
    improved: improvedOf(s.mean, hib),
    hash_mismatch: hashMismatch,
    alpha,
    n_boot: nBoot,
    note: null,
  };
}

export function unpairedDelta(
  a: { score: number | null; stderr: number | null },
  b: { score: number | null; stderr: number | null },
  hib: boolean | null,
  alpha = 0.05,
): DeltaStats {
  if (a.score == null || b.score == null) return emptyDelta(alpha, "none", "Missing score.");
  const delta = a.score - b.score;
  if (a.stderr == null || b.stderr == null) {
    return { ...emptyDelta(alpha, "none", "No stderr; delta shown without CI."), delta, improved: improvedOf(delta, hib) };
  }
  const se = Math.sqrt(a.stderr ** 2 + b.stderr ** 2);
  const z = zFor(alpha);
  const p = se > 0 ? Math.min(1, 2 * (1 - phi(Math.abs(delta / se)))) : 1;
  return {
    delta,
    ci_low: delta - z * se,
    ci_high: delta + z * se,
    p_value: p,
    n_shared: 0,
    method: "unpaired",
    significant: Math.abs(delta) > z * se,
    improved: improvedOf(delta, hib),
    hash_mismatch: false,
    alpha,
    n_boot: null,
    note: null,
  };
}

/** Combine per-task paired deltas into a suite delta with weights w_i. */
export function combineDeltas(parts: { stats: DeltaStats; se: number; w: number }[], hib: boolean | null, alpha: number): DeltaStats {
  const usable = parts.filter((p) => p.stats.delta != null && p.stats.method === "paired_bootstrap");
  if (!usable.length) return emptyDelta(alpha, "none", "No paired tasks.");
  const wsum = usable.reduce((acc, p) => acc + p.w, 0);
  const delta = usable.reduce((acc, p) => acc + (p.w / wsum) * (p.stats.delta ?? 0), 0);
  const se = Math.sqrt(usable.reduce((acc, p) => acc + (p.w / wsum) ** 2 * p.se ** 2, 0));
  const z = zFor(alpha);
  return {
    delta,
    ci_low: delta - z * se,
    ci_high: delta + z * se,
    p_value: se > 0 ? Math.min(1, 2 * (1 - phi(Math.abs(delta / se)))) : 1,
    n_shared: usable.reduce((acc, p) => acc + p.stats.n_shared, 0),
    method: "paired_bootstrap",
    significant: Math.abs(delta) > z * se,
    improved: improvedOf(delta, hib),
    hash_mismatch: usable.some((p) => p.stats.hash_mismatch),
    alpha,
    n_boot: 2000,
    note: null,
  };
}

export function isCorrect(tr: MockTaskResult, value: number, threshold: number): boolean | null {
  if (tr.task.kind === "unbounded") return null;
  const hib = tr.task.meta[tr.task.primary]?.higher_is_better !== false;
  const goodness = hib ? value : 1 - value;
  return goodness >= threshold;
}

export function contingency(a: MockTaskResult, b: MockTaskResult, threshold = 0.5): Contingency | null {
  if (a.task.kind === "unbounded" || a.error || b.error) return null;
  const n = Math.min(a.n, b.n);
  if (!n) return null;
  const da = instanceData(a).primary;
  const db = instanceData(b).primary;
  const c: Contingency = { both_right: 0, only_a: 0, only_b: 0, both_wrong: 0, n_shared: n, net: 0, threshold };
  for (let i = 0; i < n; i++) {
    const ca = isCorrect(a, da[i], threshold);
    const cb = isCorrect(b, db[i], threshold);
    if (ca && cb) c.both_right += 1;
    else if (ca) c.only_a += 1;
    else if (cb) c.only_b += 1;
    else c.both_wrong += 1;
  }
  c.net = c.only_a - c.only_b;
  return c;
}

export function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((x, y) => x - y);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function histogram(values: number[], bins: number, lo?: number, hi?: number) {
  if (!values.length) return { edges: [0, 1], counts: [0] };
  const min = lo ?? Math.min(...values);
  let max = hi ?? Math.max(...values);
  if (max <= min) max = min + 1;
  const width = (max - min) / bins;
  const edges = Array.from({ length: bins + 1 }, (_, i) => min + i * width);
  const counts = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
    counts[idx] += 1;
  }
  return { edges, counts };
}
