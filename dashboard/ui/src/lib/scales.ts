import type { DeltaStats } from "@contract/api-types";

/** Sequential teal step (0..6) for t in [0, 1]. */
export function seqStep(t: number): number {
  if (!Number.isFinite(t)) return 0;
  return Math.max(0, Math.min(6, Math.round(t * 6)));
}

export function seqColor(t: number): string {
  return `var(--seq-${seqStep(t)})`;
}

/** Text color that stays legible on a sequential cell. */
export function seqTextColor(t: number): string {
  return `var(--seq-ink-${seqStep(t)})`;
}

const DIV_TOKENS = ["--div-n3", "--div-n2", "--div-n1", "--div-0", "--div-p1", "--div-p2", "--div-p3"];
const DIV_INK = ["n3", "n2", "n1", "0", "p1", "p2", "p3"];

/**
 * Diverging step in -3..3. `goodness` is the delta signed so that positive means better
 * (already flipped for lower-is-better metrics); `scale` is the magnitude that maps to step 3.
 */
export function divStep(goodness: number, scale: number): number {
  if (!Number.isFinite(goodness) || goodness === 0 || scale <= 0) return 0;
  const t = Math.min(1, Math.abs(goodness) / scale);
  const step = Math.max(1, Math.ceil(t * 3));
  return goodness > 0 ? step : -step;
}

export function divColor(step: number): string {
  return `var(${DIV_TOKENS[step + 3]})`;
}

/** Text color that stays legible on a diverging cell. */
export function divTextColor(step: number): string {
  return `var(--div-ink-${DIV_INK[step + 3]})`;
}

/** Signed "goodness" of a delta: positive when the change is an improvement. */
export function goodness(delta: number | null | undefined, higherIsBetter: boolean | null): number {
  if (delta == null || !Number.isFinite(delta)) return 0;
  return higherIsBetter === false ? -delta : delta;
}

export type Evidence = "significant" | "ns" | "insufficient" | "none";

export function deltaEvidence(stats: DeltaStats | null | undefined): Evidence {
  if (!stats || stats.delta == null) return "none";
  if (stats.method === "insufficient") return "insufficient";
  if (stats.method === "none") return "none";
  return stats.significant ? "significant" : "ns";
}

export function catColor(slot: number | null | undefined): string {
  if (slot == null || slot < 0 || slot > 5) return "var(--cat-context)";
  return `var(--cat-${slot + 1})`;
}

/** Min/max ignoring nulls; returns null when nothing is finite. */
export function extent(values: (number | null | undefined)[]): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v == null || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo === Infinity ? null : [lo, hi];
}

/** Normalize v into [0,1] over an extent (0.5 when the extent is degenerate). */
export function normalize(v: number, ext: [number, number] | null, higherIsBetter = true): number {
  if (!ext) return 0.5;
  const [lo, hi] = ext;
  if (hi - lo < 1e-12) return 0.5;
  const t = (v - lo) / (hi - lo);
  return higherIsBetter ? t : 1 - t;
}

/** Ranks 1..N within a row (ties share the best rank), in the better direction. */
export function ranks(values: (number | null)[], higherIsBetter: boolean | null): (number | null)[] {
  const indexed = values
    .map((v, i) => ({ v, i }))
    .filter((x): x is { v: number; i: number } => x.v != null && Number.isFinite(x.v));
  indexed.sort((a, b) => (higherIsBetter === false ? a.v - b.v : b.v - a.v));
  const out: (number | null)[] = values.map(() => null);
  let rank = 0;
  let prev: number | null = null;
  indexed.forEach((x, idx) => {
    if (prev === null || Math.abs(x.v - prev) > 1e-12) rank = idx + 1;
    out[x.i] = rank;
    prev = x.v;
  });
  return out;
}

/**
 * Colors for model families on one chart: the most common families get categorical slots in
 * order of frequency, the rest share the context gray.
 */
export function familyColors(families: (string | null | undefined)[], max = 5): { colorOf: (family: string | null | undefined) => string; legend: { label: string; color: string }[] } {
  const counts = new Map<string, number>();
  for (const f of families) counts.set(f ?? "other", (counts.get(f ?? "other") ?? 0) + 1);
  const top = [...counts.entries()]
    .filter(([f]) => f !== "other")
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, max)
    .map(([f]) => f);
  const colorOf = (family: string | null | undefined) => {
    const i = top.indexOf(family ?? "other");
    return i >= 0 ? catColor(i) : "var(--cat-context)";
  };
  const legend = top.map((f, i) => ({ label: f, color: catColor(i) }));
  if (counts.size > top.length) legend.push({ label: "other", color: "var(--cat-context)" });
  return { colorOf, legend };
}
