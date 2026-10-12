import type { DisplayFormat } from "@contract/api-types";

const MINUS = "−";

function trimZeros(text: string): string {
  if (!text.includes(".") || text.includes("e")) return text;
  return text.replace(/\.?0+$/, "");
}

/** Three significant digits without scientific notation for everyday magnitudes. */
export function sig3(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1000) return Math.round(value).toLocaleString("en-US");
  if (abs < 0.001) return value.toExponential(1);
  return trimZeros(value.toPrecision(3));
}

function withMinus(text: string): string {
  return text.startsWith("-") ? MINUS + text.slice(1) : text;
}

/** Score in display units: percent metrics as value x 100 with one decimal, raw as 3 sig. digits. */
export function formatScore(
  value: number | null | undefined,
  format: DisplayFormat = "percent",
  decimals = 1,
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (format === "percent") return withMinus((value * 100).toFixed(decimals));
  return withMinus(sig3(value));
}

/** Stderr shown as "±x" in the same display units as the score. */
export function formatStderr(
  value: number | null | undefined,
  format: DisplayFormat = "percent",
): string {
  if (value == null || !Number.isFinite(value)) return "";
  if (format === "percent") return `±${(value * 100).toFixed(1)}`;
  return `±${sig3(value)}`;
}

/** Signed delta; percent metrics in percentage points. */
export function formatDelta(
  value: number | null | undefined,
  format: DisplayFormat = "percent",
  decimals = 1,
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const scaled = format === "percent" ? value * 100 : value;
  const body = format === "percent" ? Math.abs(scaled).toFixed(decimals) : sig3(Math.abs(scaled));
  if (Number(body) === 0) return format === "percent" ? (0).toFixed(decimals) : "0";
  return `${scaled > 0 ? "+" : MINUS}${body}`;
}

export function formatCI(
  low: number | null | undefined,
  high: number | null | undefined,
  format: DisplayFormat = "percent",
): string {
  if (low == null || high == null) return "—";
  return `[${formatDelta(low, format)}, ${formatDelta(high, format)}]`;
}

export function formatP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "—";
  if (p < 0.001) return "p<0.001";
  if (p < 0.01) return `p=${p.toFixed(3)}`;
  return `p=${p.toFixed(2)}`;
}

export function formatCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

export function formatCompact(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `${trimZeros((n / 1e9).toFixed(1))}B`;
  if (abs >= 1e6) return `${trimZeros((n / 1e6).toFixed(1))}M`;
  if (abs >= 1e4) return `${trimZeros((n / 1e3).toFixed(1))}k`;
  return formatCount(n);
}

/** Share in [0, 1] as a percentage. */
export function formatRate(rate: number | null | undefined, decimals = 0): string {
  if (rate == null || !Number.isFinite(rate)) return "—";
  const pct = rate * 100;
  if (pct > 0 && pct < 1 && decimals === 0) return "<1%";
  return `${pct.toFixed(decimals)}%`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(value >= 100 ? 0 : 1)} ${units[unit]}`;
}

export function formatSeconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 100) return `${s.toFixed(s < 10 ? 2 : 1)} s`;
  return formatDuration(s);
}

/**
 * Elapsed time that reads well at any size: "350 ms", "4.2 s", "42 s", "3m 12s", "1h 05m",
 * "2d 3h". A leading "~" can be added by callers for estimates.
 */
export function formatRuntime(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 0) return `${MINUS}${formatRuntime(-s)}`;
  if (s === 0) return "0 s";
  if (s < 1) return `${Math.max(1, Math.round(s * 1000))} ms`;
  if (s < 9.95) return `${s.toFixed(1)} s`;
  if (s < 59.5) return `${Math.round(s)} s`;
  return formatDuration(s);
}

/** Signed runtime difference ("+42 s", "−3m 10s"). */
export function formatRuntimeDelta(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (Math.abs(s) < 0.05) return "0 s";
  return `${s > 0 ? "+" : MINUS}${formatRuntime(Math.abs(s))}`;
}

export function formatStep(step: number | null | undefined): string {
  if (step == null) return "—";
  // Compact "k" form when it is exact to two decimals (2000 → 2k, 2400 → 2.4k, 1750 → 1.75k).
  if (step >= 1000 && step % 10 === 0) return `${trimZeros((step / 1000).toFixed(2))}k`;
  return step.toLocaleString("en-US");
}

/** Short form of a hash or ID for display. */
export function shortHash(value: string | null | undefined, length = 7): string {
  if (!value) return "—";
  return value.length > length ? value.slice(0, length) : value;
}

/** "metric:scorer" → readable label ("exact_match (flex)"). */
export function metricLabel(key: string | null | undefined): string {
  if (!key) return "—";
  const [metric, scorer] = key.split(":");
  if (!scorer || scorer === "default" || scorer === metric) return metric;
  return `${metric} (${scorer})`;
}

export function pluralize(n: number, one: string, many = `${one}s`): string {
  return `${formatCount(n)} ${n === 1 ? one : many}`;
}
