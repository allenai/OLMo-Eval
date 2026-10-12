/**
 * Task runtime helpers. Tasks in a run share inference workers, so a task's own time comes from
 * the server's `runtime` object, with a basis that says how it was obtained.
 */
import type { RuntimeBasis, RuntimeMode, TaskRuntime } from "@contract/api-types";
import { formatRuntime } from "./format";
import { oneOf } from "./url";

export const RUNTIME_MODES = ["inference", "with_startup"] as const satisfies readonly RuntimeMode[];

/** URL value to mode; Inference is the default and stays out of the URL. */
export function parseRuntimeMode(value: string | undefined): RuntimeMode {
  return oneOf<RuntimeMode>(value, RUNTIME_MODES, "inference");
}

export function runtimeModeParam(mode: RuntimeMode): string | undefined {
  return mode === "inference" ? undefined : mode;
}

export const RUNTIME_MODE_LABEL: Record<RuntimeMode, string> = {
  inference: "Inference",
  with_startup: "With startup",
};

export const RUNTIME_MODE_HELP: Record<RuntimeMode, string> = {
  inference: "Time spent on this task's requests alone.",
  with_startup:
    "Inference plus the run's startup (model load and server start): about what running this task on its own takes. Beaker queue time is never included.",
};

export const RUNTIME_MODE_OPTIONS = RUNTIME_MODES.map((m) => ({ value: m, label: RUNTIME_MODE_LABEL[m], title: RUNTIME_MODE_HELP[m] }));

export function runtimeValue(rt: TaskRuntime | null | undefined, mode: RuntimeMode): number | null {
  if (!rt) return null;
  return mode === "inference" ? rt.inference_seconds : rt.with_startup_seconds;
}

export const BASIS_LABEL: Record<RuntimeBasis, string> = {
  measured: "Measured",
  attributed: "Attributed",
  estimated: "Estimated",
  not_recorded: "Not recorded",
};

export const BASIS_HELP: Record<RuntimeBasis, string> = {
  measured: "The run contained only this task, so the run's processing time is the task's time.",
  attributed: "Split from per-batch inference metrics by this task's share of each batch.",
  estimated: "The run's processing time times this task's share of the run's tokens. Rough when tasks differ a lot in prompt versus output length.",
  not_recorded: "This run predates runtime recording or lacks token counts.",
};

/** Short prefix for values that are estimates. */
export function basisMark(basis: RuntimeBasis | null | undefined): string {
  return basis === "estimated" ? "~" : "";
}

export function formatTaskRuntime(rt: TaskRuntime | null | undefined, mode: RuntimeMode): string {
  const v = runtimeValue(rt, mode);
  return v == null ? "—" : `${basisMark(rt?.basis)}${formatRuntime(v)}`;
}

/** "NVIDIA H100 80GB HBM3" → "H100", "NVIDIA A100-SXM4-80GB" → "A100". */
export function gpuShort(type: string | null | undefined): string {
  if (!type) return "—";
  return type
    .replace(/^NVIDIA\s+/i, "")
    .replace(/[-\s](SXM|PCIE|NVL|\d+\s?GB).*$/i, "")
    .trim();
}

export function gpuLabel(type: string | null | undefined, count: number | null | undefined): string {
  if (!type && !count) return "—";
  return `${count ?? "?"}× ${type ? gpuShort(type) : "GPU"}`;
}

export const RUNTIME_METRICS = ["runtime:inference", "runtime:with_startup"] as const;

/** Mode for a compare metric key, or null when the metric is a score. */
export function runtimeMetricMode(metric: string | null | undefined): RuntimeMode | null {
  if (metric === "runtime:inference") return "inference";
  if (metric === "runtime:with_startup") return "with_startup";
  return null;
}

/** Use a log axis when the values span more than ~20x. */
export function wantsLogScale(values: number[]): boolean {
  const pos = values.filter((v) => v > 0);
  if (pos.length < 2) return false;
  return Math.max(...pos) / Math.min(...pos) >= 20;
}

/** Each value's share of the total (null when the value or every value is missing). */
export function shares(values: (number | null)[]): (number | null)[] {
  const total = values.reduce<number>((acc, v) => acc + (v ?? 0), 0);
  return values.map((v) => (v == null || total <= 0 ? null : v / total));
}

export interface TimelineInput {
  key: string;
  label: string;
  runtime: TaskRuntime;
  failed?: boolean;
}

export interface TimelineBar {
  key: string;
  label: string;
  kind: "startup" | "task";
  /** Seconds relative to the start of processing; startup is negative. */
  start: number;
  end: number;
  basis: RuntimeBasis | null;
  inference: number | null;
  failed: boolean;
}

export interface TimelineLayout {
  bars: TimelineBar[];
  domain: [number, number];
  /** Tasks without a recorded span. */
  missing: number;
}

/** Gantt layout: startup as its own bar before zero, then each task's span sorted by start. */
export function layoutTimeline(rows: TimelineInput[], startup: number | null): TimelineLayout {
  const tasks: TimelineBar[] = [];
  let missing = 0;
  for (const r of rows) {
    const { span_start_s: a, span_end_s: b } = r.runtime;
    if (a == null || b == null) {
      missing += 1;
      continue;
    }
    tasks.push({ key: r.key, label: r.label, kind: "task", start: a, end: Math.max(a, b), basis: r.runtime.basis, inference: r.runtime.inference_seconds, failed: !!r.failed });
  }
  tasks.sort((x, y) => x.start - y.start || x.end - y.end || x.label.localeCompare(y.label));
  const bars: TimelineBar[] = [];
  if (startup != null && startup > 0) {
    bars.push({ key: "__startup", label: "Startup", kind: "startup", start: -startup, end: 0, basis: null, inference: null, failed: false });
  }
  bars.push(...tasks);
  const end = Math.max(0, ...tasks.map((t) => t.end));
  return { bars, domain: [bars.length && bars[0].kind === "startup" ? bars[0].start : 0, end || 1], missing };
}

const TICK_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 4 * 3600, 6 * 3600, 12 * 3600, 86400];

/**
 * Axis ticks in seconds at human time steps (10 s, 1 min, 5 min, 1 h, ...). A log axis takes
 * every step inside the domain, thinned to `max`; a linear axis takes multiples of the smallest
 * step that fits.
 */
export function runtimeTicks(lo: number, hi: number, max = 6, log = false): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return Number.isFinite(lo) ? [lo] : [];
  if (log) {
    const inside = TICK_STEPS.filter((t) => t >= lo && t <= hi);
    if (inside.length <= max) return inside;
    const every = Math.ceil(inside.length / max);
    return inside.filter((_, i) => i % every === 0);
  }
  const step = TICK_STEPS.find((s) => (hi - lo) / s <= max) ?? TICK_STEPS[TICK_STEPS.length - 1];
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000);
  return out;
}
