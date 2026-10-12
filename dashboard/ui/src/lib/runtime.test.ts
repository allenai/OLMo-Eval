import type { TaskRuntime } from "@contract/api-types";
import { describe, expect, it } from "vitest";
import { formatRuntime, formatRuntimeDelta } from "./format";
import {
  basisMark,
  formatTaskRuntime,
  gpuLabel,
  gpuShort,
  layoutTimeline,
  parseRuntimeMode,
  runtimeMetricMode,
  runtimeModeParam,
  runtimeTicks,
  runtimeValue,
  shares,
  wantsLogScale,
} from "./runtime";
import { familyColors } from "./scales";

function rt(partial: Partial<TaskRuntime>): TaskRuntime {
  return {
    inference_seconds: null,
    basis: "estimated",
    startup_seconds: null,
    with_startup_seconds: null,
    token_share: null,
    prompt_tokens_total: null,
    completion_tokens_total: null,
    seconds_per_1k_instances: null,
    span_start_s: null,
    span_end_s: null,
    ...partial,
  };
}

describe("runtime formatting", () => {
  it("reads well from sub-second to days", () => {
    expect(formatRuntime(0)).toBe("0 s");
    expect(formatRuntime(0.35)).toBe("350 ms");
    expect(formatRuntime(0.0002)).toBe("1 ms");
    expect(formatRuntime(4.24)).toBe("4.2 s");
    expect(formatRuntime(9.96)).toBe("10 s");
    expect(formatRuntime(42.4)).toBe("42 s");
    expect(formatRuntime(59.6)).toBe("1m 00s");
    expect(formatRuntime(192)).toBe("3m 12s");
    expect(formatRuntime(3900)).toBe("1h 05m");
    expect(formatRuntime(200_000)).toBe("2d 7h");
    expect(formatRuntime(null)).toBe("—");
  });

  it("signs runtime differences", () => {
    expect(formatRuntimeDelta(42)).toBe("+42 s");
    expect(formatRuntimeDelta(-190)).toBe("−3m 10s");
    expect(formatRuntimeDelta(0.01)).toBe("0 s");
  });

  it("marks estimates and picks the value for the mode", () => {
    const r = rt({ inference_seconds: 120, with_startup_seconds: 310, basis: "estimated" });
    expect(runtimeValue(r, "inference")).toBe(120);
    expect(runtimeValue(r, "with_startup")).toBe(310);
    expect(formatTaskRuntime(r, "inference")).toBe("~2m 00s");
    expect(formatTaskRuntime({ ...r, basis: "measured" }, "with_startup")).toBe("5m 10s");
    expect(formatTaskRuntime(rt({ basis: "not_recorded" }), "inference")).toBe("—");
    expect(basisMark("attributed")).toBe("");
  });

  it("shortens GPU names", () => {
    expect(gpuShort("NVIDIA H100 80GB HBM3")).toBe("H100");
    expect(gpuShort("NVIDIA A100-SXM4-80GB")).toBe("A100");
    expect(gpuShort("NVIDIA L40S")).toBe("L40S");
    expect(gpuLabel("NVIDIA H100 80GB HBM3", 4)).toBe("4× H100");
    expect(gpuLabel(null, null)).toBe("—");
  });
});

describe("runtime URL state and metrics", () => {
  it("defaults to inference and keeps the default out of the URL", () => {
    expect(parseRuntimeMode(undefined)).toBe("inference");
    expect(parseRuntimeMode("junk")).toBe("inference");
    expect(parseRuntimeMode("with_startup")).toBe("with_startup");
    expect(runtimeModeParam("inference")).toBeUndefined();
    expect(runtimeModeParam("with_startup")).toBe("with_startup");
  });

  it("recognizes runtime compare metrics", () => {
    expect(runtimeMetricMode("runtime:inference")).toBe("inference");
    expect(runtimeMetricMode("runtime:with_startup")).toBe("with_startup");
    expect(runtimeMetricMode("primary")).toBeNull();
  });
});

describe("runtime transforms", () => {
  it("lays out startup first, then tasks by start time", () => {
    const layout = layoutTimeline(
      [
        { key: "b", label: "b", runtime: rt({ span_start_s: 30, span_end_s: 90 }) },
        { key: "a", label: "a", runtime: rt({ span_start_s: 0, span_end_s: 120, basis: "attributed" }) },
        { key: "c", label: "c", runtime: rt({ basis: "not_recorded" }) },
      ],
      45,
    );
    expect(layout.bars.map((b) => b.key)).toEqual(["__startup", "a", "b"]);
    expect(layout.bars[0]).toMatchObject({ kind: "startup", start: -45, end: 0 });
    expect(layout.domain).toEqual([-45, 120]);
    expect(layout.missing).toBe(1);
  });

  it("omits the startup bar when startup is unknown", () => {
    const layout = layoutTimeline([{ key: "a", label: "a", runtime: rt({ span_start_s: 5, span_end_s: 10 }) }], null);
    expect(layout.bars.map((b) => b.kind)).toEqual(["task"]);
    expect(layout.domain).toEqual([0, 10]);
  });

  it("computes shares and decides on a log axis", () => {
    expect(shares([1, 3, null])).toEqual([0.25, 0.75, null]);
    expect(shares([null])).toEqual([null]);
    expect(wantsLogScale([10, 100])).toBe(false);
    expect(wantsLogScale([10, 400])).toBe(true);
  });

  it("colors the most common families and groups the rest as other", () => {
    const { colorOf, legend } = familyColors(["a", "a", "b", "c"], 2);
    expect(legend.map((l) => l.label)).toEqual(["a", "b", "other"]);
    expect(colorOf("a")).toBe("var(--cat-1)");
    expect(colorOf("c")).toBe("var(--cat-context)");
  });
});

describe("runtime axis ticks", () => {
  it("uses human time steps", () => {
    expect(runtimeTicks(0, 600, 6)).toEqual([0, 120, 240, 360, 480, 600]);
    expect(runtimeTicks(0, 50, 6)).toEqual([0, 10, 20, 30, 40, 50]);
    expect(runtimeTicks(8, 4000, 6, true)).toEqual([10, 30, 120, 600, 1800]);
    expect(runtimeTicks(5, 5)).toEqual([5]);
  });
});
