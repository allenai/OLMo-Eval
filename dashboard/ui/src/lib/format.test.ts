import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatCI,
  formatCompact,
  formatDelta,
  formatDuration,
  formatP,
  formatRate,
  formatScore,
  formatStderr,
  formatStep,
  metricLabel,
  sig3,
} from "./format";

describe("format", () => {
  it("formats percent scores with one decimal and raw scores with 3 significant digits", () => {
    expect(formatScore(0.7412)).toBe("74.1");
    expect(formatScore(0.7412, "raw")).toBe("0.741");
    expect(formatScore(-21.7, "raw")).toBe("−21.7");
    expect(formatScore(1234.5, "raw")).toBe("1,235");
    expect(formatScore(null)).toBe("—");
  });

  it("formats deltas with a sign, in percentage points for percent metrics", () => {
    expect(formatDelta(0.013)).toBe("+1.3");
    expect(formatDelta(-0.024)).toBe("−2.4");
    expect(formatDelta(0.0001)).toBe("0.0");
    expect(formatDelta(0.0156, "raw")).toBe("+0.0156");
    expect(formatDelta(undefined)).toBe("—");
  });

  it("formats stderr, CIs and p-values", () => {
    expect(formatStderr(0.012)).toBe("±1.2");
    expect(formatStderr(null)).toBe("");
    expect(formatCI(0.002, 0.024)).toBe("[+0.2, +2.4]");
    expect(formatP(0.0004)).toBe("p<0.001");
    expect(formatP(0.0042)).toBe("p=0.004");
    expect(formatP(0.25)).toBe("p=0.25");
  });

  it("formats counts, rates, durations, bytes and steps", () => {
    expect(formatCompact(12431)).toBe("12.4k");
    expect(formatCompact(19_000_000)).toBe("19M");
    expect(formatRate(0.113)).toBe("11%");
    expect(formatRate(0.004)).toBe("<1%");
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(192)).toBe("3m 12s");
    expect(formatDuration(8040)).toBe("2h 14m");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatStep(14000)).toBe("14k");
    expect(formatStep(2400)).toBe("2.4k");
    expect(formatStep(1750)).toBe("1.75k");
    expect(formatStep(500)).toBe("500");
    expect(sig3(0.000123)).toBe("1.2e-4");
  });

  it("labels metric keys", () => {
    expect(metricLabel("exact_match:flex")).toBe("exact_match (flex)");
    expect(metricLabel("acc_raw:default")).toBe("acc_raw");
    expect(metricLabel(null)).toBe("—");
  });
});
