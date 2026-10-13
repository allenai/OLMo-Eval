import { describe, expect, it } from "vitest";
import type { DeltaStats } from "@contract/api-types";
import { assignSlots } from "@/state/prefs";
import { deltaEvidence, divStep, goodness, normalize, ranks, seqStep } from "./scales";
import { jsonDiff, wordDiff } from "./diff";
import { parseSubjects } from "./subjects";

const stats = (over: Partial<DeltaStats>): DeltaStats => ({
  delta: 0.01,
  ci_low: 0,
  ci_high: 0.02,
  p_value: 0.04,
  n_shared: 100,
  method: "paired_bootstrap",
  significant: true,
  improved: true,
  hash_mismatch: false,
  alpha: 0.05,
  n_boot: 2000,
  note: null,
  ...over,
});

describe("color scales and significance", () => {
  it("maps values to sequential and diverging steps", () => {
    expect(seqStep(0)).toBe(0);
    expect(seqStep(1)).toBe(6);
    expect(divStep(0, 1)).toBe(0);
    expect(divStep(0.2, 1)).toBe(1);
    expect(divStep(-5, 1)).toBe(-3);
  });

  it("flips goodness for lower-is-better metrics", () => {
    expect(goodness(-0.1, false)).toBe(0.1);
    expect(goodness(0.1, true)).toBe(0.1);
  });

  it("classifies delta evidence", () => {
    expect(deltaEvidence(stats({}))).toBe("significant");
    expect(deltaEvidence(stats({ significant: false }))).toBe("ns");
    expect(deltaEvidence(stats({ method: "insufficient" }))).toBe("insufficient");
    expect(deltaEvidence(null)).toBe("none");
  });

  it("ranks in the better direction with ties sharing a rank", () => {
    expect(ranks([0.5, 0.7, 0.7, null], true)).toEqual([3, 1, 1, null]);
    expect(ranks([0.9, 0.7], false)).toEqual([2, 1]);
    expect(normalize(5, [0, 10])).toBe(0.5);
    expect(normalize(5, [5, 5])).toBe(0.5);
  });
});

describe("subject colors", () => {
  it("assigns slots in order and keeps remembered slots", () => {
    const first = assignSlots(["r:aaaaaa", "r:bbbbbb"], {});
    expect(first.slots).toEqual({ "r:aaaaaa": 0, "r:bbbbbb": 1 });
    const second = assignSlots(["r:bbbbbb", "r:cccccc"], first.updates);
    expect(second.slots["r:bbbbbb"]).toBe(1);
    expect(second.slots["r:cccccc"]).toBe(0);
  });

  it("gives no slot past six subjects and resolves conflicts", () => {
    const keys = Array.from({ length: 8 }, (_, i) => `r:subj0${i}`);
    const { slots } = assignSlots(keys, {});
    expect(Object.values(slots).filter((s) => s == null)).toHaveLength(2);
    const conflict = assignSlots(["r:aaaaaa", "r:bbbbbb"], { "r:aaaaaa": 2, "r:bbbbbb": 2 });
    expect(conflict.slots).toEqual({ "r:aaaaaa": 2, "r:bbbbbb": 0 });
  });

  it("parses subject lists, dropping invalid and duplicate keys", () => {
    expect(parseSubjects("r:abc123,m:0123456789ab,bad,r:abc123")).toEqual(["r:abc123", "m:0123456789ab"]);
  });
});

describe("diffs", () => {
  it("diffs words", () => {
    expect(wordDiff("the answer is 48", "the answer is 72").filter((s) => s.kind !== "same").map((s) => s.text)).toEqual(["48", "72"]);
  });

  it("diffs JSON structurally", () => {
    const entries = jsonDiff({ a: 1, b: { c: 2 }, d: [1, 2] }, { a: 1, b: { c: 3 }, e: true, d: [1, 2] });
    expect(entries.filter((e) => e.kind !== "same").map((e) => `${e.kind}:${e.path}`)).toEqual(["changed:b.c", "added:e"]);
  });
});
