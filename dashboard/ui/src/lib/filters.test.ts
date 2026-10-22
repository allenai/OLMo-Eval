import { describe, expect, it } from "vitest";
import { activeFilterCount, dateParam, filtersToApi, filtersToText, parseFilterText, removeFilterValue } from "./filters";

describe("runs filter grammar", () => {
  it("parses key:value tokens, negation, step ranges and free text", () => {
    expect(parseFilterText("model:olmo3* user:me -tag:debug step:>=1000 has:failures status:failed,partial midtrain")).toEqual({
      model: "olmo3*",
      user: "me",
      ntag: "debug",
      smin: "1000",
      fail: "1",
      status: "failed,partial",
      q: "midtrain",
    });
  });

  it("merges repeated list keys and maps workspace to ws", () => {
    expect(parseFilterText("model:a model:b workspace:ai2/olmo-eval")).toEqual({ model: "a,b", ws: "ai2/olmo-eval" });
  });

  it("treats unknown keys and empty values as free text", () => {
    expect(parseFilterText("foo:bar model:")).toEqual({ q: "foo:bar model:" });
  });

  it("round-trips through text", () => {
    const filters = parseFilterText("model:olmo3* -tag:debug step:>=1000 date:7d free words");
    expect(parseFilterText(filtersToText(filters))).toEqual(filters);
  });

  it("maps URL keys to API params", () => {
    expect(filtersToApi({ ws: "ai2/x", ntag: "a,b", smin: "10", fail: "1", sc: "any", suite: "olmes:base" })).toMatchObject({
      workspace: ["ai2/x"],
      not_tag: ["a", "b"],
      step_min: "10",
      has_failures: "true",
      suite_coverage: "any",
      suite: ["olmes:base"],
    });
  });

  it("sends date filters as full ISO datetimes at local midnight", () => {
    expect(dateParam("2026-09-01")).toBe(new Date(2026, 8, 1).toISOString());
    expect(dateParam("2026-09-01T12:00:00Z")).toBe("2026-09-01T12:00:00Z");
    expect(dateParam(undefined)).toBeUndefined();
    expect(filtersToApi({ after: "2026-09-01" }).after).toMatch(/^2026-0[89]-\d{2}T\d{2}:00:00\.000Z$/);
  });

  it("removes single values and counts active filters", () => {
    const f = { model: "a,b", suite: "s", sc: "any" };
    expect(removeFilterValue(f, "model", "a")).toEqual({ model: "b", suite: "s", sc: "any" });
    expect(removeFilterValue(f, "suite")).toEqual({ model: "a,b" });
    expect(activeFilterCount(f)).toBe(2);
  });
});
