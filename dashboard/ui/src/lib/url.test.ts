import { describe, expect, it } from "vitest";
import { apiQuery, joinList, oneOf, parseList, parseNumber, parseSearch, pickStrings, stringifySearch } from "./url";

describe("URL state", () => {
  it("keeps URLs readable: commas, colons and @ are not escaped", () => {
    expect(stringifySearch({ subjects: "r:abc123def456,m:0123456789ab", scope: "suite:olmes:base" })).toBe(
      "?subjects=r:abc123def456,m:0123456789ab&scope=suite:olmes:base",
    );
  });

  it("omits empty values so defaults stay out of the URL", () => {
    expect(stringifySearch({ a: undefined, b: "", c: null, d: "x" })).toBe("?d=x");
    expect(stringifySearch({})).toBe("");
  });

  it("round-trips through parse and stringify, including spaces and special characters", () => {
    const state = { q: "model:olmo3* user:me", task: "gsm8k:cot:olmo3", inst: "a&b=c", tag: "x+y" };
    expect(parseSearch(stringifySearch(state))).toEqual(state);
  });

  it("joins repeated keys and ignores malformed escapes", () => {
    expect(parseSearch("?model=a&model=b&bad=%E0%A4%A")).toEqual({ model: "a,b" });
  });

  it("ignores unknown params when picking route keys", () => {
    expect(pickStrings({ tab: "tasks", junk: "1", alpha: 0.05, obj: {} }, ["tab", "alpha", "inst"])).toEqual({ tab: "tasks", alpha: "0.05" });
  });

  it("parses lists, enums and numbers tolerantly", () => {
    expect(parseList(" a, b,,c ")).toEqual(["a", "b", "c"]);
    expect(joinList([])).toBeUndefined();
    expect(oneOf("bogus", ["a", "b"] as const, "a")).toBe("a");
    expect(parseNumber("0.1", 0.05)).toBe(0.1);
    expect(parseNumber("abc", 0.05)).toBe(0.05);
    expect(parseNumber("5", 1, { max: 2 })).toBe(1);
  });

  it("keeps commas and percent signs inside list items through the URL", () => {
    const items = ["org/model,v2", "50% mix", "plain"];
    const url = stringifySearch({ model: joinList(items) });
    expect(parseList(parseSearch(url).model)).toEqual(items);
    expect(parseList(joinList(["%2C literal"]))).toEqual(["%2C literal"]);
  });

  it("builds API queries with repeated keys", () => {
    expect(apiQuery({ model: ["a*", "b"], limit: 50, q: undefined })).toBe("?model=a*&model=b&limit=50");
  });
});
