import type { SubjectInfo } from "@contract/api-types";
import { describe, expect, it } from "vitest";
import { brushesParam, parseBrushes } from "./ProfilesView";
import { distinctLabels } from "./types";

describe("compare state", () => {
  it("round-trips Profiles brushes through the URL, keeping colons in axis keys", () => {
    const param = brushesParam({ "task:gsm8k:cot": [80, 20], "suite:olmes:base": [10, 40] });
    expect(param).toBe("20.0~80.0~task:gsm8k:cot,10.0~40.0~suite:olmes:base");
    expect(parseBrushes(param)).toEqual({ "task:gsm8k:cot": [20, 80], "suite:olmes:base": [10, 40] });
    expect(parseBrushes("bad,1~x~k,~~")).toEqual({});
    expect(brushesParam({})).toBeUndefined();
  });

  it("appends a model hash when two checkpoints share a label", () => {
    const info = (key: string, hash: string): SubjectInfo =>
      ({ key, kind: "model", label: key, model: { step: 8000, series_label: "olmo3", model_hash: hash } }) as unknown as SubjectInfo;
    const map = new Map([
      ["m:a", info("m:a", "3862aaaa")],
      ["m:b", info("m:b", "bd62bbbb")],
    ]);
    const labels = distinctLabels(["m:a", "m:b", "r:unknown"], map);
    expect(labels.get("m:a")).toBe("olmo3 @8k ·3862");
    expect(labels.get("m:b")).toBe("olmo3 @8k ·bd62");
    expect(labels.get("r:unknown")).toBe("r:unknown");
  });
});
