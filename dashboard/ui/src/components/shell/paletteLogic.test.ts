import { describe, expect, it, vi } from "vitest";
import { fuzzyMatch, matchCommands, parsePaletteQuery } from "./paletteLogic";

describe("command palette", () => {
  it("parses type prefixes", () => {
    expect(parsePaletteQuery("> theme")).toEqual({ text: "theme", types: [], commandsOnly: true });
    expect(parsePaletteQuery("@chr")).toMatchObject({ text: "chr", types: ["user"] });
    expect(parsePaletteQuery("#olmo")).toMatchObject({ types: ["group"] });
    expect(parsePaletteQuery("t:gsm")).toMatchObject({ text: "gsm", types: ["task", "suite"] });
    expect(parsePaletteQuery("m:olmo")).toMatchObject({ types: ["model"] });
    expect(parsePaletteQuery("olmo3")).toMatchObject({ types: null, commandsOnly: false });
  });

  it("matches colon-aware fuzzy tokens", () => {
    expect(fuzzyMatch("gsm cot", "gsm8k:cot:olmo3")).toBe(true);
    expect(fuzzyMatch("gsm:mbpp", "gsm8k:cot:olmo3")).toBe(false);
  });

  it("filters commands by label and keywords", () => {
    const run = vi.fn();
    const cmds = [
      { id: "theme", label: "Toggle theme", keywords: "dark light", run },
      { id: "copy", label: "Copy link", run },
    ];
    expect(matchCommands("dark", cmds).map((c) => c.id)).toEqual(["theme"]);
    expect(matchCommands("", cmds)).toHaveLength(2);
  });
});
