import type { SearchResultType } from "@contract/api-types";

export interface PaletteCommand {
  id: string;
  label: string;
  keywords?: string;
  hint?: string[];
  run: () => void;
}

export interface ParsedQuery {
  /** What is sent to the search API (prefix stripped). */
  text: string;
  /** Restrict to these result types; null means all. */
  types: SearchResultType[] | null;
  commandsOnly: boolean;
}

/** Prefixes: `>` commands, `@` users, `#` groups, `t:` tasks and suites, `m:` models. */
export function parsePaletteQuery(raw: string): ParsedQuery {
  const q = raw.trimStart();
  if (q.startsWith(">")) return { text: q.slice(1).trim(), types: [], commandsOnly: true };
  if (q.startsWith("@")) return { text: q.slice(1).trim(), types: ["user"], commandsOnly: false };
  if (q.startsWith("#")) return { text: q.slice(1).trim(), types: ["group"], commandsOnly: false };
  if (/^t:/i.test(q)) return { text: q.slice(2).trim(), types: ["task", "suite"], commandsOnly: false };
  if (/^m:/i.test(q)) return { text: q.slice(2).trim(), types: ["model"], commandsOnly: false };
  return { text: q.trim(), types: null, commandsOnly: false };
}

/** Fuzzy, colon-aware match: every query token must appear in order-insensitive substrings. */
export function fuzzyMatch(query: string, text: string): boolean {
  const tokens = query.toLowerCase().split(/[\s:]+/).filter(Boolean);
  if (!tokens.length) return true;
  const hay = text.toLowerCase();
  return tokens.every((t) => hay.includes(t));
}

export function matchCommands(query: string, commands: PaletteCommand[]): PaletteCommand[] {
  return commands.filter((c) => fuzzyMatch(query, `${c.label} ${c.keywords ?? ""}`));
}
