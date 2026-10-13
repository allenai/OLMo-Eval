/**
 * URL state helpers. Every search param is a plain string in the URL so links stay short and
 * human-readable (`subjects=r:abc,r:def`), lists are comma-separated, and defaults are omitted.
 */

export type SearchRecord = Record<string, string | undefined>;

// Characters that are safe to leave unescaped in a query value and make URLs readable.
const READABLE: [RegExp, string][] = [
  [/%2C/gi, ","],
  [/%3A/gi, ":"],
  [/%40/gi, "@"],
  [/%2F/gi, "/"],
  [/%2A/gi, "*"],
  [/%3E/gi, ">"],
  [/%3C/gi, "<"],
  [/%3D/gi, "="],
];

function encodeValue(value: string): string {
  let out = encodeURIComponent(value);
  for (const [pattern, replacement] of READABLE) out = out.replace(pattern, replacement);
  return out;
}

/** Router `stringifySearch`: drops empty values, keeps insertion order. */
export function stringifySearch(search: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [key, raw] of Object.entries(search)) {
    if (raw === undefined || raw === null || raw === "") continue;
    const value = Array.isArray(raw) ? raw.join(",") : String(raw);
    if (!value) continue;
    parts.push(`${encodeURIComponent(key)}=${encodeValue(value)}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

/** Router `parseSearch`: every value stays a string; repeated keys join with commas. */
export function parseSearch(searchStr: string): Record<string, string> {
  const query = searchStr.startsWith("?") ? searchStr.slice(1) : searchStr;
  const out: Record<string, string> = {};
  if (!query) return out;
  for (const part of query.split("&")) {
    if (!part) continue;
    const eq = part.indexOf("=");
    const rawKey = eq === -1 ? part : part.slice(0, eq);
    const rawValue = eq === -1 ? "" : part.slice(eq + 1);
    let key: string;
    let value: string;
    try {
      key = decodeURIComponent(rawKey.replace(/\+/g, " "));
      value = decodeURIComponent(rawValue.replace(/\+/g, " "));
    } catch {
      continue;
    }
    out[key] = key in out && value ? `${out[key]},${value}` : value;
  }
  return out;
}

/** Keep only string values of the listed keys (unknown params are ignored). */
export function pickStrings<K extends string>(
  raw: Record<string, unknown>,
  keys: readonly K[],
): Partial<Record<K, string>> {
  const out: Partial<Record<K, string>> = {};
  for (const key of keys) {
    const value = raw[key];
    if (typeof value === "string" && value !== "") out[key] = value;
    else if (typeof value === "number" || typeof value === "boolean") out[key] = String(value);
  }
  return out;
}

/** Split a comma-separated list param. Items written by `joinList` may contain commas. */
export function parseList(value: string | undefined | null): string[] {
  if (!value) return [];
  return value
    .split(",")
    .map((v) => v.trim().replace(/%(2C|25)/gi, (m) => (m.toUpperCase() === "%2C" ? "," : "%")))
    .filter(Boolean);
}

/** Join list items into one param, escaping commas (and %) inside items. */
export function joinList(values: readonly string[]): string | undefined {
  const clean = values.filter(Boolean).map((v) => v.replace(/%/g, "%25").replace(/,/g, "%2C"));
  return clean.length ? clean.join(",") : undefined;
}

export function oneOf<T extends string>(
  value: string | undefined,
  options: readonly T[],
  fallback: T,
): T {
  return value && (options as readonly string[]).includes(value) ? (value as T) : fallback;
}

export function parseNumber(
  value: string | undefined,
  fallback: number,
  { min = -Infinity, max = Infinity }: { min?: number; max?: number } = {},
): number {
  if (value == null || value === "") return fallback;
  const n = Number(value);
  if (!Number.isFinite(n) || n < min || n > max) return fallback;
  return n;
}

export function parseOptionalNumber(value: string | undefined): number | undefined {
  if (value == null || value === "") return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

export function parseBool(value: string | undefined, fallback: boolean): boolean {
  if (value === "1" || value === "true") return true;
  if (value === "0" || value === "false") return false;
  return fallback;
}

/** Sort param "-key" means descending. */
export function parseSort(value: string | undefined): { key: string; desc: boolean } | null {
  if (!value) return null;
  return value.startsWith("-") ? { key: value.slice(1), desc: true } : { key: value, desc: false };
}

export function sortParam(key: string, desc: boolean): string {
  return desc ? `-${key}` : key;
}

/** Build a query string for API calls; arrays repeat the key. */
export function apiQuery(params: Record<string, unknown>): string {
  const usp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const v of value) if (v !== undefined && v !== null && v !== "") usp.append(key, String(v));
    } else {
      usp.append(key, String(value));
    }
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}
