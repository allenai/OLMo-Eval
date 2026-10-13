import type { SubjectKey } from "@contract/api-types";
import { createStore, useStore } from "./store";

// ---------------------------------------------------------------- theme and density

export type ThemePref = "light" | "dark" | "system";
export const themeStore = createStore<ThemePref>("system", "oe.theme");

export function applyTheme(pref: ThemePref): void {
  const root = document.documentElement;
  if (pref === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", pref);
}

export function resolvedTheme(pref: ThemePref): "light" | "dark" {
  if (pref !== "system") return pref;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function cycleTheme(): void {
  themeStore.set((prev) => {
    const current = resolvedTheme(prev);
    return current === "dark" ? "light" : "dark";
  });
}

export type Density = "compact" | "comfortable";
export const densityStore = createStore<Density>("compact", "oe.density");

export function applyDensity(d: Density): void {
  document.documentElement.setAttribute("data-density", d);
}

// ---------------------------------------------------------------- subject colors

/**
 * Stable subject → categorical slot (0..5) for the session. Slots are assigned on first sight
 * and never repainted by filtering. Slots free up only when the map is reset.
 */
export const colorStore = createStore<Record<string, number>>({}, "oe.colors");

/**
 * Assign categorical slots to the subjects shown together, in order. A subject keeps its
 * remembered slot unless another subject earlier in the list already holds it. Subjects past the
 * sixth get no slot and render in the muted "context" color.
 */
export function assignSlots(
  keys: string[],
  prefs: Record<string, number>,
): { slots: Record<string, number | null>; updates: Record<string, number> } {
  const slots: Record<string, number | null> = {};
  const updates: Record<string, number> = {};
  const taken = new Set<number>();
  const pending: string[] = [];
  for (const key of keys) {
    if (key in slots) continue;
    const pref = prefs[key];
    if (pref != null && pref >= 0 && pref < 6 && !taken.has(pref)) {
      slots[key] = pref;
      taken.add(pref);
    } else {
      slots[key] = null;
      pending.push(key);
    }
  }
  for (const key of pending) {
    let slot = -1;
    for (let s = 0; s < 6; s++) {
      if (!taken.has(s)) {
        slot = s;
        break;
      }
    }
    if (slot === -1) break;
    slots[key] = slot;
    taken.add(slot);
    if (prefs[key] !== slot) updates[key] = slot;
  }
  return { slots, updates };
}

export function rememberSlots(updates: Record<string, number>): void {
  if (!Object.keys(updates).length) return;
  colorStore.set((prev) => {
    const entries = Object.entries({ ...prev, ...updates });
    return Object.fromEntries(entries.slice(-200));
  });
}

export function useColorMap(): Record<string, number> {
  return useStore(colorStore);
}

// ---------------------------------------------------------------- compare tray

export interface TrayItem {
  key: SubjectKey;
  label: string;
}

export const trayStore = createStore<TrayItem[]>([], "oe.tray");

export function addToTray(items: TrayItem | TrayItem[]): void {
  const list = Array.isArray(items) ? items : [items];
  trayStore.set((prev) => {
    const next = [...prev];
    for (const item of list) if (!next.some((p) => p.key === item.key)) next.push(item);
    return next.slice(0, 30);
  });
}

export function removeFromTray(key: SubjectKey): void {
  trayStore.set((prev) => prev.filter((p) => p.key !== key));
}

export function moveTrayItem(from: number, to: number): void {
  trayStore.set((prev) => {
    const next = [...prev];
    const [item] = next.splice(from, 1);
    next.splice(to, 0, item);
    return next;
  });
}

export function clearTray(): void {
  trayStore.set([]);
}

// ---------------------------------------------------------------- recents

export interface RecentItem {
  type: "run" | "model" | "task" | "suite" | "group" | "compare";
  key: string;
  label: string;
  href: string;
  secondary?: string;
  at: number;
}

export const recentStore = createStore<RecentItem[]>([], "oe.recents");

export function pushRecent(item: Omit<RecentItem, "at">): void {
  recentStore.set((prev) => {
    const filtered = prev.filter((p) => !(p.type === item.type && p.key === item.key));
    return [{ ...item, at: Date.now() }, ...filtered].slice(0, 30);
  });
}

// ---------------------------------------------------------------- misc per-viewer conveniences

export const lastBaselineStore = createStore<{ key: SubjectKey; label: string } | null>(
  null,
  "oe.lastBaseline",
);

export const columnWidthStore = createStore<Record<string, Record<string, number>>>({}, "oe.colWidths");
