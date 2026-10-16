import { createStore } from "@/state/store";

/** UI-wide toggles shared by the shell and pages. */
export const paletteStore = createStore<{ open: boolean; mode: "default" | "baseline" | "tray"; query: string }>({
  open: false,
  mode: "default",
  query: "",
});

export const shortcutsStore = createStore<boolean>(false);

export function openPalette(mode: "default" | "baseline" | "tray" = "default", query = ""): void {
  paletteStore.set({ open: true, mode, query });
}

export function closePalette(): void {
  paletteStore.set((prev) => ({ ...prev, open: false }));
}

/** Page-level actions the palette can trigger (registered by the page that supports them). */
export const pageActions: { exportTable?: () => void; saveView?: () => void; focusFilter?: () => void } = {};
