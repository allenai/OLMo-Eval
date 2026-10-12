import { useEffect, useRef } from "react";
import { tinykeys } from "tinykeys";

/** True when the event target is a text field, so single-key shortcuts must not fire. */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "INPUT") {
    const type = (target as HTMLInputElement).type;
    return !["checkbox", "radio", "button", "range", "submit"].includes(type);
  }
  return false;
}

/** True when the target already uses arrow keys itself (sliders, tab lists, radio groups, menus). */
export function usesArrowKeys(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement && target.type === "range") return true;
  return !!target.closest('[role="tablist"],[role="radiogroup"],[role="slider"],[role="menu"],[role="listbox"],[role="tree"]');
}

/** True when a modal dialog (palette, sheet, overlay) is open. */
export function modalOpen(): boolean {
  return !!document.querySelector('[role="dialog"][data-state="open"], [data-modal-open="true"]');
}

type Bindings = Record<string, (event: KeyboardEvent) => void>;

/**
 * Register single-key shortcuts that ignore typing targets and open dialogs.
 * Handlers are read through a ref so callers can pass inline functions.
 */
export function useHotkeys(bindings: Bindings, options: { enabled?: boolean; allowInDialog?: boolean } = {}): void {
  const ref = useRef(bindings);
  useEffect(() => {
    ref.current = bindings;
  });
  const keys = Object.keys(bindings).join("|");
  const enabled = options.enabled ?? true;
  const allowInDialog = options.allowInDialog ?? false;
  useEffect(() => {
    if (!enabled) return;
    const wrapped: Bindings = {};
    for (const key of keys.split("|")) {
      if (!key) continue;
      wrapped[key] = (event) => {
        if (isTypingTarget(event.target)) return;
        if (key.startsWith("Arrow") && usesArrowKeys(event.target)) return;
        if (!allowInDialog && modalOpen()) return;
        if (event.metaKey || event.ctrlKey || event.altKey) {
          if (!/\$mod|Meta|Control|Alt/.test(key)) return;
        }
        ref.current[key]?.(event);
      };
    }
    return tinykeys(window, wrapped, { ignore: () => false });
  }, [keys, enabled, allowInDialog]);
}
