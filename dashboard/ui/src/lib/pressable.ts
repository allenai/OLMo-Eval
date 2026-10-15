import type { KeyboardEvent } from "react";

/**
 * Props that make a non-button element work like a button: focusable with Tab and activated
 * by Enter or Space as well as by a click. Keys pressed on a nested control are left to it.
 */
export function pressable(onActivate: () => void, role: "button" | "option" = "button") {
  return {
    role,
    tabIndex: 0,
    onClick: onActivate,
    onKeyDown: (e: KeyboardEvent<HTMLElement>) => {
      if (e.target !== e.currentTarget) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onActivate();
      }
    },
  };
}
