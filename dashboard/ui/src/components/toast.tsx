import { CheckCircle2, Info, XCircle } from "lucide-react";
import { createStore, useStore } from "@/state/store";
import s from "./ui.module.css";

export interface ToastItem {
  id: number;
  message: string;
  tone: "ok" | "info" | "error";
  action?: { label: string; onClick: () => void };
}

const toastStore = createStore<ToastItem[]>([]);
let nextId = 1;

export function toast(
  message: string,
  options: { tone?: ToastItem["tone"]; action?: ToastItem["action"]; duration?: number } = {},
): void {
  const id = nextId++;
  const item: ToastItem = { id, message, tone: options.tone ?? "ok", action: options.action };
  toastStore.set((prev) => [...prev.slice(-2), item]);
  setTimeout(() => toastStore.set((prev) => prev.filter((t) => t.id !== id)), options.duration ?? 2600);
}

export function Toaster() {
  const items = useStore(toastStore);
  return (
    <div className={s.toasts} role="status" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={s.toast}>
          {t.tone === "error" ? <XCircle /> : t.tone === "info" ? <Info /> : <CheckCircle2 />}
          <span>{t.message}</span>
          {t.action && (
            <button type="button" onClick={t.action.onClick}>
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
