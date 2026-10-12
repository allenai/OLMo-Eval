import * as Dialog from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { useStore } from "@/state/store";
import { IconButton, Kbd } from "../primitives";
import { shortcutsStore } from "./registry";
import s from "./shell.module.css";
import ui from "../ui.module.css";

const GROUPS: { title: string; rows: [string[], string][] }[] = [
  {
    title: "Global",
    rows: [
      [["⌘", "K"], "Command palette"],
      [["/"], "Focus filter or search"],
      [["?"], "This overlay"],
      [["b"], "Set current run as baseline"],
      [["⇧", "B"], "Clear baseline"],
      [["y"], "Copy link to this view"],
      [["esc"], "Close drawer, popover, palette"],
    ],
  },
  {
    title: "Go to",
    rows: [
      [["g", "h"], "Home"],
      [["g", "r"], "Runs"],
      [["g", "c"], "Compare"],
      [["g", "m"], "Models"],
      [["g", "t"], "Tasks"],
      [["g", "g"], "Groups"],
    ],
  },
  {
    title: "Tables",
    rows: [
      [["j"], "Next row"],
      [["k"], "Previous row"],
      [["x"], "Select row (adds to compare)"],
      [["⇧", "click"], "Range select"],
      [["↵"], "Open row"],
      [["c"], "Compare selection"],
      [["e"], "Expand or collapse groups"],
    ],
  },
  {
    title: "Run detail and instances",
    rows: [
      [["1", "…", "7"], "Switch tab"],
      [["[", "]"], "Previous or next task"],
      [["←", "→"], "Previous or next instance"],
      [["d"], "Side-by-side with baseline"],
      [["f"], "Only changed vs baseline"],
    ],
  },
];

export function ShortcutOverlay() {
  const open = useStore(shortcutsStore);
  return (
    <Dialog.Root open={open} onOpenChange={(o) => shortcutsStore.set(o)}>
      <Dialog.Portal>
        <Dialog.Overlay className={ui.overlay} />
        <Dialog.Content className={ui.dialog} style={{ width: "min(760px, calc(100vw - 32px))" }} aria-describedby={undefined}>
          <div className="row" style={{ marginBottom: 14 }}>
            <Dialog.Title className={ui.dialogTitle} style={{ margin: 0 }}>
              Keyboard shortcuts
            </Dialog.Title>
            <span className="spacer" />
            <Dialog.Close asChild>
              <IconButton label="Close" icon={<X />} />
            </Dialog.Close>
          </div>
          <div className={s.shortcuts}>
            {GROUPS.map((g) => (
              <div key={g.title} className={s.shortcutGroup}>
                <h3>{g.title}</h3>
                {g.rows.map(([keys, label]) => (
                  <div key={label} className={s.shortcutRow}>
                    <span>{label}</span>
                    <span className={s.shortcutKeys}>
                      {keys.map((k, i) => (k === "…" ? <span key={i}>…</span> : <Kbd key={i}>{k}</Kbd>))}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p className="t-caption" style={{ marginTop: 14 }}>
            Single-key shortcuts never fire while you are typing in a field.
          </p>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
