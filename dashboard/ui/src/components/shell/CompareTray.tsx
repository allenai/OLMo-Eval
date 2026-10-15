import { ArrowRight, Plus, Star, X } from "lucide-react";
import { useRef, useState } from "react";
import { buildHref, useAppNavigate } from "../AppLink";
import { useHotkeys } from "@/lib/keyboard";
import { useSubjectSlots } from "@/state/colors";
import { useBaseline } from "@/state/nav";
import { clearTray, moveTrayItem, removeFromTray, trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { Button, cx, IconButton, ModelDot } from "../primitives";
import { openPalette } from "./registry";
import s from "./shell.module.css";

export function CompareTray() {
  const items = useStore(trayStore);
  const [baseline, setBaseline] = useBaseline();
  const navigate = useAppNavigate();
  const slots = useSubjectSlots(items.map((i) => i.key));
  const [drag, setDrag] = useState<number | null>(null);
  const lastG = useRef(0);
  const open = () => {
    const subjects = items.map((i) => i.key);
    if (baseline && !subjects.includes(baseline)) subjects.unshift(baseline);
    navigate(buildHref("/compare", { subjects: subjects.join(",") }));
  };
  useHotkeys(
    {
      // tinykeys fires both "g c" and "c" on the second key, so "c" right after "g" is ignored.
      g: () => {
        lastG.current = Date.now();
      },
      c: () => {
        if (Date.now() - lastG.current > 1000) open();
      },
    },
    { enabled: items.length > 0 },
  );
  if (!items.length) return null;
  return (
    <div className={s.tray} role="region" aria-label="Compare tray">
      <span className={s.trayLabel}>Compare</span>
      <div className={s.trayChips}>
        {items.map((item, idx) => {
          const isBase = item.key === baseline;
          return (
            <span
              key={item.key}
              className={cx(s.trayChip, isBase && s.trayChipBase, drag === idx && s.trayChipDrag)}
              draggable
              tabIndex={0}
              aria-label={`${item.label}, position ${idx + 1} of ${items.length}. Alt+Left or Alt+Right moves it.`}
              onKeyDown={(e) => {
                if (!e.altKey || e.target !== e.currentTarget) return;
                const to = e.key === "ArrowLeft" ? idx - 1 : e.key === "ArrowRight" ? idx + 1 : null;
                if (to == null || to < 0 || to >= items.length) return;
                e.preventDefault();
                moveTrayItem(idx, to);
              }}
              onDragStart={() => setDrag(idx)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => {
                if (drag != null && drag !== idx) moveTrayItem(drag, idx);
                setDrag(null);
              }}
              onDragEnd={() => setDrag(null)}
              title={item.label}
            >
              <ModelDot slot={slots[item.key]} baseline={isBase} />
              <span>{item.label}</span>
              <button
                type="button"
                className={cx(s.trayIcon, isBase && s.trayIconOn)}
                aria-label={isBase ? "Clear baseline" : "Set as baseline"}
                title={isBase ? "Clear baseline" : "Set as baseline"}
                onClick={() => setBaseline(isBase ? null : item.key, item.label)}
              >
                <Star fill={isBase ? "currentColor" : "none"} />
              </button>
              <button type="button" className={s.trayIcon} aria-label={`Remove ${item.label}`} onClick={() => removeFromTray(item.key)}>
                <X />
              </button>
            </span>
          );
        })}
      </div>
      <IconButton size="sm" label="Add a run or model" icon={<Plus />} onClick={() => openPalette("tray")} />
      <Button variant="primary" size="sm" onClick={open} disabled={items.length + (baseline && !items.some((i) => i.key === baseline) ? 1 : 0) < 2}>
        Compare {items.length}
        <ArrowRight />
      </Button>
      <IconButton size="sm" label="Clear tray" icon={<X />} onClick={clearTray} />
    </div>
  );
}
