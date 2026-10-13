import * as RadixTooltip from "@radix-ui/react-tooltip";
import { Outlet, useRouterState } from "@tanstack/react-router";
import { useEffect } from "react";
import { copyText } from "@/lib/csv";
import { useHotkeys } from "@/lib/keyboard";
import { currentSubjectStore, useBaseline } from "@/state/nav";
import { applyDensity, applyTheme, densityStore, themeStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { useAppNavigate } from "../AppLink";
import { toast, Toaster } from "../toast";
import { CommandPalette } from "./CommandPalette";
import { CompareTray } from "./CompareTray";
import { openPalette, pageActions, shortcutsStore } from "./registry";
import { ShortcutOverlay } from "./ShortcutOverlay";
import s from "./shell.module.css";
import { TopBar } from "./TopBar";

export function AppShell() {
  const theme = useStore(themeStore);
  const density = useStore(densityStore);
  const navigate = useAppNavigate();
  const [, setBaseline] = useBaseline();
  const pathname = useRouterState({ select: (st) => st.location.pathname });

  useEffect(() => applyTheme(theme), [theme]);
  useEffect(() => applyDensity(density), [density]);
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [pathname]);

  // Cmd/Ctrl+K works everywhere, including inside text fields.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        openPalette();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useHotkeys({
    "g h": () => navigate("/"),
    "g r": () => navigate("/runs"),
    "g c": () => navigate("/compare"),
    "g m": () => navigate("/models"),
    "g t": () => navigate("/tasks"),
    "g g": () => navigate("/groups"),
    "Shift+?": () => shortcutsStore.set(true),
    "?": () => shortcutsStore.set(true),
    "/": (e) => {
      if (pageActions.focusFilter) {
        e.preventDefault();
        pageActions.focusFilter();
      } else {
        e.preventDefault();
        openPalette();
      }
    },
    b: () => {
      const current = currentSubjectStore.get();
      if (current) {
        setBaseline(current.key, current.label);
        toast(`Baseline set to ${current.label}`);
      } else openPalette("baseline");
    },
    "Shift+B": () => {
      setBaseline(null);
      toast("Baseline cleared");
    },
    y: () => {
      copyText(window.location.href).then((ok) => ok && toast("Link to this view copied"));
    },
  });

  return (
    <RadixTooltip.Provider delayDuration={250} skipDelayDuration={100}>
      <div className={s.app}>
        <a href="#main" className="sr-only">
          Skip to content
        </a>
        <TopBar />
        <main id="main" className={s.main}>
          <Outlet />
        </main>
        <CompareTray />
        <CommandPalette />
        <ShortcutOverlay />
        <Toaster />
      </div>
    </RadixTooltip.Provider>
  );
}
