import type { SearchResult, SubjectKey } from "@contract/api-types";
import { Command } from "cmdk";
import {
  ArrowRightLeft,
  Boxes,
  Clock,
  Command as CommandIcon,
  Cpu,
  Fingerprint,
  FolderKanban,
  ListChecks,
  Play,
  Search,
  User,
} from "lucide-react";
import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { useSearch } from "@/api/hooks/catalog";
import { copyText } from "@/lib/csv";
import { useBaseline } from "@/state/nav";
import { addToTray, cycleTheme, densityStore, lastBaselineStore, recentStore, trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { buildHref, useAppNavigate } from "../AppLink";
import { Kbd } from "../primitives";
import { toast } from "../toast";
import { matchCommands, type PaletteCommand, parsePaletteQuery } from "./paletteLogic";
import { closePalette, openPalette, pageActions, paletteStore, shortcutsStore } from "./registry";
import s from "./shell.module.css";
import ui from "../ui.module.css";

const TYPE_LABEL: Record<string, string> = {
  id: "Matching IDs",
  run: "Runs",
  model: "Models",
  task: "Tasks",
  suite: "Suites",
  group: "Groups",
  user: "Users",
};

const TYPE_ICON: Record<string, ReactNode> = {
  id: <Fingerprint />,
  run: <Play />,
  model: <Cpu />,
  task: <ListChecks />,
  suite: <Boxes />,
  group: <FolderKanban />,
  user: <User />,
};

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

interface Entry {
  value: string;
  label: string;
  secondary?: string | null;
  icon: ReactNode;
  href?: string;
  subject?: SubjectKey | null;
  run?: () => void;
  hint?: string[];
}

export function CommandPalette() {
  const state = useStore(paletteStore);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState("");
  const navigate = useAppNavigate();
  const [baseline, setBaseline] = useBaseline();
  const recents = useStore(recentStore);
  const tray = useStore(trayStore);
  const lastBaseline = useStore(lastBaselineStore);

  useEffect(() => {
    // Reset the input to the requested prefill and clear the selection each time the palette opens.
    if (state.open) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setQuery(state.query);
      setSelected("");
    }
  }, [state.open, state.query]);

  const parsed = parsePaletteQuery(query);
  const debounced = useDebounced(parsed.text, 120);
  const search = useSearch(state.open && !parsed.commandsOnly ? debounced : "");

  const go = useCallback(
    (href: string) => {
      closePalette();
      navigate(href);
    },
    [navigate],
  );

  const commands: PaletteCommand[] = useMemo(
    () => [
      {
        id: "compare",
        label: "Compare selected",
        keywords: "tray open compare",
        hint: ["c"],
        run: () => {
          const subjects = trayStore.get().map((i) => i.key);
          if (subjects.length < 1) {
            toast("Add runs to the compare tray first", { tone: "info" });
            return;
          }
          go(buildHref("/compare", { subjects: subjects.join(",") }));
        },
      },
      { id: "baseline", label: "Set baseline…", keywords: "reference", hint: ["b"], run: () => openPalette("baseline") },
      {
        id: "clear-baseline",
        label: "Clear baseline",
        hint: ["⇧", "B"],
        run: () => {
          setBaseline(null);
          closePalette();
        },
      },
      {
        id: "copy-link",
        label: "Copy link to this view",
        keywords: "share url",
        hint: ["y"],
        run: () => {
          copyText(window.location.href).then((ok) => ok && toast("Link copied"));
          closePalette();
        },
      },
      { id: "theme", label: "Toggle theme", keywords: "dark light", run: () => (cycleTheme(), closePalette()) },
      {
        id: "density",
        label: "Toggle row density",
        keywords: "compact comfortable",
        run: () => {
          densityStore.set((d) => (d === "compact" ? "comfortable" : "compact"));
          closePalette();
        },
      },
      { id: "my-runs", label: "Go to my runs", keywords: "mine", run: () => go("/runs?user=me") },
      {
        id: "save-view",
        label: "Save view",
        keywords: "bookmark",
        run: () => {
          closePalette();
          if (pageActions.saveView) pageActions.saveView();
          else toast("Saved views are available on the Runs page", { tone: "info" });
        },
      },
      {
        id: "export",
        label: "Export table as CSV",
        keywords: "download",
        run: () => {
          closePalette();
          if (pageActions.exportTable) pageActions.exportTable();
          else toast("This page has no exportable table", { tone: "info" });
        },
      },
      { id: "shortcuts", label: "Show keyboard shortcuts", hint: ["?"], run: () => (closePalette(), shortcutsStore.set(true)) },
      { id: "go-home", label: "Go to Home", hint: ["g", "h"], run: () => go("/") },
      { id: "go-runs", label: "Go to Runs", hint: ["g", "r"], run: () => go("/runs") },
      { id: "go-compare", label: "Go to Compare", hint: ["g", "c"], run: () => go("/compare") },
      { id: "go-models", label: "Go to Models", hint: ["g", "m"], run: () => go("/models") },
      { id: "go-tasks", label: "Go to Tasks", hint: ["g", "t"], run: () => go("/tasks") },
      { id: "go-groups", label: "Go to Groups", hint: ["g", "g"], run: () => go("/groups") },
      { id: "go-views", label: "Go to saved views", run: () => go("/views") },
    ],
    [go, setBaseline],
  );

  const pickMode = state.mode !== "default";
  const entries: { heading: string; items: Entry[] }[] = [];
  const resultEntry = (r: SearchResult): Entry => ({
    value: `${r.type}:${r.key}`,
    label: r.label,
    secondary: r.secondary,
    icon: TYPE_ICON[r.type],
    href: r.href,
    subject: r.subject,
  });

  if (!parsed.text && !parsed.commandsOnly) {
    if (pickMode) {
      const suggestions: Entry[] = [];
      if (lastBaseline && state.mode === "baseline") {
        suggestions.push({ value: `s:${lastBaseline.key}`, label: lastBaseline.label, secondary: "Your last baseline", icon: <Clock />, subject: lastBaseline.key });
      }
      for (const t of tray) suggestions.push({ value: `t:${t.key}`, label: t.label, secondary: "In compare tray", icon: <ArrowRightLeft />, subject: t.key });
      for (const r of recents.filter((x) => x.type === "run").slice(0, 6)) {
        suggestions.push({ value: `r:${r.key}`, label: r.label, secondary: r.secondary ?? "Recently viewed", icon: <Clock />, subject: `r:${r.key}` });
      }
      if (suggestions.length) entries.push({ heading: "Suggestions", items: suggestions });
    } else {
      const recentItems = recents.slice(0, 8).map((r) => ({
        value: `recent:${r.type}:${r.key}`,
        label: r.label,
        secondary: r.secondary ?? r.type,
        icon: TYPE_ICON[r.type] ?? <Clock />,
        href: r.href,
        subject: r.type === "run" ? (`r:${r.key}` as SubjectKey) : null,
      }));
      if (recentItems.length) entries.push({ heading: "Recent", items: recentItems });
    }
  }

  if (!parsed.commandsOnly && parsed.text) {
    for (const group of search.data?.groups ?? []) {
      if (parsed.types && !parsed.types.includes(group.type)) continue;
      if (pickMode && !group.items.some((i) => i.subject)) continue;
      const items = group.items.filter((i) => !pickMode || i.subject).map(resultEntry);
      if (items.length) entries.push({ heading: TYPE_LABEL[group.type] ?? group.type, items });
    }
  }

  if (!pickMode && (parsed.commandsOnly || !parsed.types)) {
    const cmds = matchCommands(parsed.text, commands).map((c) => ({
      value: `cmd:${c.id}`,
      label: c.label,
      icon: <CommandIcon />,
      run: c.run,
      hint: c.hint,
    }));
    if (cmds.length) entries.push({ heading: "Commands", items: parsed.text || parsed.commandsOnly ? cmds : cmds.slice(0, 8) });
  }

  const allItems = entries.flatMap((g) => g.items);
  // A selection left over from an earlier query matches nothing, and cmdk would then highlight
  // nothing and ignore Enter. Fall back to the first item.
  const current = allItems.some((i) => i.value === selected) ? selected : (allItems[0]?.value ?? "");

  const activate = (entry: Entry, modifier: "none" | "tray" | "baseline" = "none") => {
    if (state.mode === "baseline" || modifier === "baseline") {
      if (entry.subject) {
        setBaseline(entry.subject, entry.label);
        toast(`Baseline set to ${entry.label}`);
        closePalette();
      }
      return;
    }
    if (state.mode === "tray" || modifier === "tray") {
      if (entry.subject) {
        addToTray({ key: entry.subject, label: entry.label });
        toast(`Added ${entry.label} to compare`);
        if (state.mode !== "tray") return;
        closePalette();
      }
      return;
    }
    if (entry.run) entry.run();
    else if (entry.href) go(entry.href);
  };

  const placeholder =
    state.mode === "baseline"
      ? "Pick a baseline: search runs and models…"
      : state.mode === "tray"
        ? "Add to compare: search runs and models…"
        : "Search runs, models, tasks, groups, IDs, or type > for commands";

  return (
    <Command.Dialog
      open={state.open}
      onOpenChange={(open) => (open ? undefined : closePalette())}
      label="Command palette"
      shouldFilter={false}
      value={current}
      onValueChange={setSelected}
      overlayClassName={ui.overlay}
      contentClassName={s.palette}
      onKeyDown={(e) => {
        if (e.key !== "Enter") return;
        const entry = allItems.find((i) => i.value === current);
        if (!entry) return;
        if (e.metaKey || e.ctrlKey) {
          e.preventDefault();
          activate(entry, "tray");
        } else if (e.shiftKey) {
          e.preventDefault();
          activate(entry, "baseline");
        }
      }}
    >
      <div className={s.paletteInputRow}>
        <Search />
        <Command.Input className={s.paletteInput} value={query} onValueChange={setQuery} placeholder={placeholder} autoFocus />
        {baseline && state.mode === "baseline" && <span className="t-caption">current: set</span>}
        <Kbd>esc</Kbd>
      </div>
      <Command.List className={s.paletteList}>
        {search.isFetching && parsed.text && !entries.length && <div className={s.paletteEmpty}>Searching…</div>}
        {!search.isFetching && !entries.length && (
          <Command.Empty className={s.paletteEmpty}>
            {parsed.text ? `No matches for “${parsed.text}”.` : "Type to search."}
          </Command.Empty>
        )}
        {entries.map((group) => (
          <Command.Group key={group.heading} heading={group.heading} className={s.paletteGroup}>
            {group.items.map((entry) => (
              <Command.Item key={entry.value} value={entry.value} className={s.paletteItem} onSelect={() => activate(entry)}>
                {entry.icon}
                <span className={s.paletteText}>
                  <span className={s.paletteLabel}>{entry.label}</span>
                  {entry.secondary && <span className={s.paletteSecondary}>{entry.secondary}</span>}
                </span>
                {entry.hint && (
                  <span className={s.paletteHint}>
                    {entry.hint.map((h, i) => (
                      <Kbd key={i}>{h}</Kbd>
                    ))}
                  </span>
                )}
              </Command.Item>
            ))}
          </Command.Group>
        ))}
      </Command.List>
      <div className={s.paletteFooter}>
        <span>
          <Kbd>↵</Kbd> open
        </span>
        <span>
          <Kbd>⌘</Kbd>
          <Kbd>↵</Kbd> add to compare
        </span>
        <span>
          <Kbd>⇧</Kbd>
          <Kbd>↵</Kbd> set baseline
        </span>
        <span className="hide-sm">
          <span className="mono">&gt;</span> commands · <span className="mono">@</span> users · <span className="mono">#</span> groups ·{" "}
          <span className="mono">t:</span> tasks · <span className="mono">m:</span> models
        </span>
      </div>
    </Command.Dialog>
  );
}
