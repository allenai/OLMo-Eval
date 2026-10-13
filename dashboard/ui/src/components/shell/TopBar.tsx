import { useRouterState } from "@tanstack/react-router";
import { Bookmark, Keyboard, Menu as MenuIcon, Moon, Rows3, Search, Sun, User, X } from "lucide-react";
import { useMe } from "@/api/hooks/catalog";
import { useResolveSubjects } from "@/api/hooks/compare";
import { useBaseline } from "@/state/nav";
import { cycleTheme, densityStore, resolvedTheme, themeStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { AppLink, useAppNavigate } from "../AppLink";
import { cx, IconButton, Kbd, Menu, ModelDot, Tip } from "../primitives";
import { UserAvatar } from "../UserAvatar";
import { openPalette, shortcutsStore } from "./registry";
import s from "./shell.module.css";

const NAV = [
  { href: "/", label: "Home", match: (p: string) => p === "/" },
  { href: "/runs", label: "Runs", match: (p: string) => p.startsWith("/runs") },
  { href: "/compare", label: "Compare", match: (p: string) => p.startsWith("/compare") },
  { href: "/models", label: "Models", match: (p: string) => p.startsWith("/models") },
  { href: "/tasks", label: "Tasks", match: (p: string) => p.startsWith("/tasks") || p.startsWith("/suites") },
  { href: "/groups", label: "Groups", match: (p: string) => p.startsWith("/groups") },
];

/** The Ai2 mark, drawn in the text color (same as the go-links site). */
function Logo() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden focusable="false">
      <path
        d="M19.2,18.73h-9.6v-9.13h7.73c1.03,0,1.87-.84,1.87-1.87V0h9.13v9.6c0,5.06-4.1,9.13-9.13,9.13ZM9.6,19.67H0v9.13h7.73c1.03,0,1.87.84,1.87,1.87v7.73h9.13v-9.6c0-5.06-4.1-9.13-9.13-9.13ZM40.3,19.2c-1.03,0-1.87-.84-1.87-1.87v-7.73h-9.13v9.6c0,5.06,4.1,9.13,9.13,9.13h9.6v-9.13h-7.73ZM19.67,38.4v9.6h9.13v-7.73c0-1.03.84-1.87,1.87-1.87h7.73v-9.13h-9.6c-5.06,0-9.13,4.1-9.13,9.13Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function BaselineChip() {
  const [baseline, setBaseline] = useBaseline();
  const resolved = useResolveSubjects(baseline ? [baseline] : []);
  const info = resolved.data?.items.find((i) => i.key === baseline);
  if (!baseline) {
    return (
      <Tip content="Pick a run or model as the reference for deltas on every page (b)">
        <button type="button" className={cx(s.baseline, s.baselineEmpty)} onClick={() => openPalette("baseline")}>
          <span className={s.baselineKey}>Baseline</span>
          <span className={s.baselineLabelWrap}>none</span>
        </button>
      </Tip>
    );
  }
  return (
    <span className={s.baseline}>
      <button type="button" className={s.baselineMain} onClick={() => openPalette("baseline")} title="Change baseline">
        <ModelDot baseline />
        <span className={s.baselineKey}>Base</span>
        <span className={cx(s.baselineLabel, s.baselineLabelWrap)}>{info?.label ?? baseline}</span>
      </button>
      <IconButton
        size="sm"
        label="Clear baseline (Shift+B)"
        icon={<X />}
        onClick={(e) => {
          e.stopPropagation();
          setBaseline(null);
        }}
        style={{ width: 20, height: 20 }}
      />
    </span>
  );
}

export function TopBar() {
  const pathname = useRouterState({ select: (st) => st.location.pathname });
  const theme = useStore(themeStore);
  const density = useStore(densityStore);
  const me = useMe();
  const navigate = useAppNavigate();
  const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);
  return (
    <header className={s.topbar}>
      <span className={s.mobileMenu}>
        <Menu
          align="start"
          trigger={<IconButton label="Navigation" icon={<MenuIcon />} />}
          items={NAV.map((n) => ({ label: n.label, onSelect: () => navigate(n.href) }))}
        />
      </span>
      <AppLink href="/" className={s.brand} aria-label="olmo-eval home">
        <Logo />
        <span>olmo-eval</span>
      </AppLink>
      <nav className={s.nav} aria-label="Main">
        {NAV.map((n) => (
          <AppLink key={n.href} href={n.href} className={cx(s.navItem, n.match(pathname) && s.navActive)}>
            {n.label}
          </AppLink>
        ))}
      </nav>
      <button type="button" className={s.search} onClick={() => openPalette()} aria-label="Search or jump to">
        <Search />
        <span className={s.searchText}>Search runs, models, tasks, IDs…</span>
        <span className={s.searchKbd}>
          <Kbd>{isMac ? "⌘" : "Ctrl"}</Kbd> <Kbd>K</Kbd>
        </span>
      </button>
      <div className={s.right}>
        <BaselineChip />
        {window.__OE_MOCK__ && (
          <Tip content="The API is mocked with synthetic data (npm run dev without a backend).">
            <span className={cx(s.mockBadge, "hide-md")}>mock data</span>
          </Tip>
        )}
        <IconButton
          label={resolvedTheme(theme) === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          icon={resolvedTheme(theme) === "dark" ? <Sun /> : <Moon />}
          onClick={cycleTheme}
          className="hide-sm"
        />
        <Menu
          trigger={
            <button type="button" className={s.avatar} aria-label="Account menu" title={me.data?.email}>
              <UserAvatar seed={me.data?.email ?? me.data?.username} />
            </button>
          }
          items={[
            { heading: me.data?.email ?? "Signed in" },
            { label: "My runs", icon: <User />, onSelect: () => navigate("/runs?user=me") },
            { label: "Saved views", icon: <Bookmark />, onSelect: () => navigate("/views") },
            { separator: true },
            {
              label: density === "compact" ? "Comfortable rows" : "Compact rows",
              icon: <Rows3 />,
              onSelect: () => densityStore.set(density === "compact" ? "comfortable" : "compact"),
            },
            {
              label: `Theme: ${theme}`,
              icon: <Moon />,
              onSelect: () => themeStore.set(theme === "system" ? "light" : theme === "light" ? "dark" : "system"),
            },
            { label: "Keyboard shortcuts", icon: <Keyboard />, hint: "?", onSelect: () => shortcutsStore.set(true) },
          ]}
        />
      </div>
    </header>
  );
}

declare global {
  interface Window {
    __OE_MOCK__?: boolean;
  }
}
