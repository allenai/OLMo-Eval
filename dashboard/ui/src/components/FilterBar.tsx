import type { FacetValue, RunsFacetsResponse } from "@contract/api-types";
import * as Popover from "@radix-ui/react-popover";
import { Filter, Plus } from "lucide-react";
import { forwardRef, type KeyboardEvent, type ReactNode, useCallback, useImperativeHandle, useMemo, useRef, useState } from "react";
import {
  FILTER_LABELS,
  isListKey,
  parseFilterText,
  removeFilterValue,
  RUN_FILTER_KEYS,
  type RunFilterKey,
  type RunFilters,
} from "@/lib/filters";
import { formatCount } from "@/lib/format";
import { joinList, parseList } from "@/lib/url";
import { Button, Checkbox, Chip, cx, Input, SearchInput, uiStyles as ui } from "./primitives";
import f from "./filterbar.module.css";
import { pressable } from "@/lib/pressable";

type FacetKey = "model" | "family" | "user" | "group" | "ws" | "tag" | "status";

const FACET_SOURCE: Record<FacetKey, keyof RunsFacetsResponse> = {
  model: "model",
  family: "family",
  user: "user",
  group: "group",
  ws: "workspace",
  tag: "tag",
  status: "status",
};

export interface FilterBarHandle {
  focus: () => void;
}

interface Props {
  filters: RunFilters;
  onChange: (next: RunFilters) => void;
  facets?: RunsFacetsResponse;
  taskOptions: string[];
  suiteOptions: string[];
  me?: string;
}

/** Value picker with search, facet counts and multi-select. */
function FacetPicker({
  label,
  values,
  selected,
  onApply,
  extra,
  allowCustom,
}: {
  label: string;
  values: FacetValue[];
  selected: string[];
  onApply: (values: string[]) => void;
  extra?: ReactNode;
  allowCustom?: boolean;
}) {
  const [query, setQuery] = useState("");
  const max = Math.max(1, ...values.map((v) => v.count));
  const shown = values.filter((v) => v.value.toLowerCase().includes(query.toLowerCase()));
  const toggle = (v: string) => onApply(selected.includes(v) ? selected.filter((s) => s !== v) : [...selected, v]);
  return (
    <div className={ui.facet}>
      <div className={ui.facetHead}>
        <span className="t-overline">{label}</span>
        <SearchInput
          autoFocus
          placeholder={allowCustom ? "Search, or type a glob like olmo3*" : "Search"}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && query.trim()) {
              const exact = shown.find((v) => v.value.toLowerCase() === query.toLowerCase());
              if (exact) toggle(exact.value);
              else if (allowCustom) toggle(query.trim());
              else if (shown[0]) toggle(shown[0].value);
              setQuery("");
            }
          }}
        />
        {extra}
      </div>
      <div className={ui.facetList} role="listbox" aria-multiselectable>
        {selected
          .filter((s) => !values.some((v) => v.value === s))
          .map((s) => (
            <div key={s} className={ui.facetItem} {...pressable(() => toggle(s), "option")} aria-selected>
              <Checkbox checked onChange={() => toggle(s)} />
              <span className={ui.facetValue}>{s}</span>
              <span className={ui.facetCount}>custom</span>
            </div>
          ))}
        {shown.length === 0 && <div className="t-caption" style={{ padding: 8 }}>No values{allowCustom && query ? " (press Enter to use as a pattern)" : ""}</div>}
        {shown.map((v) => (
          <div key={v.value} className={ui.facetItem} {...pressable(() => toggle(v.value), "option")} aria-selected={selected.includes(v.value)}>
            <Checkbox checked={selected.includes(v.value)} onChange={() => toggle(v.value)} />
            <span className={ui.facetValue} title={v.value}>
              {v.value}
            </span>
            <span className={ui.facetBar}>
              <span style={{ width: `${(v.count / max) * 100}%` }} />
            </span>
            <span className={ui.facetCount}>{formatCount(v.count)}</span>
          </div>
        ))}
      </div>
      <div className={ui.facetFoot}>
        <span className="t-caption">Counts reflect the other active filters.</span>
        {selected.length > 0 && (
          <Button size="sm" variant="ghost" onClick={() => onApply([])}>
            Clear
          </Button>
        )}
      </div>
    </div>
  );
}

function DatePicker({ filters, onChange }: { filters: RunFilters; onChange: (f: RunFilters) => void }) {
  const presets = [
    ["24h", "Last 24 hours"],
    ["7d", "Last 7 days"],
    ["30d", "Last 30 days"],
    ["90d", "Last 90 days"],
  ] as const;
  return (
    <div className={ui.facet}>
      <div className={ui.facetHead}>
        <span className="t-overline">Created</span>
      </div>
      <div className={ui.facetList}>
        {presets.map(([v, l]) => (
          <div
            key={v}
            className={ui.facetItem}
            data-active={filters.date === v}
            onClick={() => onChange({ ...filters, date: filters.date === v ? undefined : v, after: undefined, before: undefined })}
          >
            <span className={ui.facetValue}>{l}</span>
          </div>
        ))}
      </div>
      <div style={{ padding: 8, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, borderTop: "1px solid var(--border)" }}>
        <label className="t-caption">
          After
          <Input type="date" value={filters.after?.slice(0, 10) ?? ""} onChange={(e) => onChange({ ...filters, after: e.target.value || undefined, date: undefined })} />
        </label>
        <label className="t-caption">
          Before
          <Input type="date" value={filters.before?.slice(0, 10) ?? ""} onChange={(e) => onChange({ ...filters, before: e.target.value || undefined, date: undefined })} />
        </label>
      </div>
    </div>
  );
}

function StepPicker({ filters, onChange }: { filters: RunFilters; onChange: (f: RunFilters) => void }) {
  return (
    <div className={ui.facet}>
      <div className={ui.facetHead}>
        <span className="t-overline">Checkpoint step</span>
      </div>
      <div style={{ padding: 8, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <label className="t-caption">
          At least
          <Input inputMode="numeric" value={filters.smin ?? ""} onChange={(e) => onChange({ ...filters, smin: e.target.value.replace(/\D/g, "") || undefined })} />
        </label>
        <label className="t-caption">
          At most
          <Input inputMode="numeric" value={filters.smax ?? ""} onChange={(e) => onChange({ ...filters, smax: e.target.value.replace(/\D/g, "") || undefined })} />
        </label>
      </div>
    </div>
  );
}

function TextPicker({ label, value, onChange, placeholder }: { label: string; value?: string; onChange: (v?: string) => void; placeholder: string }) {
  const [draft, setDraft] = useState(value ?? "");
  return (
    <div className={ui.facet}>
      <div className={ui.facetHead}>
        <span className="t-overline">{label}</span>
        <Input
          autoFocus
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onChange(draft.trim() || undefined)}
        />
        <div className="row">
          <span className="spacer" />
          <Button size="sm" variant="primary" onClick={() => onChange(draft.trim() || undefined)}>
            Apply
          </Button>
        </div>
      </div>
    </div>
  );
}

const ADDABLE: { key: RunFilterKey; label: string }[] = [
  { key: "model", label: "Model" },
  { key: "family", label: "Family" },
  { key: "task", label: "Task" },
  { key: "suite", label: "Suite" },
  { key: "user", label: "User" },
  { key: "group", label: "Group" },
  { key: "ws", label: "Workspace" },
  { key: "tag", label: "Tag" },
  { key: "status", label: "Status" },
  { key: "date", label: "Date" },
  { key: "commit", label: "Commit" },
  { key: "smin", label: "Step" },
];

export const FilterBar = forwardRef<FilterBarHandle, Props>(function FilterBar(
  { filters, onChange, facets, taskOptions, suiteOptions },
  ref,
) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [text, setText] = useState("");
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [activeSuggestion, setActiveSuggestion] = useState(0);
  useImperativeHandle(ref, () => ({ focus: () => inputRef.current?.focus() }));

  const facetValues = useCallback(
    (key: RunFilterKey): FacetValue[] => {
      if (key === "task") return taskOptions.map((t) => ({ value: t, count: 0 }));
      if (key === "suite") return suiteOptions.map((t) => ({ value: t, count: 0 }));
      if (key in FACET_SOURCE && facets) return facets[FACET_SOURCE[key as FacetKey]];
      return [];
    },
    [taskOptions, suiteOptions, facets],
  );

  // Autocomplete for the token being typed.
  const lastToken = text.split(/\s+/).pop() ?? "";
  const suggestions = useMemo(() => {
    const m = /^(-?)([a-z]+):(.*)$/i.exec(lastToken);
    if (m) {
      const key = (m[2].toLowerCase() === "workspace" ? "ws" : m[2].toLowerCase()) as RunFilterKey;
      if (m[2].toLowerCase() === "has") return ["has:failures"];
      const values = facetValues(key)
        .map((v) => v.value)
        .filter((v) => v.toLowerCase().includes(m[3].toLowerCase()))
        .slice(0, 8);
      return values.map((v) => `${m[1]}${m[2]}:${v}`);
    }
    if (!lastToken) return [];
    const keys = ["model:", "family:", "task:", "suite:", "user:", "group:", "ws:", "tag:", "-tag:", "status:", "after:", "before:", "date:", "commit:", "step:>=", "has:failures"];
    return keys.filter((k) => k.startsWith(lastToken.toLowerCase())).slice(0, 6);
  }, [lastToken, facetValues]);

  const commit = (value: string) => {
    if (!value.trim()) return;
    const parsed = parseFilterText(value);
    const next: RunFilters = { ...filters };
    for (const key of RUN_FILTER_KEYS) {
      const v = parsed[key];
      if (!v) continue;
      if (isListKey(key)) {
        const merged = Array.from(new Set([...parseList(next[key]), ...parseList(v)]));
        next[key] = joinList(merged);
      } else if (key === "q") next.q = [next.q, v].filter(Boolean).join(" ");
      else next[key] = v;
    }
    onChange(next);
    setText("");
  };

  const applySuggestion = (s: string) => {
    const parts = text.split(/\s+/);
    parts[parts.length - 1] = s;
    const joined = parts.join(" ");
    if (s.endsWith(":") || s.endsWith(">=")) setText(joined);
    else commit(joined);
    inputRef.current?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (suggestions.length && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      setActiveSuggestion((i) => (i + (e.key === "ArrowDown" ? 1 : -1) + suggestions.length) % suggestions.length);
      return;
    }
    if ((e.key === "Tab" || e.key === "Enter") && suggestions.length && lastToken && suggestions[activeSuggestion] !== lastToken) {
      e.preventDefault();
      applySuggestion(suggestions[activeSuggestion]);
      return;
    }
    if (e.key === "Enter") {
      e.preventDefault();
      commit(text);
    }
    if (e.key === "Escape") {
      setText("");
      inputRef.current?.blur();
    }
    if (e.key === "Backspace" && !text) {
      const keys = RUN_FILTER_KEYS.filter((k) => filters[k] && k !== "sc");
      const last = keys[keys.length - 1];
      if (last) onChange(removeFilterValue(filters, last));
    }
  };

  const chips: { key: RunFilterKey; value?: string; display: string }[] = [];
  for (const key of RUN_FILTER_KEYS) {
    const v = filters[key];
    if (!v || key === "sc") continue;
    if (key === "smax" && filters.smin) continue;
    if (key === "smin") {
      chips.push({ key, display: filters.smax ? (filters.smax === v ? v : `${v}–${filters.smax}`) : `≥ ${v}` });
      continue;
    }
    if (key === "fail") {
      chips.push({ key, display: "yes" });
      continue;
    }
    if (isListKey(key)) {
      const values = parseList(v);
      chips.push({ key, value: undefined, display: values.join(", ") });
    } else chips.push({ key, display: v });
  }

  const picker = (key: RunFilterKey) => {
    const label = FILTER_LABELS[key];
    if (key === "date") return <DatePicker filters={filters} onChange={onChange} />;
    if (key === "smin" || key === "smax") return <StepPicker filters={filters} onChange={onChange} />;
    if (key === "commit") return <TextPicker label="olmo-eval commit" value={filters.commit} placeholder="SHA prefix, e.g. 3f2a1b" onChange={(v) => (onChange({ ...filters, commit: v }), setOpenKey(null))} />;
    if (key === "q") return <TextPicker label="Text" value={filters.q} placeholder="Matches model, run name, group, tags" onChange={(v) => (onChange({ ...filters, q: v }), setOpenKey(null))} />;
    const selected = parseList(filters[key]);
    return (
      <FacetPicker
        label={label}
        values={facetValues(key)}
        selected={selected}
        allowCustom={key === "model" || key === "task"}
        onApply={(values) => onChange({ ...filters, [key]: joinList(values) })}
        extra={
          key === "suite" ? (
            <Checkbox
              checked={filters.sc === "any"}
              onChange={(on) => onChange({ ...filters, sc: on ? "any" : undefined })}
              label="Match runs with any task of the suite"
            />
          ) : key === "tag" ? (
            <span className="t-caption">Exclude a tag by typing -tag:name in the filter box.</span>
          ) : undefined
        }
      />
    );
  };

  const popoverFor = (key: RunFilterKey, trigger: ReactNode) => (
    <Popover.Root key={key} open={openKey === key} onOpenChange={(o) => setOpenKey(o ? key : null)}>
      <Popover.Trigger asChild>{trigger}</Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={ui.popover} align="start" sideOffset={4} collisionPadding={8}>
          {picker(key)}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );

  const activeKeys = new Set(chips.map((c) => c.key));
  return (
    <div className={f.bar}>
      <div className={f.inputRow}>
        <div className={f.inputWrap}>
          <Filter className={f.icon} />
          <input
            ref={inputRef}
            className={cx(ui.input, f.input)}
            placeholder="Filter: model:olmo3* user:me after:2026-09-01 status:failed  (press / to focus)"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setActiveSuggestion(0);
            }}
            onKeyDown={onKeyDown}
            aria-label="Filter runs"
            spellCheck={false}
          />
          {suggestions.length > 0 && text && (
            <div className={cx(ui.popover, f.suggest)} role="listbox">
              {suggestions.map((s, i) => (
                <div
                  key={s}
                  role="option"
                  aria-selected={i === activeSuggestion}
                  className={cx(ui.menuItem, f.suggestItem)}
                  data-highlighted={i === activeSuggestion ? "" : undefined}
                  onMouseDown={(e) => {
                    e.preventDefault();
                    applySuggestion(s);
                  }}
                >
                  <span className="mono">{s}</span>
                </div>
              ))}
              <div className={f.suggestHint}>Tab to complete · Enter to apply</div>
            </div>
          )}
        </div>
        {chips.length > 0 && (
          <Button variant="ghost" size="sm" onClick={() => onChange({})}>
            Clear all
          </Button>
        )}
      </div>
      <div className={f.chips}>
        {chips.map((c) =>
          popoverFor(
            c.key,
            <span>
              <Chip
                label={FILTER_LABELS[c.key]}
                value={c.display}
                active
                onClick={() => setOpenKey(c.key)}
                onRemove={() => onChange(removeFilterValue(filters, c.key === "smin" ? "smin" : c.key, c.value))}
                title={`${FILTER_LABELS[c.key]}: ${c.display}`}
              />
            </span>,
          ),
        )}
        {ADDABLE.filter((a) => !activeKeys.has(a.key)).map((a) =>
          popoverFor(
            a.key,
            <button type="button" className={ui.addChip}>
              <Plus /> {a.label}
            </button>,
          ),
        )}
        {!filters.fail && (
          <button type="button" className={ui.addChip} onClick={() => onChange({ ...filters, fail: "1" })}>
            <Plus /> Has failures
          </button>
        )}
        {filters.q && !chips.some((c) => c.key === "q") && (
          <Chip label="Text" value={filters.q} onRemove={() => onChange({ ...filters, q: undefined })} />
        )}
      </div>
    </div>
  );
});
