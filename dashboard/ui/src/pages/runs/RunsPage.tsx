import type { RunsListResponse, SavedView } from "@contract/api-types";
import * as Dialog from "@radix-ui/react-dialog";
import * as Popover from "@radix-ui/react-popover";
import { ArrowRight, Bookmark, BookmarkPlus, ChevronDown, Copy, Plus, Settings2, Star, Tag, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSavedViews, useSuites, useTasks, useViewMutations } from "@/api/hooks/catalog";
import { useBulkTag, useRunsFacets } from "@/api/hooks/runs";
import { buildHref, useAppNavigate } from "@/components/AppLink";
import { FilterBar, type FilterBarHandle } from "@/components/FilterBar";
import { PageHeader } from "@/components/PageHeader";
import { Button, Checkbox, cx, IconButton, Input, Menu, Panel, SearchInput, Segmented, uiStyles as ui } from "@/components/primitives";
import { pageActions } from "@/components/shell/registry";
import { toast } from "@/components/toast";
import { copyText } from "@/lib/csv";
import { filtersToApi, RUN_FILTER_KEYS, type RunFilters } from "@/lib/filters";
import { formatCount, pluralize } from "@/lib/format";
import { oneOf, parseList } from "@/lib/url";
import { useBaseline, useSearchParams } from "@/state/nav";
import { trayStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import { type GroupByKey, RunsTable } from "./RunsTable";
import { pressable } from "@/lib/pressable";

type RunsSearch = RunFilters & { sort?: string; cols?: string; gb?: string; baseline?: string };

const BUILTIN_VIEWS: { name: string; query: string }[] = [
  { name: "My runs", query: "user=me" },
  { name: "Failed in the last 7 days", query: "status=failed,partial&date=7d" },
  { name: "Running now", query: "status=running" },
];

function ScoreColumnPicker({ cols, onChange }: { cols: string[]; onChange: (cols: string[]) => void }) {
  const tasks = useTasks({});
  const suites = useSuites();
  const [query, setQuery] = useState("");
  const options = [
    ...(suites.data?.items ?? []).map((s) => ({ key: `suite:${s.suite_name}`, label: s.suite_name, kind: "suite" })),
    ...(tasks.data?.items ?? []).map((t) => ({ key: `task:${t.task_name}`, label: t.task_name, kind: "task" })),
  ].filter((o) => o.label.toLowerCase().includes(query.toLowerCase().replace(/^(task|suite):/, "")));
  const toggle = (key: string) => onChange(cols.includes(key) ? cols.filter((c) => c !== key) : [...cols, key].slice(0, 20));
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button size="sm" icon={<Plus />}>
          Score columns{cols.length ? ` (${cols.length})` : ""}
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className={ui.popover} align="end" sideOffset={4} collisionPadding={8}>
          <div className={ui.facet} style={{ width: 320 }}>
            <div className={ui.facetHead}>
              <span className="t-overline">Add a score column</span>
              <SearchInput autoFocus placeholder="Find a task or suite" value={query} onChange={(e) => setQuery(e.target.value)} />
              <span className="t-caption">Each column shows the primary score with stderr, plus a delta when a baseline is set.</span>
            </div>
            <div className={ui.facetList}>
              {options.map((o) => (
                <div key={o.key} className={ui.facetItem} {...pressable(() => toggle(o.key), "option")} aria-selected={cols.includes(o.key)}>
                  <Checkbox checked={cols.includes(o.key)} onChange={() => toggle(o.key)} />
                  <span className={ui.facetValue}>{o.label}</span>
                  <span className={ui.facetCount}>{o.kind}</span>
                </div>
              ))}
            </div>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function SaveViewDialog({ open, onOpenChange, query }: { open: boolean; onOpenChange: (o: boolean) => void; query: string }) {
  const [name, setName] = useState("");
  const [shared, setShared] = useState(false);
  const { create } = useViewMutations();
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={ui.overlay} />
        <Dialog.Content className={ui.dialog} aria-describedby={undefined}>
          <Dialog.Title className={ui.dialogTitle}>Save view</Dialog.Title>
          <p className="muted" style={{ fontSize: 13 }}>
            Saves the current filters, columns and sort. Views store only the URL query string.
          </p>
          <form
            style={{ display: "contents" }}
            onSubmit={async (e) => {
              e.preventDefault();
              if (!name.trim() || create.isPending) return;
              // Failures are reported by the global mutation error toast.
              const ok = await create.mutateAsync({ name: name.trim(), page: "runs", query, shared }).then(() => true, () => false);
              if (!ok) return;
              toast(`Saved view “${name.trim()}”`);
              setName("");
              onOpenChange(false);
            }}
          >
            <div className={ui.dialogBody}>
              <Input autoFocus placeholder="Name, e.g. 7B midtrain OLMES" value={name} onChange={(e) => setName(e.target.value)} />
              <Checkbox checked={shared} onChange={setShared} label="Share with everyone at Ai2" />
              <code className="t-caption" style={{ overflowWrap: "anywhere" }}>
                ?{query || "(no filters)"}
              </code>
            </div>
            <div className={ui.dialogActions}>
              <Dialog.Close asChild>
                <Button variant="ghost">Cancel</Button>
              </Dialog.Close>
              <Button type="submit" variant="primary" disabled={!name.trim() || create.isPending}>
                Save view
              </Button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function TagDialog({ open, onOpenChange, runIds }: { open: boolean; onOpenChange: (o: boolean) => void; runIds: string[] }) {
  const [tag, setTag] = useState("");
  const bulk = useBulkTag();
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={ui.overlay} />
        <Dialog.Content className={ui.dialog} aria-describedby={undefined}>
          <Dialog.Title className={ui.dialogTitle}>Add tag to {pluralize(runIds.length, "run")}</Dialog.Title>
          <form
            style={{ display: "contents" }}
            onSubmit={async (e) => {
              e.preventDefault();
              if (!tag || bulk.isPending) return;
              const res = await bulk.mutateAsync({ run_ids: runIds, add: [tag], remove: [] }).catch(() => null);
              if (!res) return;
              toast(`Tagged ${pluralize(res.updated, "run")} with “${tag}”`);
              setTag("");
              onOpenChange(false);
            }}
          >
            <div className={ui.dialogBody}>
              <Input autoFocus placeholder="tag, e.g. candidate" value={tag} onChange={(e) => setTag(e.target.value.replace(/\s+/g, "-"))} />
            </div>
            <div className={ui.dialogActions}>
              <Dialog.Close asChild>
                <Button variant="ghost">Cancel</Button>
              </Dialog.Close>
              <Button type="submit" variant="primary" disabled={!tag || bulk.isPending}>
                Add tag
              </Button>
            </div>
          </form>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function SavedViewsMenu({ onApply, onManage }: { onApply: (query: string) => void; onManage: () => void }) {
  const views = useSavedViews("runs");
  const item = (v: SavedView | { name: string; query: string }, own?: boolean) => ({
    label: (
      <span className="row" style={{ gap: 8, width: "100%" }}>
        <span className="truncate">{v.name}</span>
        {"owner_email" in v && !own && <span className="t-caption">{v.owner_email.split("@")[0]}</span>}
      </span>
    ),
    icon: <Bookmark />,
    onSelect: () => onApply(v.query),
  });
  return (
    <Menu
      align="end"
      trigger={
        <Button size="sm" icon={<Bookmark />}>
          Saved views <ChevronDown />
        </Button>
      }
      items={[
        { heading: "Built in" },
        ...BUILTIN_VIEWS.map((v) => item(v)),
        { separator: true },
        { heading: "My views" },
        ...(views.data?.mine.length ? views.data.mine.map((v) => item(v, true)) : [{ label: <span className="muted">None yet</span>, disabled: true }]),
        { separator: true },
        { heading: "Shared views" },
        ...(views.data?.shared.length ? views.data.shared.map((v) => item(v)) : [{ label: <span className="muted">None</span>, disabled: true }]),
        { separator: true },
        { label: "Manage saved views…", icon: <Settings2 />, onSelect: onManage },
      ]}
    />
  );
}

export function RunsPage() {
  const [search, setSearch] = useSearchParams<RunsSearch>();
  const navigate = useAppNavigate();
  const [baseline, setBaseline] = useBaseline();
  const filterRef = useRef<FilterBarHandle>(null);
  const [saveOpen, setSaveOpen] = useState(false);
  const [tagOpen, setTagOpen] = useState(false);
  const [total, setTotal] = useState<number | null>(null);
  const tray = useStore(trayStore);
  const filters: RunFilters = useMemo(() => {
    const out: RunFilters = {};
    for (const k of RUN_FILTER_KEYS) if (search[k]) out[k] = search[k];
    return out;
  }, [search]);
  const cols = parseList(search.cols);
  const groupBy = oneOf<GroupByKey>(search.gb, ["none", "group", "launch", "model"], "none");
  const apiFilters = useMemo(() => filtersToApi(filters), [filters]);
  const facets = useRunsFacets(apiFilters);
  const tasks = useTasks({});
  const suites = useSuites();
  const params = useMemo(
    () => ({ ...apiFilters, sort: search.sort, cols: search.cols, baseline, limit: 100 }),
    [apiFilters, search.sort, search.cols, baseline],
  );

  useEffect(() => {
    pageActions.focusFilter = () => filterRef.current?.focus();
    pageActions.saveView = () => setSaveOpen(true);
    return () => {
      pageActions.focusFilter = undefined;
      pageActions.saveView = undefined;
    };
  }, []);

  const setFilters = (next: RunFilters) => {
    const patch: Partial<RunsSearch> = {};
    for (const k of RUN_FILTER_KEYS) patch[k] = next[k];
    setSearch(patch);
  };

  const currentQuery = useMemo(() => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(search)) if (v && k !== "baseline") q.set(k, v);
    return q.toString().replace(/%2C/g, ",").replace(/%3A/g, ":").replace(/%40/g, "@");
  }, [search]);

  const selectedRuns = tray.filter((t) => t.key.startsWith("r:"));
  const onResponse = useCallback((d: RunsListResponse | undefined) => setTotal(d?.total ?? null), []);

  return (
    <div className="page">
      <PageHeader
        title="Runs"
        titleExtra={total != null && <span className="muted mono" style={{ fontSize: 14, marginTop: 4 }}>{formatCount(total)}</span>}
        actions={
          <>
            <SavedViewsMenu onApply={(q) => navigate(`/runs?${q}`)} onManage={() => navigate("/views")} />
            <Button size="sm" icon={<BookmarkPlus />} onClick={() => setSaveOpen(true)}>
              Save view
            </Button>
            <ScoreColumnPicker cols={cols} onChange={(next) => setSearch({ cols: next.join(",") || undefined })} />
          </>
        }
      />
      <FilterBar
        ref={filterRef}
        filters={filters}
        onChange={setFilters}
        facets={facets.data}
        taskOptions={(tasks.data?.items ?? []).map((t) => t.task_name)}
        suiteOptions={(suites.data?.items ?? []).map((s) => s.suite_name)}
      />
      {selectedRuns.length > 0 && (
        <div className="row-wrap" style={{ padding: "6px 10px", borderRadius: 6, background: "var(--accent-soft)", fontSize: 13 }}>
          <strong>{selectedRuns.length} selected</strong>
          <span className="muted">in the compare tray</span>
          <span className="spacer" />
          <Button size="sm" variant="primary" onClick={() => navigate(buildHref("/compare", { subjects: selectedRuns.map((t) => t.key).join(",") }))}>
            Compare <ArrowRight />
          </Button>
          <Button size="sm" icon={<Star />} disabled={selectedRuns.length !== 1} onClick={() => setBaseline(selectedRuns[0].key, selectedRuns[0].label)}>
            Set baseline
          </Button>
          <Button size="sm" icon={<Tag />} onClick={() => setTagOpen(true)}>
            Add tag
          </Button>
          <Button
            size="sm"
            icon={<Copy />}
            onClick={() => copyText(selectedRuns.map((t) => t.key.slice(2)).join("\n")).then((ok) => ok && toast("Copied run IDs"))}
          >
            Copy run IDs
          </Button>
          <IconButton size="sm" label="Clear selection" icon={<X />} onClick={() => trayStore.set((prev) => prev.filter((t) => !t.key.startsWith("r:")))} />
        </div>
      )}
      <Panel pad="none">
        <RunsTable
          tableId="runs"
          params={params}
          sort={search.sort}
          onSortChange={(sort) => setSearch({ sort }, { replace: true })}
          groupBy={groupBy}
          onResponse={onResponse}
          maxHeight="calc(100vh - 290px)"
          toolbar={
            <div className="row" style={{ gap: 8 }}>
              <span className="t-caption">Group by</span>
              <Segmented
                value={groupBy}
                onChange={(v) => setSearch({ gb: v === "none" ? undefined : v })}
                options={[
                  { value: "none", label: "None" },
                  { value: "group", label: "Group" },
                  { value: "launch", label: "Launch" },
                  { value: "model", label: "Model" },
                ]}
                label="Group by"
              />
              {cols.length > 0 && (
                <span className={cx("row-wrap", "hide-sm")} style={{ gap: 4 }}>
                  {cols.map((c) => (
                    <span key={c} className={ui.chip} style={{ height: 22 }}>
                      <span className="mono" style={{ fontSize: 11 }}>
                        {c.replace(/^(task|suite):/, "")}
                      </span>
                      <button type="button" className={ui.chipRemove} aria-label={`Remove column ${c}`} onClick={() => setSearch({ cols: cols.filter((x) => x !== c).join(",") || undefined })}>
                        <X />
                      </button>
                    </span>
                  ))}
                </span>
              )}
            </div>
          }
        />
      </Panel>
      <SaveViewDialog open={saveOpen} onOpenChange={setSaveOpen} query={currentQuery} />
      <TagDialog open={tagOpen} onOpenChange={setTagOpen} runIds={selectedRuns.map((t) => t.key.slice(2))} />
    </div>
  );
}
