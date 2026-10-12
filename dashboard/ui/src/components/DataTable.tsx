import * as Popover from "@radix-ui/react-popover";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDown, ArrowUp, ChevronDown, Columns3, Download, MoreHorizontal } from "lucide-react";
import {
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { type Cell, copyText, downloadText, slugify, toCSV, toTSV } from "@/lib/csv";
import { formatCount } from "@/lib/format";
import { useHotkeys } from "@/lib/keyboard";
import { columnWidthStore, densityStore } from "@/state/prefs";
import { useStore } from "@/state/store";
import s from "./DataTable.module.css";
import { Checkbox, cx, EmptyState, IconButton, Menu, SearchInput, SkeletonRows, uiStyles } from "./primitives";
import { toast } from "./toast";

/** Width of the trailing row-actions column; fits three small icon buttons. */
const ROW_ACTIONS_WIDTH = 96;

export interface Column<T> {
  id: string;
  header: ReactNode;
  /** Plain-text header for CSV and the column chooser. */
  title?: string;
  cell: (row: T, index: number) => ReactNode;
  width?: number;
  minWidth?: number;
  align?: "left" | "right" | "center";
  /** Server or client sort key; enables sorting on this column. */
  sortKey?: string;
  /** Client-side sort value (when the table sorts locally). */
  sortValue?: (row: T) => string | number | null | undefined;
  pin?: boolean;
  /** Share leftover width with other growing columns; shrinks to minWidth before the table scrolls. */
  grow?: boolean;
  hideable?: boolean;
  defaultHidden?: boolean;
  csv?: (row: T) => Cell;
  headerTitle?: string;
}

export interface SortSpec {
  key: string;
  desc: boolean;
}

export function parseSortList(value: string | undefined): SortSpec[] {
  if (!value) return [];
  return value
    .split(",")
    .filter(Boolean)
    .map((v) => (v.startsWith("-") ? { key: v.slice(1), desc: true } : { key: v, desc: false }));
}

export function sortListParam(list: SortSpec[]): string | undefined {
  return list.length ? list.map((x) => (x.desc ? `-${x.key}` : x.key)).join(",") : undefined;
}

/** Click cycles desc → asc → none; shift-click adds a secondary sort. */
export function nextSort(current: SortSpec[], key: string, additive: boolean): SortSpec[] {
  const existing = current.find((x) => x.key === key);
  const cycled: SortSpec | null = !existing ? { key, desc: true } : existing.desc ? { key, desc: false } : null;
  if (additive) {
    const rest = current.filter((x) => x.key !== key);
    return cycled ? [...rest, cycled] : rest;
  }
  return cycled ? [cycled] : [];
}

export function sortRows<T>(rows: T[], sort: SortSpec[], columns: Column<T>[]): T[] {
  if (!sort.length) return rows;
  const getters = sort
    .map((spec) => {
      const col = columns.find((c) => (c.sortKey ?? c.id) === spec.key);
      return col?.sortValue ? { get: col.sortValue, desc: spec.desc } : null;
    })
    .filter((g): g is { get: (row: T) => string | number | null | undefined; desc: boolean } => !!g);
  if (!getters.length) return rows;
  return [...rows].sort((a, b) => {
    for (const g of getters) {
      const va = g.get(a);
      const vb = g.get(b);
      if (va == null && vb == null) continue;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (va === vb) continue;
      const cmp = va < vb ? -1 : 1;
      return g.desc ? -cmp : cmp;
    }
    return 0;
  });
}

/*
 * Keyboard arbitration: several tables can share a page, but only one answers j/k/x/Enter.
 * The active table is the one the user last clicked or focused, else the first one mounted.
 */
const keyboardTables: string[] = [];
let activeKeyboardTable: string | null = null;

function isKeyboardTable(id: string): boolean {
  if (activeKeyboardTable && keyboardTables.includes(activeKeyboardTable)) return activeKeyboardTable === id;
  return keyboardTables[0] === id;
}

/** Elements that handle Enter themselves; table shortcuts must not also fire. */
function isInteractive(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && !!target.closest('a,button,summary,[role="button"],[role="tab"],[role="menuitem"],[role="option"],[role="checkbox"]');
}

type Item<T> = { kind: "group"; key: string; rows: T[] } | { kind: "row"; row: T; index: number };

export interface DataTableProps<T> {
  tableId: string;
  columns: Column<T>[];
  rows: T[];
  getRowId: (row: T) => string;
  loading?: boolean;
  fetching?: boolean;
  sort?: string;
  onSortChange?: (sort: string | undefined) => void;
  /** Sort rows locally using Column.sortValue (default when no onSortChange). */
  clientSort?: boolean;
  selected?: Set<string>;
  onSelectionChange?: (next: Set<string>) => void;
  onRowOpen?: (row: T, event?: MouseEvent) => void;
  rowHref?: (row: T) => string | undefined;
  rowActions?: (row: T) => ReactNode;
  rowClassName?: (row: T) => string | undefined;
  groupBy?: (row: T) => string | null;
  renderGroup?: (key: string, rows: T[]) => ReactNode;
  total?: number;
  hasMore?: boolean;
  onLoadMore?: () => void;
  fetchAll?: () => Promise<T[]>;
  empty?: ReactNode;
  maxHeight?: number | string;
  keyboard?: boolean;
  focusedId?: string | null;
  onFocusChange?: (row: T | null) => void;
  toolbar?: ReactNode;
  footerExtra?: ReactNode;
  exportName?: string;
  hideFooter?: boolean;
  rowHeight?: number;
}

export function DataTable<T>(props: DataTableProps<T>) {
  const {
    tableId,
    columns,
    rows,
    getRowId,
    loading,
    fetching,
    sort,
    onSortChange,
    selected,
    onSelectionChange,
    onRowOpen,
    rowActions,
    rowClassName,
    groupBy,
    renderGroup,
    total,
    hasMore,
    onLoadMore,
    fetchAll,
    empty,
    maxHeight = "calc(100vh - 260px)",
    keyboard = true,
    toolbar,
    footerExtra,
    exportName,
    hideFooter,
  } = props;
  const density = useStore(densityStore);
  const rowH = props.rowHeight ?? (density === "comfortable" ? 36 : 30);
  const savedWidths = useStore(columnWidthStore)[tableId];
  const [hidden, setHidden] = useState<Set<string>>(
    () => new Set(columns.filter((c) => c.defaultHidden).map((c) => c.id)),
  );
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [localSort, setLocalSort] = useState<SortSpec[]>([]);
  const [focusIdx, setFocusIdx] = useState<number>(-1);
  const lastClicked = useRef<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const instanceId = useId();

  useEffect(() => {
    if (!keyboard) return;
    keyboardTables.push(instanceId);
    return () => {
      const i = keyboardTables.indexOf(instanceId);
      if (i >= 0) keyboardTables.splice(i, 1);
      if (activeKeyboardTable === instanceId) activeKeyboardTable = null;
    };
  }, [keyboard, instanceId]);

  const sortedOnServer = !!onSortChange;
  const sortList = useMemo(() => (sortedOnServer ? parseSortList(sort) : localSort), [sortedOnServer, sort, localSort]);
  const visibleColumns = useMemo(() => columns.filter((c) => !hidden.has(c.id)), [columns, hidden]);
  const selectable = !!onSelectionChange;

  const widths = useMemo(
    () => visibleColumns.map((c) => savedWidths?.[c.id] ?? c.width ?? 140),
    [visibleColumns, savedWidths],
  );
  // A growing column (not pinned, not resized by the user) takes leftover width in proportion to
  // its default width and shrinks to its minimum before the table scrolls sideways.
  const growMins = visibleColumns.map((c, i) =>
    c.grow && !c.pin && savedWidths?.[c.id] == null ? (c.minWidth ?? Math.round(widths[i] * 0.6)) : null,
  );
  const selectWidth = selectable ? 34 : 0;
  // Row actions get their own trailing column, so showing them on hover never covers or
  // displaces a cell.
  const actionsWidth = rowActions ? ROW_ACTIONS_WIDTH : 0;
  const template = `${selectable ? `${selectWidth}px ` : ""}${widths
    .map((w, i) => (growMins[i] != null ? `minmax(${growMins[i]}px, ${w}fr)` : `${w}px`))
    .join(" ")}${actionsWidth ? ` ${actionsWidth}px` : ""}`;
  const totalWidth =
    selectWidth + actionsWidth + widths.reduce((a, w, i) => a + (growMins[i] ?? w), 0);

  // Offsets for pinned columns (pinned columns must come first).
  const pinOffsets = useMemo(() => {
    const out: (number | null)[] = [];
    let left = selectWidth;
    visibleColumns.forEach((c, i) => {
      if (c.pin) {
        out.push(left);
        left += widths[i];
      } else out.push(null);
    });
    return out;
  }, [visibleColumns, widths, selectWidth]);
  const lastPinned = pinOffsets.reduce<number>((acc, v, i) => (v != null ? i : acc), -1);

  const sortedRows = useMemo(
    () => (props.clientSort || !sortedOnServer ? sortRows(rows, sortList, columns) : rows),
    [rows, sortList, columns, props.clientSort, sortedOnServer],
  );

  const items: Item<T>[] = useMemo(() => {
    if (!groupBy) return sortedRows.map((row, index) => ({ kind: "row", row, index }));
    const groups = new Map<string, T[]>();
    for (const row of sortedRows) {
      const key = groupBy(row) ?? "—";
      const list = groups.get(key);
      if (list) list.push(row);
      else groups.set(key, [row]);
    }
    const out: Item<T>[] = [];
    let index = 0;
    for (const [key, list] of groups) {
      out.push({ kind: "group", key, rows: list });
      if (!collapsed.has(key)) for (const row of list) out.push({ kind: "row", row, index: index++ });
      else index += list.length;
    }
    return out;
  }, [sortedRows, groupBy, collapsed]);

  const rowItems = useMemo(() => items.filter((i): i is Extract<Item<T>, { kind: "row" }> => i.kind === "row"), [items]);

  // TanStack Virtual returns non-memoizable functions; the React Compiler skips this component.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => rowH,
    overscan: 14,
    // Used until the scroller is measured (and in environments without layout).
    initialRect: { width: 1200, height: typeof maxHeight === "number" ? maxHeight : 720 },
  });

  const virtualItems = virtualizer.getVirtualItems();
  const lastVirtual = virtualItems[virtualItems.length - 1];
  useEffect(() => {
    if (!lastVirtual || !hasMore || fetching || !onLoadMore) return;
    if (lastVirtual.index >= items.length - 15) onLoadMore();
  }, [lastVirtual, hasMore, fetching, onLoadMore, items.length]);

  // Controlled focus (e.g. instance list) syncs into the local index.
  useEffect(() => {
    if (props.focusedId === undefined) return;
    const idx = props.focusedId == null ? -1 : rowItems.findIndex((r) => getRowId(r.row) === props.focusedId);
    setFocusIdx(idx);
  }, [props.focusedId, rowItems, getRowId]);

  const setSort = (key: string, additive: boolean) => {
    const next = nextSort(sortList, key, additive);
    if (onSortChange) onSortChange(sortListParam(next));
    else setLocalSort(next);
  };

  const toggleRow = useCallback(
    (row: T, idx: number, shift: boolean) => {
      if (!onSelectionChange) return;
      const next = new Set(selected ?? []);
      const id = getRowId(row);
      if (shift && lastClicked.current != null) {
        const [a, b] = [lastClicked.current, idx].sort((x, y) => x - y);
        const turnOn = !next.has(id);
        for (let i = a; i <= b; i++) {
          const rid = getRowId(rowItems[i]?.row);
          if (!rid) continue;
          if (turnOn) next.add(rid);
          else next.delete(rid);
        }
      } else if (next.has(id)) next.delete(id);
      else next.add(id);
      lastClicked.current = idx;
      onSelectionChange(next);
    },
    [onSelectionChange, selected, getRowId, rowItems],
  );

  const moveFocus = (delta: number) => {
    if (!rowItems.length) return;
    const next = Math.max(0, Math.min(rowItems.length - 1, (focusIdx < 0 ? -1 : focusIdx) + delta));
    setFocusIdx(next);
    props.onFocusChange?.(rowItems[next].row);
    const itemIndex = items.indexOf(rowItems[next]);
    virtualizer.scrollToIndex(itemIndex, { align: "auto" });
  };

  const openFocused = () => focusIdx >= 0 && rowItems[focusIdx] && onRowOpen?.(rowItems[focusIdx].row);
  const toggleFocused = (shift: boolean) => focusIdx >= 0 && rowItems[focusIdx] && toggleRow(rowItems[focusIdx].row, focusIdx, shift);
  const mine = (fn: (event: KeyboardEvent) => void) => (event: KeyboardEvent) => {
    if (isKeyboardTable(instanceId)) fn(event);
  };
  useHotkeys(
    {
      j: mine(() => moveFocus(1)),
      k: mine(() => moveFocus(-1)),
      x: mine(() => toggleFocused(false)),
      "Shift+X": mine(() => toggleFocused(true)),
      Enter: mine((event) => {
        if (!isInteractive(event.target)) openFocused();
      }),
      o: mine(() => openFocused()),
      e: mine(() => {
        if (!groupBy) return;
        const keys = items.filter((i) => i.kind === "group").map((i) => (i as { key: string }).key);
        setCollapsed((prev) => (prev.size ? new Set() : new Set(keys)));
      }),
    },
    { enabled: keyboard },
  );

  const onScrollerKey = (event: React.KeyboardEvent) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === " " && selectable) {
      event.preventDefault();
      toggleFocused(event.shiftKey);
    }
  };

  const startResize = (e: React.PointerEvent, colId: string, startWidth: number, minWidth: number) => {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const onMove = (ev: PointerEvent) => {
      const w = Math.max(minWidth, Math.round(startWidth + ev.clientX - startX));
      columnWidthStore.set((prev) => ({ ...prev, [tableId]: { ...(prev[tableId] ?? {}), [colId]: w } }));
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const resetWidth = (colId: string) => {
    columnWidthStore.set((prev) => {
      const next = { ...(prev[tableId] ?? {}) };
      delete next[colId];
      return { ...prev, [tableId]: next };
    });
  };

  const exportRows = async (format: "csv" | "tsv") => {
    let data = sortedRows;
    if (fetchAll && hasMore) {
      toast("Fetching all rows for export…", { tone: "info" });
      try {
        data = await fetchAll();
      } catch {
        toast("Export failed while fetching rows", { tone: "error" });
        return;
      }
    }
    const cols = visibleColumns.filter((c) => c.csv);
    const header = cols.map((c) => c.title ?? (typeof c.header === "string" ? c.header : c.id));
    const body = data.slice(0, 100_000).map((row) => cols.map((c) => c.csv!(row)));
    if (format === "csv") {
      downloadText(`${slugify(exportName ?? tableId)}.csv`, toCSV(header, body));
      toast(`Exported ${formatCount(body.length)} rows`);
    } else {
      const ok = await copyText(toTSV(header, body));
      toast(ok ? `Copied ${formatCount(body.length)} rows as TSV` : "Clipboard unavailable", { tone: ok ? "ok" : "error" });
    }
  };

  const allSelected = selectable && rowItems.length > 0 && rowItems.every((r) => selected?.has(getRowId(r.row)));
  const someSelected = selectable && !allSelected && rowItems.some((r) => selected?.has(getRowId(r.row)));

  const alignClass = (c: Column<T>, base: "th" | "td") =>
    c.align === "right" ? (base === "th" ? s.thRight : s.tdRight) : c.align === "center" ? (base === "th" ? s.thCenter : s.tdCenter) : undefined;

  const pinStyle = (i: number): CSSProperties | undefined =>
    pinOffsets[i] != null ? { left: pinOffsets[i]! } : undefined;

  const shown = rows.length;
  const footer = !hideFooter && (
    <div className={s.footer}>
      <span>
        {loading ? "Loading…" : total != null && total > shown ? `Showing 1–${formatCount(shown)} of ${formatCount(total)}` : `${formatCount(shown)} ${shown === 1 ? "row" : "rows"}`}
      </span>
      {selectable && selected && selected.size > 0 && <span>{formatCount(selected.size)} selected</span>}
      {fetching && !loading && <span>Updating…</span>}
      <span className="spacer" />
      {footerExtra}
      <span className="hide-sm t-caption">
        <span className="mono">j</span>/<span className="mono">k</span> move · <span className="mono">x</span> select · <span className="mono">↵</span> open
      </span>
    </div>
  );

  return (
    <div
      className={s.wrap}
      onPointerDownCapture={keyboard ? () => (activeKeyboardTable = instanceId) : undefined}
      onFocusCapture={keyboard ? () => (activeKeyboardTable = instanceId) : undefined}
    >
      <TableToolbar
        left={toolbar}
        columns={columns}
        hidden={hidden}
        setHidden={setHidden}
        onExport={exportRows}
        canExport={visibleColumns.some((c) => c.csv)}
      />
      <div
        ref={scrollRef}
        className={s.scroller}
        style={{ maxHeight }}
        role="grid"
        tabIndex={0}
        aria-label={exportName ?? tableId}
        onKeyDown={keyboard ? onScrollerKey : undefined}
        aria-rowcount={total ?? rows.length}
        aria-busy={loading || fetching}
      >
        <div className={s.grid} style={{ width: totalWidth, minWidth: "100%" }}>
          <div className={s.header} style={{ gridTemplateColumns: template }} role="row">
            {selectable && (
              <div className={cx(s.th, s.pinned)} style={{ left: 0 }} role="columnheader">
                <Checkbox
                  checked={!!allSelected}
                  indeterminate={!!someSelected}
                  ariaLabel="Select all rows"
                  onChange={(on) => {
                    const next = new Set(selected ?? []);
                    for (const r of rowItems) {
                      if (on) next.add(getRowId(r.row));
                      else next.delete(getRowId(r.row));
                    }
                    onSelectionChange?.(next);
                  }}
                />
              </div>
            )}
            {visibleColumns.map((c, i) => {
              const key = c.sortKey ?? (c.sortValue ? c.id : undefined);
              const sortIdx = key ? sortList.findIndex((x) => x.key === key) : -1;
              const active = sortIdx >= 0 ? sortList[sortIdx] : null;
              return (
                <div
                  key={c.id}
                  role="columnheader"
                  aria-sort={active ? (active.desc ? "descending" : "ascending") : undefined}
                  title={c.headerTitle ?? c.title}
                  className={cx(
                    s.th,
                    key && s.thSortable,
                    active && s.thSorted,
                    alignClass(c, "th"),
                    pinOffsets[i] != null && s.pinned,
                    i === lastPinned && s.pinnedLast,
                  )}
                  style={pinStyle(i)}
                  onClick={key ? (e) => setSort(key, e.shiftKey) : undefined}
                >
                  <span>{c.header}</span>
                  {active && (
                    <span className={s.sortIcon}>
                      {active.desc ? <ArrowDown /> : <ArrowUp />}
                      {sortList.length > 1 && sortIdx + 1}
                    </span>
                  )}
                  <span
                    className={s.resizer}
                    onPointerDown={(e) => startResize(e, c.id, widths[i], c.minWidth ?? 48)}
                    onDoubleClick={(e) => {
                      e.stopPropagation();
                      resetWidth(c.id);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    aria-hidden
                  />
                </div>
              );
            })}
            {rowActions && <div className={cx(s.th, s.actionsHead)} aria-hidden />}
          </div>
          {loading ? (
            <SkeletonRows rows={10} columns={Math.min(7, visibleColumns.length)} />
          ) : items.length === 0 ? (
            <div className={s.empty}>{empty ?? <EmptyState title="No rows" />}</div>
          ) : (
            <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
              {virtualItems.map((v) => {
                const item = items[v.index];
                if (item.kind === "group") {
                  const isCollapsed = collapsed.has(item.key);
                  return (
                    <div
                      key={`g:${item.key}`}
                      className={cx(s.groupRow, isCollapsed && s.groupCollapsed)}
                      style={{ top: v.start, height: rowH, right: 0 }}
                      onClick={() =>
                        setCollapsed((prev) => {
                          const next = new Set(prev);
                          if (next.has(item.key)) next.delete(item.key);
                          else next.add(item.key);
                          return next;
                        })
                      }
                    >
                      <ChevronDown />
                      {renderGroup ? renderGroup(item.key, item.rows) : (
                        <>
                          <strong>{item.key}</strong>
                          <span className="muted">{item.rows.length}</span>
                        </>
                      )}
                    </div>
                  );
                }
                const row = item.row;
                const id = getRowId(row);
                const isSelected = selected?.has(id);
                const isFocused = rowItems[focusIdx]?.row === row;
                return (
                  <div
                    key={id}
                    role="row"
                    aria-selected={isSelected}
                    className={cx(s.row, onRowOpen && s.clickable, isSelected && s.selected, isFocused && s.focused, rowClassName?.(row))}
                    style={{ top: v.start, height: rowH, gridTemplateColumns: template, right: 0 }}
                    onClick={(e) => {
                      if ((e.target as HTMLElement).closest("a,button,input,label")) return;
                      setFocusIdx(item.index);
                      props.onFocusChange?.(row);
                      onRowOpen?.(row, e);
                    }}
                  >
                    {selectable && (
                      <div className={cx(s.td, s.pinned)} style={{ left: 0 }} role="gridcell">
                        <Checkbox
                          checked={!!isSelected}
                          ariaLabel="Select row"
                          onChange={(_on, shift) => toggleRow(row, item.index, shift)}
                        />
                      </div>
                    )}
                    {visibleColumns.map((c, i) => (
                      <div
                        key={c.id}
                        role="gridcell"
                        className={cx(s.td, alignClass(c, "td"), pinOffsets[i] != null && s.pinned, i === lastPinned && s.pinnedLast)}
                        style={pinStyle(i)}
                      >
                        {c.cell(row, item.index)}
                      </div>
                    ))}
                    {rowActions && (
                      <div className={s.actions} role="gridcell">
                        {rowActions(row)}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
      {footer}
    </div>
  );
}

function TableToolbar<T>({
  left,
  columns,
  hidden,
  setHidden,
  onExport,
  canExport,
}: {
  left: ReactNode;
  columns: Column<T>[];
  hidden: Set<string>;
  setHidden: (next: Set<string>) => void;
  onExport: (format: "csv" | "tsv") => void;
  canExport: boolean;
}) {
  const [query, setQuery] = useState("");
  const hideable = columns.filter((c) => c.hideable !== false && !c.pin);
  if (!left && !hideable.length && !canExport) return null;
  return (
    <div className="row" style={{ padding: "8px 10px", borderBottom: "1px solid var(--border)", gap: 8, minHeight: 44, flexWrap: "wrap" }}>
      <div className="row-wrap" style={{ flex: 1, minWidth: 0 }}>
        {left}
      </div>
      {hideable.length > 0 && (
        <Popover.Root>
          <Popover.Trigger asChild>
            <IconButton label="Choose columns" icon={<Columns3 />} size="sm" />
          </Popover.Trigger>
          <Popover.Portal>
            <Popover.Content className={cx(uiStyles.popover, s.chooser)} align="end" sideOffset={4} collisionPadding={8}>
              <div style={{ padding: 8 }}>
                <SearchInput placeholder="Find column" value={query} onChange={(e) => setQuery(e.target.value)} />
              </div>
              <div className={s.chooserList}>
                {hideable
                  .filter((c) => (c.title ?? c.id).toLowerCase().includes(query.toLowerCase()))
                  .map((c) => (
                    <div key={c.id} className={s.chooserItem}>
                      <Checkbox
                        checked={!hidden.has(c.id)}
                        label={c.title ?? (typeof c.header === "string" ? c.header : c.id)}
                        onChange={(on) => {
                          const next = new Set(hidden);
                          if (on) next.delete(c.id);
                          else next.add(c.id);
                          setHidden(next);
                        }}
                      />
                    </div>
                  ))}
              </div>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      )}
      {canExport && (
        <Menu
          trigger={<IconButton label="Table actions" icon={<MoreHorizontal />} size="sm" />}
          items={[
            { label: "Export CSV", icon: <Download />, onSelect: () => onExport("csv") },
            { label: "Copy as TSV (for Sheets)", icon: <Columns3 />, onSelect: () => onExport("tsv") },
          ]}
        />
      )}
    </div>
  );
}
