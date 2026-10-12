import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronDown } from "lucide-react";
import { type KeyboardEvent, type ReactNode, useId, useMemo, useRef, useState } from "react";
import { cx } from "@/components/primitives";
import { ChartTooltip, useTooltip } from "./core";
import c from "./charts.module.css";

export interface HeatColumn {
  key: string;
  label: ReactNode;
  title: string;
  sub?: ReactNode;
  active?: boolean;
  onClick?: () => void;
}

export interface HeatCell {
  text: string;
  background?: string;
  color?: string;
  sig?: boolean;
  status?: "ok" | "missing" | "failed" | "partial";
  hashMismatch?: boolean;
  tooltip?: ReactNode;
  onClick?: () => void;
  muted?: boolean;
}

export interface HeatRow {
  key: string;
  label: string;
  kind: "task" | "suite";
  depth: number;
  parent: string | null;
  cells: HeatCell[];
  badge?: ReactNode;
  onLabelClick?: () => void;
}

const ROW_H = 26;
const HEAD_H = 46;

/**
 * Task × subject matrix with sticky headers, collapsible suite rows and row virtualization.
 * Keyboard: focus the grid, then arrow keys move the focused cell, Enter opens it, and Space
 * (or Enter on a row header) collapses a suite.
 */
export function Heatmap({
  columns,
  rows,
  rowHeaderWidth = 240,
  cellWidth = 64,
  cornerLabel = "Task",
  collapsible = true,
  maxHeight,
  label = "Heatmap",
}: {
  columns: HeatColumn[];
  rows: HeatRow[];
  rowHeaderWidth?: number;
  cellWidth?: number;
  cornerLabel?: ReactNode;
  collapsible?: boolean;
  maxHeight?: number | string;
  label?: string;
}) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<{ row: string; col: number } | null>(null);
  // Focused cell: col -1 is the row header.
  const [focus, setFocus] = useState<{ row: number; col: number } | null>(null);
  const tip = useTooltip();
  const scrollRef = useRef<HTMLDivElement>(null);
  const domId = useId();

  // Hide rows whose ancestor suite is collapsed.
  const visible = useMemo(() => {
    const parentOf = new Map(rows.map((r) => [r.key, r.parent]));
    const suiteKey = (name: string | null) => (name ? `suite:${name}` : null);
    return rows.filter((r) => {
      let p = suiteKey(r.parent);
      while (p) {
        if (collapsed.has(p)) return false;
        const next = parentOf.get(p);
        p = suiteKey(next ?? null);
      }
      return true;
    });
  }, [rows, collapsed]);

  // TanStack Virtual returns non-memoizable functions; the React Compiler skips this component.
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: visible.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_H,
    overscan: 20,
    scrollPaddingStart: HEAD_H,
    initialRect: { width: 1200, height: 720 },
  });

  const toggle = (key: string) =>
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const activateLabel = (row: HeatRow) => {
    if (row.kind === "suite" && collapsible) toggle(row.key);
    else row.onLabelClick?.();
  };

  const showFocusTip = (r: number, col: number) => {
    const cell = visible[r]?.cells[col];
    const el = document.getElementById(`${domId}-${r}-${col}`);
    if (!cell?.tooltip || !el) return tip.hide();
    const rect = el.getBoundingClientRect();
    tip.show({ clientX: rect.right, clientY: rect.bottom }, cell.tooltip);
  };

  const moveFocus = (r: number, col: number) => {
    const row = Math.max(0, Math.min(visible.length - 1, r));
    const column = Math.max(-1, Math.min(columns.length - 1, col));
    setFocus({ row, col: column });
    setHover({ row: visible[row]?.key ?? "", col: column });
    virtualizer.scrollToIndex(row, { align: "auto" });
    // Wait for the virtualizer to render the row before measuring it.
    requestAnimationFrame(() => {
      document.getElementById(`${domId}-${row}-${column}`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
      if (column >= 0) showFocusTip(row, column);
      else tip.hide();
    });
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget || !visible.length) return;
    const f = focus ?? { row: 0, col: 0 };
    const page = Math.max(1, Math.floor((scrollRef.current?.clientHeight ?? 400) / ROW_H) - 2);
    const moves: Record<string, () => void> = {
      ArrowDown: () => moveFocus(f.row + (focus ? 1 : 0), f.col),
      ArrowUp: () => moveFocus(f.row - 1, f.col),
      ArrowRight: () => moveFocus(f.row, f.col + (focus ? 1 : 0)),
      ArrowLeft: () => moveFocus(f.row, f.col - 1),
      PageDown: () => moveFocus(f.row + page, f.col),
      PageUp: () => moveFocus(f.row - page, f.col),
      Home: () => moveFocus(e.ctrlKey || e.metaKey ? 0 : f.row, e.ctrlKey || e.metaKey ? f.col : -1),
      End: () => moveFocus(e.ctrlKey || e.metaKey ? visible.length - 1 : f.row, e.ctrlKey || e.metaKey ? f.col : columns.length - 1),
    };
    if (moves[e.key]) {
      e.preventDefault();
      moves[e.key]();
      return;
    }
    if (!focus) return;
    const row = visible[focus.row];
    if (!row) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      if (focus.col < 0 || (e.key === " " && row.kind === "suite")) activateLabel(row);
      else row.cells[focus.col]?.onClick?.();
    } else if (e.key === "Escape") {
      setFocus(null);
      tip.hide();
    }
  };

  // Row headers give up width on phones so at least a few subject columns stay visible.
  const headerWidth = `min(${rowHeaderWidth}px, 38vw)`;
  const template = `${headerWidth} repeat(${columns.length}, minmax(${cellWidth}px, 1fr))`;
  const gridWidth = `max(100%, calc(${headerWidth} + ${columns.length * cellWidth}px))`;
  const activeId = focus ? `${domId}-${focus.row}-${focus.col}` : undefined;

  return (
    <div
      ref={scrollRef}
      className={c.heat}
      style={{ maxHeight }}
      role="grid"
      aria-label={label}
      aria-rowcount={visible.length + 1}
      aria-colcount={columns.length + 1}
      aria-activedescendant={activeId}
      tabIndex={0}
      onKeyDown={onKeyDown}
      onBlur={() => (setFocus(null), tip.hide())}
      onMouseLeave={() => (setHover(null), tip.hide())}
    >
      <div style={{ width: gridWidth }}>
        <div className={c.heatHead} style={{ gridTemplateColumns: template }} role="row" aria-rowindex={1}>
          <div className={c.heatCorner} role="columnheader">
            {cornerLabel}
          </div>
          {columns.map((col, j) => (
            <div
              key={col.key}
              role="columnheader"
              className={cx(c.heatColHead, col.onClick && c.heatColHeadClickable, (col.active || hover?.col === j) && c.heatColHeadActive)}
              title={col.title}
              aria-sort={col.active ? "descending" : undefined}
            >
              {col.onClick ? (
                <button type="button" className={c.heatColButton} onClick={col.onClick} aria-label={col.title}>
                  <span className={c.heatColName}>{col.label}</span>
                  {col.sub && <span className={c.heatColSub}>{col.sub}</span>}
                </button>
              ) : (
                <>
                  <span className={c.heatColName}>{col.label}</span>
                  {col.sub && <span className={c.heatColSub}>{col.sub}</span>}
                </>
              )}
            </div>
          ))}
        </div>
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {virtualizer.getVirtualItems().map((v) => {
            const row = visible[v.index];
            const r = v.index;
            const isSuite = row.kind === "suite";
            const isCollapsed = collapsed.has(row.key);
            return (
              <div
                key={row.key}
                role="row"
                aria-rowindex={r + 2}
                className={c.heatRow}
                style={{ top: v.start, height: ROW_H, gridTemplateColumns: template }}
              >
                <div
                  id={`${domId}-${r}--1`}
                  role="rowheader"
                  aria-expanded={isSuite && collapsible ? !isCollapsed : undefined}
                  className={cx(
                    c.heatRowHead,
                    isSuite && c.heatSuite,
                    hover?.row === row.key && c.heatRowHeadActive,
                    focus?.row === r && focus.col === -1 && c.heatFocused,
                  )}
                  style={{ paddingLeft: 8 + row.depth * 14 }}
                  title={isSuite && collapsible ? `${row.label} (click to ${isCollapsed ? "expand" : "collapse"})` : row.label}
                  onClick={() => activateLabel(row)}
                >
                  {isSuite && collapsible && <ChevronDown className={cx(c.heatChevron, isCollapsed && c.heatChevronClosed)} aria-hidden />}
                  <span>{row.label}</span>
                  {row.badge}
                </div>
                {row.cells.map((cell, j) => (
                  <div
                    key={j}
                    id={`${domId}-${r}-${j}`}
                    role="gridcell"
                    aria-label={`${row.label}, ${columns[j]?.title.split(" — ")[0] ?? ""}: ${cell.status === "missing" ? "no result" : cell.status === "failed" ? "failed" : cell.text}`}
                    className={cx(
                      c.heatCell,
                      isSuite && c.heatCellSuite,
                      cell.status === "missing" && c.heatMissing,
                      cell.status === "failed" && c.heatFailed,
                      !cell.onClick && c.heatCellStatic,
                      focus?.row === r && focus.col === j && c.heatFocused,
                    )}
                    style={{
                      background: cell.status === "missing" || cell.status === "failed" ? undefined : cell.background,
                      color: cell.color ?? (cell.muted ? "var(--muted)" : undefined),
                    }}
                    onMouseMove={(e) => {
                      if (hover?.row !== row.key || hover.col !== j) setHover({ row: row.key, col: j });
                      if (cell.tooltip) tip.show(e, cell.tooltip);
                    }}
                    onClick={cell.onClick}
                  >
                    {cell.status === "failed" ? "failed" : cell.status === "missing" ? "" : cell.text}
                    {cell.sig && <span className={c.heatSig} />}
                    {cell.hashMismatch && <span className={c.heatHash} title="Task config differs" />}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>
      <ChartTooltip state={tip.state} />
    </div>
  );
}
