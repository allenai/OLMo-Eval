import { Download, FileSpreadsheet, Link2, MoreHorizontal, Table2 } from "lucide-react";
import { type ReactNode, type RefObject, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { IconButton, Menu, Panel } from "@/components/primitives";
import { toast } from "@/components/toast";
import { type Cell, copyText, downloadText, slugify, toCSV } from "@/lib/csv";
import c from "./charts.module.css";

/** Width (and height) of an element, updated on resize. */
export function useSize<T extends HTMLElement>(): [RefObject<T | null>, { width: number; height: number }] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setSize({ width: el.clientWidth, height: el.clientHeight });
    update();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

export interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** Floating tooltip at viewport coordinates, flipped to stay on screen. */
export function ChartTooltip({ state }: { state: TooltipState | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);
  useLayoutEffect(() => {
    if (!state || !ref.current) {
      setPos(null);
      return;
    }
    const rect = ref.current.getBoundingClientRect();
    let left = state.x + 14;
    let top = state.y + 14;
    if (left + rect.width > window.innerWidth - 8) left = state.x - rect.width - 14;
    if (top + rect.height > window.innerHeight - 8) top = state.y - rect.height - 14;
    setPos({ left: Math.max(8, left), top: Math.max(8, top) });
  }, [state]);
  if (!state) return null;
  return createPortal(
    <div ref={ref} className={c.tooltip} style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999 }} role="tooltip">
      {state.content}
    </div>,
    document.body,
  );
}

export function useTooltip() {
  const [state, setState] = useState<TooltipState | null>(null);
  return {
    state,
    show: (e: { clientX: number; clientY: number }, content: ReactNode) => setState({ x: e.clientX, y: e.clientY, content }),
    hide: () => setState(null),
  };
}

export function TipRow({ color, label, value, dashed }: { color?: string; label: ReactNode; value: ReactNode; dashed?: boolean }) {
  return (
    <div className={c.tipRow}>
      {color && <span className={c.tipSwatch} style={dashed ? { background: "transparent", border: `2px solid ${color}` } : { background: color }} />}
      <span className={c.tipLabel}>{label}</span>
      <span className={c.tipValue}>{value}</span>
    </div>
  );
}

function resolveVars(text: string, style: CSSStyleDeclaration): string {
  return text.replace(/var\((--[a-z0-9-]+)\)/gi, (_m, name: string) => style.getPropertyValue(name).trim() || "#888");
}

/** Render the first SVG inside `container` to a PNG download. */
export async function downloadSvgPng(container: HTMLElement | null, filename: string): Promise<void> {
  const svg = container?.querySelector("svg");
  if (!svg) {
    toast("This chart cannot be exported as PNG", { tone: "info" });
    return;
  }
  const style = getComputedStyle(document.documentElement);
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const { width, height } = svg.getBoundingClientRect();
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const bg = style.getPropertyValue("--surface").trim() || "#fff";
  const xml = resolveVars(new XMLSerializer().serializeToString(clone), style).replace(
    /font-family:[^;"]+/g,
    "font-family: sans-serif",
  );
  const img = new Image();
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml" }));
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("render failed"));
    img.src = url;
  });
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0, width, height);
  URL.revokeObjectURL(url);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${slugify(filename)}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}

/**
 * Chart panel: title, one-line caption, and a menu with View as table, Download PNG,
 * Download CSV and Copy link to this chart.
 */
export function ChartPanel({
  title,
  caption,
  actions,
  csv,
  table,
  children,
  refetching,
  id,
  legend,
  className,
}: {
  title: ReactNode;
  caption?: ReactNode;
  actions?: ReactNode;
  csv?: () => { header: string[]; rows: Cell[][] };
  table?: () => ReactNode;
  children: ReactNode;
  refetching?: boolean;
  id?: string;
  legend?: ReactNode;
  className?: string;
}) {
  const [asTable, setAsTable] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const name = typeof title === "string" ? title : (id ?? "chart");
  const items = [
    ...(table ? [{ label: asTable ? "View as chart" : "View as table", icon: <Table2 />, onSelect: () => setAsTable(!asTable) }] : []),
    { label: "Download PNG", icon: <Download />, onSelect: () => void downloadSvgPng(bodyRef.current, name) },
    ...(csv
      ? [
          {
            label: "Download CSV",
            icon: <FileSpreadsheet />,
            onSelect: () => {
              const { header, rows } = csv();
              downloadText(`${slugify(name)}.csv`, toCSV(header, rows));
            },
          },
        ]
      : []),
    {
      label: "Copy link to this chart",
      icon: <Link2 />,
      onSelect: () => {
        const url = `${window.location.href.split("#")[0]}${id ? `#${id}` : ""}`;
        copyText(url).then((ok) => ok && toast("Link copied"));
      },
    },
  ];
  return (
    <Panel
      id={id}
      className={className}
      pad="chart"
      title={title}
      caption={caption}
      refetching={refetching}
      actions={
        <>
          {actions}
          <Menu trigger={<IconButton size="sm" label="Chart options" icon={<MoreHorizontal />} />} items={items} />
        </>
      }
    >
      {legend && <div className={c.legendRow}>{legend}</div>}
      <div ref={bodyRef}>{asTable && table ? table() : children}</div>
    </Panel>
  );
}

export function Legend({ items }: { items: { label: ReactNode; color: string; dashed?: boolean; hollow?: boolean }[] }) {
  return (
    <div className={c.legend}>
      {items.map((it, i) => (
        <span key={i} className={c.legendItem}>
          {it.dashed ? (
            <svg width="16" height="8" aria-hidden>
              <line x1="0" y1="4" x2="16" y2="4" stroke={it.color} strokeWidth="2" strokeDasharray="4 3" />
            </svg>
          ) : (
            <span
              className={c.legendSwatch}
              style={it.hollow ? { background: "transparent", border: `2px solid ${it.color}` } : { background: it.color }}
            />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}

/** "Nice" ticks for a numeric domain. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (hi === lo) return [lo];
  const span = hi - lo;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const norm = step0 / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export function ChartEmpty({ children, height = 160 }: { children: ReactNode; height?: number }) {
  return (
    <div className={c.empty} style={{ height }}>
      {children}
    </div>
  );
}
