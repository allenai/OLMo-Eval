import { ChevronRight, Copy, Download, Search } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";
import { copyText, downloadText } from "@/lib/csv";
import { diffSummary, type JsonDiffEntry, jsonDiff } from "@/lib/diff";
import { Button, Checkbox, cx, SearchInput } from "./primitives";
import { toast } from "./toast";
import j from "./json.module.css";
import { pressable } from "@/lib/pressable";

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

function preview(value: unknown): string {
  if (Array.isArray(value)) return `[${value.length}]`;
  if (isObj(value)) return `{${Object.keys(value).length}}`;
  return "";
}

function Scalar({ value }: { value: unknown }) {
  if (value === null) return <span className={j.null}>null</span>;
  if (typeof value === "string") return <span className={j.string}>"{value}"</span>;
  if (typeof value === "number") return <span className={j.number}>{String(value)}</span>;
  if (typeof value === "boolean") return <span className={j.bool}>{String(value)}</span>;
  return <span>{String(value)}</span>;
}

function matches(value: unknown, key: string, q: string): boolean {
  if (!q) return true;
  if (key.toLowerCase().includes(q)) return true;
  if (isObj(value)) return Object.entries(value).some(([k, v]) => matches(v, k, q));
  return String(value).toLowerCase().includes(q);
}

function Node({
  name,
  value,
  path,
  depth,
  expandAll,
  query,
}: {
  name: string | null;
  value: unknown;
  path: string;
  depth: number;
  expandAll: boolean | null;
  query: string;
}) {
  const [open, setOpen] = useState(depth < 2);
  const isOpen = expandAll ?? (query ? true : open);
  const container = isObj(value);
  if (!matches(value, name ?? "", query)) return null;
  const copyPath = () => {
    copyText(path).then((ok) => ok && toast(`Copied path ${path}`));
  };
  const copyValue = () => {
    copyText(typeof value === "string" ? value : JSON.stringify(value, null, 2)).then((ok) => ok && toast("Copied value"));
  };
  return (
    <div className={j.node}>
      <div className={j.line}>
        {container ? (
          <button type="button" className={cx(j.toggle, isOpen && j.open)} onClick={() => setOpen(!isOpen)} aria-label={isOpen ? "Collapse" : "Expand"}>
            <ChevronRight />
          </button>
        ) : (
          <span className={j.toggleSpacer} />
        )}
        {name != null && (
          <span className={j.key} {...pressable(copyPath)} title={`Copy path ${path}`}>
            {name}
          </span>
        )}
        {name != null && <span className={j.colon}>:</span>}
        {container ? (
          <span className={j.preview} {...pressable(() => setOpen(!isOpen))} aria-expanded={isOpen}>
            {Array.isArray(value) ? "[" : "{"}
            {!isOpen && <span className={j.count}>{preview(value)}</span>}
            {!isOpen && (Array.isArray(value) ? "]" : "}")}
          </span>
        ) : (
          <span className={j.value} {...pressable(copyValue)} title="Copy value">
            <Scalar value={value} />
          </span>
        )}
      </div>
      {container && isOpen && (
        <div className={j.children}>
          {(Array.isArray(value) ? value.map((v, i) => [String(i), v] as const) : Object.entries(value)).map(([k, v]) => (
            <Node
              key={k}
              name={k}
              value={v}
              path={Array.isArray(value) ? `${path}[${k}]` : path ? `${path}.${k}` : k}
              depth={depth + 1}
              expandAll={expandAll}
              query={query}
            />
          ))}
          <div className={j.close}>{Array.isArray(value) ? "]" : "}"}</div>
        </div>
      )}
    </div>
  );
}

export function JsonTree({
  value,
  filename = "data.json",
  toolbar = true,
  maxHeight,
}: {
  value: unknown;
  filename?: string;
  toolbar?: boolean;
  maxHeight?: number | string;
}) {
  const [query, setQuery] = useState("");
  const [expandAll, setExpandAll] = useState<boolean | null>(null);
  return (
    <div className={j.wrap}>
      {toolbar && (
        <div className={j.toolbar}>
          <SearchInput placeholder="Search keys and values" value={query} onChange={(e) => setQuery(e.target.value)} wrapStyle={{ flex: 1, maxWidth: 320 }} />
          <Button size="sm" onClick={() => setExpandAll(true)}>
            Expand all
          </Button>
          <Button size="sm" onClick={() => setExpandAll(false)}>
            Collapse
          </Button>
          <span className="spacer" />
          <Button
            size="sm"
            icon={<Copy />}
            onClick={() => copyText(JSON.stringify(value, null, 2)).then((ok) => ok && toast("Copied JSON"))}
          >
            Copy
          </Button>
          <Button size="sm" icon={<Download />} onClick={() => downloadText(filename, JSON.stringify(value, null, 2), "application/json")}>
            Download
          </Button>
        </div>
      )}
      <div className={j.tree} style={{ maxHeight }} onDoubleClick={() => setExpandAll(null)}>
        <Node name={null} value={value} path="" depth={0} expandAll={expandAll} query={query.toLowerCase()} />
      </div>
    </div>
  );
}

function short(value: unknown): ReactNode {
  if (value === undefined) return <span className={j.null}>—</span>;
  const text = typeof value === "string" ? `"${value}"` : JSON.stringify(value);
  return <span title={text}>{text && text.length > 120 ? `${text.slice(0, 117)}…` : text}</span>;
}

/** Structural diff with unchanged entries collapsed by default. */
export function JsonDiff({
  a,
  b,
  aLabel = "Baseline",
  bLabel = "This run",
}: {
  a: unknown;
  b: unknown;
  aLabel?: string;
  bLabel?: string;
}) {
  const [showSame, setShowSame] = useState(false);
  const entries: JsonDiffEntry[] = useMemo(() => jsonDiff(a, b), [a, b]);
  const summary = diffSummary(entries);
  const visible = showSame ? entries : entries.filter((e) => e.kind !== "same");
  return (
    <div className={j.wrap}>
      <div className={j.toolbar}>
        <span className="t-caption">
          <span className={j.chg}>{summary.changed} changed</span> · <span className={j.add}>{summary.added} added</span> ·{" "}
          <span className={j.rem}>{summary.removed} removed</span> · {summary.same} unchanged
        </span>
        <span className="spacer" />
        <Checkbox checked={showSame} onChange={setShowSame} label="Show unchanged" />
      </div>
      {visible.length === 0 ? (
        <div className="muted" style={{ padding: 12, fontSize: 12.5 }}>
          <Search style={{ width: 13, height: 13, verticalAlign: -2 }} /> No differences.
        </div>
      ) : (
        <div className={j.diffTable} role="table">
          <div className={j.diffHead} role="row">
            <span role="columnheader">Path</span>
            <span role="columnheader">{aLabel}</span>
            <span role="columnheader">{bLabel}</span>
          </div>
          {visible.map((e) => (
            <div key={e.path} className={cx(j.diffRow, j[`k_${e.kind}`])} role="row">
              <span role="cell" className="mono">{e.path || "(root)"}</span>
              <span role="cell" className="mono">{e.kind === "added" ? <span className={j.null}>—</span> : short(e.a)}</span>
              <span role="cell" className="mono">{e.kind === "removed" ? <span className={j.null}>—</span> : short(e.b)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
