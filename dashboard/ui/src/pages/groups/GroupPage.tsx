import type { GroupDetailResponse } from "@contract/api-types";
import { useParams } from "@tanstack/react-router";
import { useEffect } from "react";
import { useGroupDetail } from "@/api/hooks/catalog";
import { ChartTooltip, Legend, useTooltip } from "@/charts/core";
import { buildHref, useAppNavigate } from "@/components/AppLink";
import { MetaItem, PageHeader } from "@/components/PageHeader";
import { ErrorPanel, ModelDot, Panel, Skeleton, Tabs } from "@/components/primitives";
import { formatCount } from "@/lib/format";
import { absoluteLocal } from "@/lib/time";
import { useSubjectSlots } from "@/state/colors";
import { useSearchParams } from "@/state/nav";
import { pushRecent } from "@/state/prefs";
import { ComparePage } from "../compare/ComparePage";
import { distinctLabels } from "../compare/types";
import { NotFoundPage } from "../NotFoundPage";
import { RunsTable } from "../runs/RunsTable";

const STATUS_BG = { ok: "var(--seq-3)", failed: "var(--status-failed)", missing: "transparent" } as const;

function CoverageGrid({ g }: { g: GroupDetailResponse }) {
  const tip = useTooltip();
  const navigate = useAppNavigate();
  const slots = useSubjectSlots(g.subjects.map((s) => s.key));
  const byKey = new Map(g.coverage.map((c) => [`${c.subject}|${c.task_name}`, c]));
  const labels = distinctLabels(
    g.subjects.map((s) => s.key),
    new Map(g.subjects.map((s) => [s.key, s])),
  );
  const cell = 16;
  return (
    <Panel
      title="Coverage"
      caption="Which subject × task combinations exist. Filled = result, red = failed, hatched = missing. Click a filled cell to open that task result."
      actions={<Legend items={[{ label: "result", color: STATUS_BG.ok }, { label: "failed", color: STATUS_BG.failed }, { label: "missing", color: "var(--border-strong)", hollow: true }]} />}
    >
      <div style={{ overflow: "auto" }} onMouseLeave={tip.hide}>
        <div style={{ display: "grid", gridTemplateColumns: `220px repeat(${g.tasks.length}, ${cell}px)`, gap: 2, alignItems: "end", width: "max-content" }}>
          <span />
          {g.tasks.map((t) => (
            <span key={t} title={t} style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 10, color: "var(--muted)", height: 150, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {t}
            </span>
          ))}
          {g.subjects.map((s) => (
            <div key={s.key} style={{ display: "contents" }}>
              <span className="row truncate" style={{ fontSize: 12, gap: 6, height: cell }}>
                <ModelDot slot={slots[s.key]} />
                <span className="truncate">{labels.get(s.key) ?? s.label}</span>
              </span>
              {g.tasks.map((t) => {
                const c = byKey.get(`${s.key}|${t}`);
                const status = c?.status ?? "missing";
                return (
                  <span
                    key={t}
                    style={{
                      width: cell,
                      height: cell,
                      borderRadius: 2,
                      background: status === "missing" ? "repeating-linear-gradient(45deg, var(--border) 0 1px, transparent 1px 4px)" : STATUS_BG[status],
                      border: status === "missing" ? "1px solid var(--border)" : undefined,
                      cursor: c?.run_id ? "pointer" : "default",
                    }}
                    onClick={() => c?.run_id && navigate(buildHref(`/runs/${c.run_id}`, { tab: "tasks", task: t }))}
                    onMouseMove={(e) =>
                      tip.show(
                        e,
                        <>
                          <div style={{ fontWeight: 800 }}>{t}</div>
                          <div>{s.label}</div>
                          <div className="muted">{status === "ok" ? "Has a result" : status === "failed" ? "Failed" : "No result"}</div>
                        </>,
                      )
                    }
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <ChartTooltip state={tip.state} />
    </Panel>
  );
}

export function GroupPage() {
  const { groupName } = useParams({ strict: false }) as { groupName: string };
  const name = decodeURIComponent(groupName);
  const [search, setSearch] = useSearchParams<{ tab?: string; sort?: string }>();
  const g = useGroupDetail(name);
  useEffect(() => pushRecent({ type: "group", key: name, label: name, href: `/groups/${encodeURIComponent(name)}` }), [name]);
  if (g.error) return (g.error as { status?: number }).status === 404 ? <NotFoundPage what={`Group ${name}`} /> : <div className="page"><ErrorPanel error={g.error} /></div>;
  const d = g.data;
  const defaultTab = d && d.n_models >= 2 ? "compare" : "runs";
  const tab = (search.tab as "runs" | "coverage" | "compare") ?? defaultTab;
  return (
    <div className="page">
      <PageHeader
        crumbs={[{ label: "Groups", href: "/groups" }]}
        title={name}
        meta={
          d ? (
            <>
              <MetaItem>{formatCount(d.n_runs)} runs</MetaItem>
              <MetaItem>{formatCount(d.n_models)} models</MetaItem>
              <MetaItem>{formatCount(d.n_tasks)} tasks</MetaItem>
              <MetaItem label="by">{d.users.join(", ")}</MetaItem>
              <MetaItem>
                {absoluteLocal(d.first_run_at)} – {absoluteLocal(d.last_run_at)}
              </MetaItem>
            </>
          ) : (
            <Skeleton width={320} height={14} />
          )
        }
      />
      <Tabs
        label="Group sections"
        value={tab}
        onChange={(v) => setSearch({ tab: v === defaultTab ? undefined : v })}
        tabs={[
          { value: "compare", label: "Compare" },
          { value: "runs", label: "Runs", count: d?.n_runs },
          { value: "coverage", label: "Coverage" },
        ]}
      />
      {!d ? (
        <Skeleton height={400} />
      ) : tab === "runs" ? (
        <Panel pad="none">
          <RunsTable tableId="group-runs" params={{ group: name, sort: search.sort ?? "-created_at", limit: 100 }} sort={search.sort} onSortChange={(sort) => setSearch({ sort }, { replace: true })} />
        </Panel>
      ) : tab === "coverage" ? (
        <CoverageGrid g={d} />
      ) : (
        <ComparePage presetGroup={name} />
      )}
    </div>
  );
}
