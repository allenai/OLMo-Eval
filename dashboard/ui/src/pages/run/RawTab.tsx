import { JsonTree } from "@/components/Json";
import { Panel } from "@/components/primitives";
import type { RunCtx } from "./types";

export function RawTab(ctx: RunCtx) {
  return (
    <Panel title="Run record" caption="The full run record as returned by GET /api/runs/{run_id}.">
      <JsonTree value={ctx.run} filename={`run-${ctx.run.run_id}.json`} maxHeight="calc(100vh - 320px)" />
    </Panel>
  );
}
