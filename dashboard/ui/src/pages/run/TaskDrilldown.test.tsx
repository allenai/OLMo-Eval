import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { runDetail, runTaskResults } from "@/mocks/api";
import { getWorld } from "@/mocks/fixtures";
import { createQueryClient } from "@/queryClient";
import { TaskDrilldown } from "./TaskDrilldown";

describe("TaskDrilldown generation diagnostics", () => {
  it("keeps counts visible when a task fails", () => {
    const id = getWorld().runs[0].summary.run_id;
    const run = runDetail(id);
    const task = runTaskResults(id, new URLSearchParams()).items[0];
    const row = { ...task, error: "Too many inference failures", generation_counts: { generations: 50, cap_hit: 40, empty: 38 } };
    render(
      <QueryClientProvider client={createQueryClient()}>
        <TaskDrilldown run={run} row={row} onOpenInstances={() => {}} />
      </QueryClientProvider>,
    );

    expect(screen.getByText("cap 40/50 · empty 38")).toBeInTheDocument();
  });
});
