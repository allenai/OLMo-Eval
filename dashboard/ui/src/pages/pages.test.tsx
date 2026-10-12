/** Page render tests against the MSW mock API. */
import { QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { getWorld } from "@/mocks/fixtures";
import { createQueryClient } from "@/queryClient";
import { createAppRouter } from "@/router";

function renderAt(url: string) {
  const router = createAppRouter({ history: createMemoryHistory({ initialEntries: [url] }) });
  const qc = createQueryClient();
  render(
    <QueryClientProvider client={qc}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

const world = getWorld();
const run = world.runs.find((r) => r.summary.status === "partial" && r.model.series === "olmo3-7b-midtrain")!;
const runId = run.summary.run_id;
const group = run.summary.experiment_group!;

describe("pages", () => {
  it("Home shows recent runs, groups and activity", async () => {
    renderAt("/");
    expect(await screen.findByText(/Your recent runs/i)).toBeInTheDocument();
    expect(await screen.findByText("Active groups")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText(group).length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Runs list renders rows and score columns from the URL", async () => {
    renderAt("/runs?cols=suite:olmes:base&series=olmo3-7b-midtrain");
    expect(await screen.findByRole("heading", { name: "Runs" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("olmo3-7b-midtrain").length).toBeGreaterThan(3), { timeout: 5000 });
    expect(screen.getAllByText("olmes:base").length).toBeGreaterThan(0);
  });

  it("Run overview shows the suite tree and health", async () => {
    renderAt(`/runs/${runId}`);
    expect(await screen.findByText("Suites and tasks", {}, { timeout: 5000 })).toBeInTheDocument();
    expect(await screen.findByText("Health")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("olmes:base").length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Run tasks and config tabs render", async () => {
    renderAt(`/runs/${runId}?tab=tasks`);
    await waitFor(() => expect(screen.getAllByText("gsm8k:cot:olmo3").length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Compare heatmap renders the group's subjects", async () => {
    renderAt(`/compare?group=${group}&scope=suite:olmes:base`);
    expect(await screen.findByText("Heatmap", { selector: "h2" }, { timeout: 8000 })).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("mmlu_anatomy:mc").length).toBeGreaterThan(0), { timeout: 8000 });
  });

  it("Compare leaves out unknown subjects instead of failing", async () => {
    const other = world.runs.find((r) => r.summary.run_id !== runId && r.model.series === "olmo3-7b-midtrain")!.summary.run_id;
    renderAt(`/compare?subjects=r:${runId},r:${other},r:zzzzzzzzzzzz&scope=suite:arc:rc`);
    expect(await screen.findByText(/1 unknown subject ignored/, {}, { timeout: 8000 })).toBeInTheDocument();
    expect(await screen.findByText("Heatmap", { selector: "h2" }, { timeout: 8000 })).toBeInTheDocument();
    expect(screen.queryByText(/Something went wrong/)).not.toBeInTheDocument();
  });

  it("Compare pairwise renders win rates", async () => {
    renderAt(`/compare?group=${group}&scope=suite:arc:rc&view=pairwise`);
    await waitFor(() => expect(screen.getAllByText(/% wins/).length).toBeGreaterThan(2), { timeout: 10000 });
    expect(screen.getByText("mean win rate")).toBeInTheDocument();
  });

  it("Compare disagreements list instances as a keyboard listbox", async () => {
    renderAt(`/compare?group=${group}&scope=suite:arc:rc&view=disagree`);
    const list = await screen.findByRole("listbox", { name: "Disagreeing instances" }, { timeout: 10000 });
    await waitFor(() => expect(list.querySelectorAll('[role="option"]').length).toBeGreaterThan(0), { timeout: 10000 });
    expect(screen.getAllByRole("button", { name: /instances\. List them\./ }).length).toBeGreaterThan(0);
  });

  it("Groups, Models and Tasks indexes list their rows with totals", async () => {
    renderAt("/groups");
    await waitFor(() => expect(screen.getAllByText(group).length).toBeGreaterThan(0), { timeout: 5000 });
    cleanup();
    renderAt("/models");
    await waitFor(() => expect(screen.getAllByText("olmo3-7b-midtrain").length).toBeGreaterThan(0), { timeout: 5000 });
    cleanup();
    renderAt("/tasks");
    await waitFor(() => expect(screen.getAllByText("gsm8k:cot:olmo3").length).toBeGreaterThan(0), { timeout: 5000 });
  });

  it("Unknown runs show a not-found state", async () => {
    renderAt("/runs/zzzzzzzzzzzz");
    expect(await screen.findByText(/not found/i, {}, { timeout: 5000 })).toBeInTheDocument();
  });
});
