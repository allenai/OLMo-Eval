import {
  createRootRoute,
  createRoute,
  createRouter,
  lazyRouteComponent,
  retainSearchParams,
} from "@tanstack/react-router";
import { AppShell } from "./components/shell/AppShell";
import { NotFoundPage } from "./pages/NotFoundPage";
import { RUN_FILTER_KEYS } from "./lib/filters";
import { parseSearch, pickStrings, stringifySearch } from "./lib/url";

/** Search keys each route accepts; anything else in the URL is ignored. */
export const SEARCH_KEYS = {
  root: ["baseline"],
  runs: [...RUN_FILTER_KEYS, "sort", "cols", "gb"],
  run: ["tab", "task", "hash", "inst", "cell", "correct", "thr", "fin", "len", "score", "has", "iq", "isort", "only", "side", "diff", "tq", "tf", "all", "sel", "ts", "rt"],
  runTask: ["hash"],
  compare: ["subjects", "group", "merge", "scope", "metric", "view", "mode", "alpha", "margin", "shared", "a", "b", "task", "cell", "filter", "rsort", "scale", "inst", "x", "size", "plevel", "pscale", "porder", "pbrush"],
  models: ["q", "family", "sort"],
  model: ["variant", "scope", "tasks", "x", "refs", "merge", "delta", "sel"],
  tasks: ["q", "suite", "sort", "tab"],
  task: ["hash", "metric", "pm", "family", "group", "user", "sort", "gpu", "rt"],
  suite: ["pm", "family", "group", "user", "sort"],
  groups: ["q", "sort"],
  group: ["tab", "sort", "subjects", "merge", "scope", "metric", "view", "mode", "alpha", "margin", "shared", "a", "b", "task", "cell", "filter", "rsort", "scale", "inst", "x", "size", "plevel", "pscale", "porder", "pbrush"],
  views: [] as string[],
} as const;

function searchFor<K extends string>(keys: readonly K[]) {
  return (raw: Record<string, unknown>) => pickStrings(raw, [...keys, "baseline"] as (K | "baseline")[]);
}

const rootRoute = createRootRoute({
  validateSearch: searchFor(SEARCH_KEYS.root),
  search: { middlewares: [retainSearchParams<{ baseline?: string }>(["baseline"])] },
  component: AppShell,
  notFoundComponent: () => <NotFoundPage />,
});

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: lazyRouteComponent(() => import("./pages/home/HomePage"), "HomePage"),
});

const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs",
  validateSearch: searchFor(SEARCH_KEYS.runs),
  component: lazyRouteComponent(() => import("./pages/runs/RunsPage"), "RunsPage"),
});

const runRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId",
  validateSearch: searchFor(SEARCH_KEYS.run),
  component: lazyRouteComponent(() => import("./pages/run/RunPage"), "RunPage"),
});

const runTaskRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/runs/$runId/tasks/$taskName",
  validateSearch: searchFor(SEARCH_KEYS.runTask),
  component: lazyRouteComponent(() => import("./pages/run/TaskDrilldownPage"), "TaskDrilldownPage"),
});

const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  validateSearch: searchFor(SEARCH_KEYS.compare),
  component: lazyRouteComponent(() => import("./pages/compare/ComparePage"), "ComparePage"),
});

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/models",
  validateSearch: searchFor(SEARCH_KEYS.models),
  component: lazyRouteComponent(() => import("./pages/models/ModelsPage"), "ModelsPage"),
});

const modelRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/models/$series",
  validateSearch: searchFor(SEARCH_KEYS.model),
  component: lazyRouteComponent(() => import("./pages/models/ModelPage"), "ModelPage"),
});

const tasksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tasks",
  validateSearch: searchFor(SEARCH_KEYS.tasks),
  component: lazyRouteComponent(() => import("./pages/tasks/TasksPage"), "TasksPage"),
});

const taskRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tasks/$taskName",
  validateSearch: searchFor(SEARCH_KEYS.task),
  component: lazyRouteComponent(() => import("./pages/tasks/LeaderboardPage"), "TaskPage"),
});

const suiteRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/suites/$suiteName",
  validateSearch: searchFor(SEARCH_KEYS.suite),
  component: lazyRouteComponent(() => import("./pages/tasks/LeaderboardPage"), "SuitePage"),
});

const groupsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/groups",
  validateSearch: searchFor(SEARCH_KEYS.groups),
  component: lazyRouteComponent(() => import("./pages/groups/GroupsPage"), "GroupsPage"),
});

const groupRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/groups/$groupName",
  validateSearch: searchFor(SEARCH_KEYS.group),
  component: lazyRouteComponent(() => import("./pages/groups/GroupPage"), "GroupPage"),
});

const viewsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/views",
  component: lazyRouteComponent(() => import("./pages/views/ViewsPage"), "ViewsPage"),
});

export const routeTree = rootRoute.addChildren([
  homeRoute,
  runsRoute,
  runRoute,
  runTaskRoute,
  compareRoute,
  modelsRoute,
  modelRoute,
  tasksRoute,
  taskRoute,
  suiteRoute,
  groupsRoute,
  groupRoute,
  viewsRoute,
]);

export function createAppRouter(options: { history?: Parameters<typeof createRouter>[0]["history"] } = {}) {
  return createRouter({
    routeTree,
    history: options.history,
    parseSearch,
    stringifySearch,
    defaultPreload: "intent",
    defaultPreloadDelay: 120,
    scrollRestoration: true,
  });
}

export type AppRouter = ReturnType<typeof createAppRouter>;

declare module "@tanstack/react-router" {
  interface Register {
    router: AppRouter;
  }
}
