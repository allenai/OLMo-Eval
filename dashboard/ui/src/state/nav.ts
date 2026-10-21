import type { SubjectKey } from "@contract/api-types";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useCallback } from "react";
import { isSubjectKey } from "@/lib/subjects";
import { lastBaselineStore } from "./prefs";
import { createStore } from "./store";

export type SearchState = Record<string, string | undefined>;

function clean(search: Record<string, unknown>): SearchState {
  const out: SearchState = {};
  for (const [k, v] of Object.entries(search)) {
    if (v === undefined || v === null || v === "") out[k] = undefined;
    else out[k] = String(v);
  }
  return out;
}

/**
 * Current route's search params as plain strings, plus a setter that merges a patch.
 * `replace` is for transient state (sort, focus); everything else pushes history.
 */
export function useSearchParams<T extends SearchState = SearchState>(): [
  T,
  (patch: Partial<T>, options?: { replace?: boolean }) => void,
] {
  const search = useRouterState({ select: (st) => st.location.search }) as unknown as T;
  const navigate = useNavigate();
  const set = useCallback(
    (patch: Partial<T>, options: { replace?: boolean } = {}) => {
      void navigate({
        to: ".",
        search: ((prev: Record<string, unknown>) => clean({ ...prev, ...patch })) as never,
        replace: options.replace ?? false,
        resetScroll: false,
      });
    },
    [navigate],
  );
  return [search, set];
}

/** The global baseline subject, kept in the URL (`baseline=`) across navigation. */
export function useBaseline(): [SubjectKey | undefined, (key: SubjectKey | null, label?: string) => void] {
  const [search, setSearch] = useSearchParams<{ baseline?: string }>();
  const value = search.baseline && isSubjectKey(search.baseline) ? search.baseline : undefined;
  const set = useCallback(
    (key: SubjectKey | null, label?: string) => {
      if (key && label) lastBaselineStore.set({ key, label });
      setSearch({ baseline: key ?? undefined });
    },
    [setSearch],
  );
  return [value, set];
}

/** The subject the current page is about (run page), used by the `b` shortcut. */
export const currentSubjectStore = createStore<{ key: SubjectKey; label: string } | null>(null);
