import { useRouter, useRouterState } from "@tanstack/react-router";
import { type AnchorHTMLAttributes, forwardRef, type MouseEvent, useCallback } from "react";
import { stringifySearch } from "@/lib/url";

/** Build an app URL from a path and search params (empty values dropped). */
export function buildHref(path: string, search?: Record<string, string | number | undefined | null>): string {
  return `${path}${search ? stringifySearch(search as Record<string, unknown>) : ""}`;
}

function withBaseline(href: string, baseline: string | undefined): string {
  if (!baseline || href.startsWith("http") || /[?&]baseline=/.test(href)) return href;
  const [path, hash] = href.split("#");
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}baseline=${baseline}${hash ? `#${hash}` : ""}`;
}

/** Navigate to an app href, keeping the global baseline. */
export function useAppNavigate() {
  const router = useRouter();
  const search = useRouterState({ select: (st) => st.location.search }) as { baseline?: string };
  return useCallback(
    (href: string, options: { replace?: boolean } = {}) => {
      void router.navigate({ href: withBaseline(href, search.baseline), replace: options.replace });
    },
    [router, search.baseline],
  );
}

type Props = AnchorHTMLAttributes<HTMLAnchorElement> & { href: string; keepBaseline?: boolean };

/** Anchor for in-app navigation: real href (open in new tab works), client-side routing on click. */
export const AppLink = forwardRef<HTMLAnchorElement, Props>(function AppLink(
  { href, keepBaseline = true, onClick, ...rest },
  ref,
) {
  const router = useRouter();
  const search = useRouterState({ select: (st) => st.location.search }) as { baseline?: string };
  const target = keepBaseline ? withBaseline(href, search.baseline) : href;
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (rest.target === "_blank") return;
    e.preventDefault();
    void router.navigate({ href: target });
  };
  return <a ref={ref} href={target} onClick={handle} {...rest} />;
});
