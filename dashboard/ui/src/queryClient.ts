import { MutationCache, QueryClient } from "@tanstack/react-query";
import { ApiError } from "./api/client";
import { toast } from "./components/toast";

export function createQueryClient(): QueryClient {
  return new QueryClient({
    // Every write reports its failure; reads show errors inline in their panels.
    mutationCache: new MutationCache({
      onError: (error) => {
        const id = error instanceof ApiError && error.requestId ? ` (request ${error.requestId})` : "";
        toast(`${error.message}${id}`, { tone: "error", duration: 6000 });
      },
    }),
    defaultOptions: {
      queries: {
        staleTime: 30_000,
        gcTime: 10 * 60_000,
        refetchOnWindowFocus: false,
        retry: (count, error) => {
          if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
          return count < 2;
        },
      },
    },
  });
}
