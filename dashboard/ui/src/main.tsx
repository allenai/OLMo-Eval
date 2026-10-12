import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/manrope/200.css";
import "@fontsource/manrope/400.css";
import "@fontsource/manrope/800.css";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/base.css";
import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createQueryClient } from "./queryClient";
import { createAppRouter } from "./router";
import { applyDensity, applyTheme, densityStore, themeStore } from "./state/prefs";

/**
 * Decide whether to use the mock API. Production builds always use the real /api.
 * In development, VITE_MOCK_API=1 forces the mock, VITE_MOCK_API=0 forces the real API, and
 * otherwise the mock starts when no local API answers /api/health.
 */
async function shouldMock(): Promise<boolean> {
  if (!import.meta.env.DEV) return false;
  const flag = import.meta.env.VITE_MOCK_API;
  if (flag === "1") return true;
  if (flag === "0") return false;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 800);
    const res = await fetch("/api/health", { signal: controller.signal });
    clearTimeout(timer);
    const type = res.headers.get("content-type") ?? "";
    return !(res.ok && type.includes("json"));
  } catch {
    return true;
  }
}

async function main() {
  applyTheme(themeStore.get());
  applyDensity(densityStore.get());
  // The DEV check is a compile-time constant, so production builds drop the mock entirely.
  if (import.meta.env.DEV && (await shouldMock())) {
    const { startMockWorker } = await import("./mocks/browser");
    await startMockWorker();
    window.__OE_MOCK__ = true;
  }
  const queryClient = createQueryClient();
  const router = createAppRouter();
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>
    </StrictMode>,
  );
}

void main();
