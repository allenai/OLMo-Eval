import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const contractDir = fileURLToPath(new URL("../contract", import.meta.url));
const srcDir = fileURLToPath(new URL("./src", import.meta.url));

export default defineConfig({
  plugins: [
    react(),
    {
      // The MSW worker in public/ is for development only; keep it out of production builds.
      name: "drop-mock-worker",
      apply: "build",
      closeBundle() {
        rmSync(fileURLToPath(new URL("./dist/mockServiceWorker.js", import.meta.url)), { force: true });
      },
    },
  ],
  resolve: {
    alias: {
      "@contract": contractDir,
      "@": srcDir,
    },
  },
  optimizeDeps: {
    // Pre-bundle everything up front so the dev server never re-optimizes mid-session.
    include: [
      "react",
      "react-dom",
      "react-dom/client",
      "@tanstack/react-query",
      "@tanstack/react-router",
      "@tanstack/react-virtual",
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-popover",
      "@radix-ui/react-tooltip",
      "cmdk",
      "tinykeys",
      "diff",
      "d3-scale",
      "d3-shape",
      "lucide-react",
      "react-markdown",
      "remark-gfm",
      "highlight.js/lib/core",
      "msw",
      "msw/browser",
    ],
  },
  build: {
    chunkSizeWarningLimit: 900,
  },
  server: {
    fs: { allow: [".."] },
    proxy: {
      // The API serves its routes under /api, so the path passes through unchanged.
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./test/setup.ts"],
    css: { modules: { classNameStrategy: "non-scoped" } },
    include: ["src/**/*.test.{ts,tsx}"],
    testTimeout: 20000,
  },
});
