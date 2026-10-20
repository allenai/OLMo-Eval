# Dashboard UI

React + Vite + TypeScript single-page app for exploring olmo-eval results. It talks to the
dashboard API at `/api` (served by `dashboard/api`; nginx proxies `/api/*` to the API sidecar).
Types come from `../contract/api-types.ts`.

## Commands

```bash
npm ci
npm run dev          # mock API when nothing answers /api/health on :8000, else the real API
npm run dev:mock     # always use the mock API (synthetic data, no backend needed)
VITE_MOCK_API=0 npm run dev   # always use the real API (vite proxies /api to localhost:8000)
npm run lint
npm run typecheck
npm run test         # vitest: unit, component, page and mock-vs-contract tests
npm run build
npm run contract:check     # schemas up to date and examples valid
npm run contract:generate  # regenerate ../contract/*.schema.json after editing the .ts files
```

Production builds always use the real API; the mock code is not included.

## Fonts

Telegraf is a licensed font and this repository is public, so the `.otf` files are never
committed. CI downloads them into `public/fonts/` (gitignored) before the Docker build. Without
them the UI falls back to Manrope (bundled through `@fontsource`). For local work you can copy
`Telegraf-200.otf`, `Telegraf-400.otf` and `Telegraf-800.otf` into `public/fonts/`.

## Layout

| Path | Contents |
|---|---|
| `src/router.tsx` | Routes and the search params each route accepts |
| `src/api/` | Fetch client and one React Query hook per endpoint |
| `src/state/` | URL state helpers, baseline, compare tray, subject colors, preferences |
| `src/styles/` | Design tokens (light and dark), fonts, base styles |
| `src/components/` | Design system: primitives, DataTable, filter bar, JSON tree and diff, app shell |
| `src/charts/` | Chart components (heatmap, small multiples, histograms, timelines, ...) |
| `src/pages/` | One folder per route |
| `src/mocks/` | Deterministic synthetic dataset and MSW handlers for every endpoint |

Every view keeps its state in the URL query string, so links reproduce what you see. The global
baseline (`baseline=`) is carried across navigation.
