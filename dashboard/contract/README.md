# Dashboard API contract

Shared types for the two HTTP interfaces of the dashboard backend:

| Source of truth | Generated schema | Used by |
|---|---|---|
| `ingest-types.ts` | `ingest-v1.schema.json` | olmo-eval upload client (`src/olmo_eval/upload/`) and the ingest routes in `dashboard/api` |
| `api-types.ts` | `api-v1.schema.json` | the UI (imports the types directly) and the dashboard routes in `dashboard/api` |

Edit the `.ts` files, then regenerate the schemas:

```bash
npm --prefix dashboard/ui run contract:generate
```

CI runs `npm --prefix dashboard/ui run contract:check`, which fails when a schema is out of
date or an example does not validate.

`examples/` holds example payloads. `examples/index.json` maps each file to the schema
definition it must validate against. The API tests, the olmo-eval client tests and the UI
mock server all use these files.
