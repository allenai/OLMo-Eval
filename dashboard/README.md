# olmo-eval dashboard

Stores olmo-eval results in Postgres and shows them in a web dashboard. It runs on
[Skiff2](https://github.com/allenai/skiff-commodore) in the GCP project `ai2-skiff2-olmo-eval`.

## Architecture

- `api/`: FastAPI app (Python package `olmo-eval-api`, import name `olmo_eval_api`). It is
  independent of olmo-eval: neither package imports the other. SQLModel tables, alembic
  migrations (run at startup), Cloud SQL Postgres through the Cloud SQL Python Connector with IAM
  database auth.
- `ui/`: React + Vite + TypeScript single-page app, served by nginx.
- `contract/`: TypeScript types and generated JSON Schemas for the ingest and read APIs. The UI,
  the API, and olmo-eval all test against them.
- `../infra/terraform/`: Cloud SQL, the results and build-assets buckets, the runtime service
  account and IAM.

Two Cloud Run services, declared in `/skiff2.json`:

| Service | Contents | Access |
|---|---|---|
| `ui` | nginx + SPA, with the API as an `api` sidecar in `dashboard` mode. nginx proxies `/api/*` to it. | IAP, `@allenai.org` only |
| `ingest` | Same API image built with `API_MODE=ingest`. Receives uploads from olmo-eval. | Open at IAP; the app verifies a Google token on every request |

Run files (predictions, requests, metrics, logs) live in
`gs://ai2-skiff2-olmo-eval-results/[dev/]runs/<run_id>/`. Postgres stores the queryable data and
the `gs://` URIs, plus Beaker IDs so the dashboard can link to beaker.org. Production uses the
database `olmo_eval`; every other environment uses `olmo_eval_dev` and the `dev/` prefix, which
expires after 30 days.

URLs:

| Environment | UI | Ingest |
|---|---|---|
| prod (`main`) | https://olmo-eval.allen.ai | https://prod-ingest.olmo-eval.apps.allenai.org |
| branch `<branch>` | `https://<env>-ui.olmo-eval.apps.allenai.org` | `https://<env>-ingest.olmo-eval.apps.allenai.org` |

`<env>` is the branch name with `/` replaced by `-`.

## How uploads authenticate

olmo-eval gets a Google OAuth access token from Application Default Credentials (scopes `openid`,
`userinfo.email`, `cloud-platform`) and sends it in the `X-Olmo-Eval-Token` header. The ingest
service checks it with Google's tokeninfo endpoint and caches the result. It accepts verified
`@allenai.org` users and allowlisted service accounts (by default only
`olmo-eval-uploader@ai2-skiff2-olmo-eval.iam.gserviceaccount.com`). The verified email is
recorded as the run's uploader.

- On a laptop: `gcloud auth application-default login`.
- On Beaker: jobs upload as `olmo-eval-uploader`, a service account with no Google Cloud roles
  (`infra/terraform/uploader.tf`). `olmo-eval beaker launch` reads its key from the Secret
  Manager secret `olmo-eval-uploader-key` with the launching user's local credentials and copies
  it into the workspace's `olmo_eval_uploader_key` Beaker secret, which the job reads as
  `OLMO_EVAL_UPLOAD_CREDENTIALS`. The CLI moves the key out of the environment as it starts, so
  inference servers, sandboxes and other subprocesses never see it. Personal credentials never
  reach Beaker. The run's `author`
  is the launching Beaker user, and only that user (or a direct uploader) can delete it;
  service accounts cannot delete runs. A re-upload keeps the stored author unless it comes from
  the user who uploaded the run.

Only the same people can change an existing run: its uploader, or for a service-account upload,
its author. Anyone else, including other Beaker jobs, must send the run's write secret in the
`X-Olmo-Eval-Run-Secret` header. olmo-eval creates the secret in the results directory
(`.upload-secret`, never uploaded) the first time it uploads, so re-uploading from that directory
works for anyone who has a copy of it.

Large files go straight to GCS: the ingest service returns V4 signed upload URLs, signed by the
ingest service account through IAM `signBlob`. Each URL is signed for the declared MD5 and size
(`x-goog-content-length-range`), so GCS rejects any other body. A run may hold at most 50 GiB of
artifacts and 10 million instance rows. To re-upload a local results directory, run
`olmo-eval results upload <dir>`.

To call the ingest API by hand:

```bash
curl -H "X-Olmo-Eval-Token: $(gcloud auth application-default print-access-token)" \
  https://prod-ingest.olmo-eval.apps.allenai.org/v1/whoami
```

## Local development

Requires Docker, uv and Node 24.

```bash
# API in "all" mode (ingest + read routes) on :8000, with Postgres and local file storage
docker compose -f dashboard/docker-compose.yml up

# UI against that API (Vite proxies /api to :8000)
cd dashboard/ui && npm install && npm run dev

# UI with an in-browser mock API (no backend needed)
cd dashboard/ui && npm run dev:mock
```

Send a local results directory to the local API:

```bash
OLMO_EVAL_API_URL=http://localhost:8000 uv run olmo-eval results upload <dir>
```

Tests and checks:

```bash
docker compose -f dashboard/docker-compose.yml up -d db   # Postgres on :5433
cd dashboard/api
export TEST_DATABASE_URL=postgresql+asyncpg://olmo@localhost:5433/olmo_eval_test
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest

cd dashboard/ui
npm run lint && npm run typecheck && npm run test && npm run contract:check
```

Without `TEST_DATABASE_URL`, the database tests are skipped. After editing `contract/*.ts`, run
`npm run contract:generate` in `ui/` to regenerate the JSON Schemas.

Without the Telegraf font files in `ui/public/fonts/`, the UI falls back to Manrope. Telegraf is a
licensed font and this repo is public, so never commit the `.otf` files (they are gitignored).

## Deploys

`.github/workflows/skiff2-deploy.yml` runs on pushes to `main` that touch `dashboard/`,
`skiff2.json` or the workflow. It authenticates to GCP through Workload Identity Federation,
downloads the Telegraf fonts from `gs://ai2-skiff2-olmo-eval-build-assets/fonts/`, builds one
image per container (`ui`, `api`, `ingest`), and deploys them with Skiff2's Terraform. The `ui`
service and its `api` sidecar run as `olmo-eval-dashboard`, which can read the database and the
results bucket and write only saved views, run tags and notes, and the stats cache. The `ingest`
service runs as `olmo-eval-api`, which owns the tables. Ingest runs migrations at startup under a
Postgres advisory lock, and the startup probe holds traffic until they finish. A final smoke test fails the deploy unless the prod ingest service's `/health`
reports the merged commit with a working database, and the dashboard redirects to Google sign-in.

Pull requests get a Skiff2 Terraform plan (`skiff2-plan.yml`; skipped for fork PRs, which cannot
authenticate to GCP) and the dashboard checks (`dashboard-ci.yml`: API and UI tests, plus Docker
builds of both images).

Containers need no secrets. Configuration comes from code defaults keyed on `SKIFF_ENV`, which
Cloud Run sets on every container. To allow another service account to upload, create the Secret
Manager secret `global-ingest-INGEST_ALLOWED_SERVICE_ACCOUNTS` (comma-separated emails, which
replaces the default list, so include the uploader account), grant
`olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com` the
`roles/secretmanager.secretAccessor` role on that secret only, and redeploy. Neither runtime
service account has project-wide secret access.

## Terraform

`infra/terraform` manages the resources the Skiff2 project does not create. State is in
`gs://ai2-skiff2-olmo-eval-tf-state` under the prefix `terraform/olmo-eval`. Skiff2 owns the
`terraform/infra` and `terraform/services` prefixes and the resources in them (load balancer,
certificates, Cloud Armor, the `github-actions` service account, Workload Identity Federation).

`.github/workflows/terraform.yml` deploys it continuously: pull requests that touch
`infra/terraform` get a `terraform plan` against the real project in the job summary, and merging
to `main` runs `terraform apply`. Both run as Skiff2's `github-actions` service account. To apply
by hand instead:

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

This needs owner-level access to the project. Apply Terraform before the first deploy, then run
`infra/terraform/scripts/grant_db.py` once: the services run as the `olmo-eval-api` and
`olmo-eval-dashboard` service accounts and connect to the database at startup. The Cloud SQL
instance has deletion protection on. See `infra/terraform/README.md` for details.
