# olmo-eval dashboard infrastructure

Terraform for the GCP resources the dashboard needs in the Skiff2 project `ai2-skiff2-olmo-eval`
(region `us-west1`). Skiff2 manages the load balancer, Cloud Run services, IAP, certificates,
Cloud Armor, the `github-actions` service account and Workload Identity Federation. This
configuration does not touch any of those.

State lives in the Skiff2 state bucket `ai2-skiff2-olmo-eval-tf-state` under the prefix
`terraform/olmo-eval`. Skiff2 uses `terraform/infra` and `terraform/services`.

## Resources

| Resource | Name |
|---|---|
| Cloud SQL Postgres 17 instance | `olmo-eval-db` (connection name `ai2-skiff2-olmo-eval:us-west1:olmo-eval-db`) |
| Databases | `olmo_eval` (prod Skiff environment), `olmo_eval_dev` (all other environments) |
| Ingest runtime service account | `olmo-eval-api@ai2-skiff2-olmo-eval.iam.gserviceaccount.com` (`ingest` service) |
| Ingest IAM database user | `olmo-eval-api@ai2-skiff2-olmo-eval.iam` (owns the tables) |
| Dashboard runtime service account | `olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam.gserviceaccount.com` (`ui` service and its `api` sidecar) |
| Dashboard IAM database user | `olmo-eval-dashboard@ai2-skiff2-olmo-eval.iam` |
| Uploader service account | `olmo-eval-uploader@ai2-skiff2-olmo-eval.iam.gserviceaccount.com` (Beaker jobs; key in Secret Manager `olmo-eval-uploader-key`) |
| Results bucket | `gs://ai2-skiff2-olmo-eval-results` (`runs/` for prod, `dev/runs/` for other environments, deleted after 30 days) |
| Build assets bucket | `gs://ai2-skiff2-olmo-eval-build-assets` (Telegraf fonts under `fonts/`) |

The instance has a public IP with no authorized networks and requires the Cloud SQL connectors
(`connector_enforcement = "REQUIRED"`). Clients connect with the Cloud SQL Python Connector and
IAM database authentication. There are no database passwords. The built-in `postgres` user
that Cloud SQL creates has no password set and cannot log in. Deletion protection, automated
backups and point-in-time recovery are on.

## Access

| Account | Project roles | Results bucket | Database |
|---|---|---|---|
| `olmo-eval-api` (ingest) | `cloudsql.client`, `cloudsql.instanceUser`, `logging.logWriter` | `storage.objectUser` | Owns the tables; `CREATE` on schema `public` |
| `olmo-eval-dashboard` | `cloudsql.client`, `cloudsql.instanceUser`, `logging.logWriter` | `storage.objectViewer` | `SELECT` on every table, plus the writes in `scripts/grant_db.py` |

Each account has `iam.serviceAccountTokenCreator` on itself, to sign V4 URLs through IAM
`signBlob` (ingest signs uploads, the dashboard signs downloads). Neither has Secret Manager
access. `var.debug_iam_users` can impersonate both and are `cloudsqlsuperuser` members.

## Database privileges

Cloud SQL makes `cloudsqlsuperuser` the owner of both databases, so schema `public` belongs to its
members. The runtime users are not members (`database_roles = []` for the ingest user), so the
privileges they need come from grants. `scripts/grant_db.py` applies them to both databases. It is
idempotent; run it after creating the instance or a database, or after changing a runtime user:

```bash
uv run infra/terraform/scripts/grant_db.py --user you@allenai.org
```

It grants the ingest user `CREATE` on schema `public` (as you), then, as the ingest user, grants
the dashboard user `SELECT` on every table, default `SELECT` on tables created later, and its
writes (`saved_views`, `stats_cache`, and some columns of `runs`). The ingest service re-applies
the dashboard grants after each migration, so new tables need no manual step.

Check the privileges. By default the script impersonates the ingest service account:

```bash
uv run infra/terraform/scripts/check_db.py
uv run infra/terraform/scripts/check_db.py --as dashboard
uv run infra/terraform/scripts/check_db.py --as-self --user you@allenai.org
```

## Applying changes

```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

`.github/workflows/terraform.yml` plans pull requests that touch this directory (the plan is in
the job summary) and applies on merge to `main`. Applying by hand needs owner (or equivalent) on
`ai2-skiff2-olmo-eval`. Changes to `uploader.tf`'s key rotation only take effect on the next apply,
so a rotation is due on the first apply after 90 days. Commit `.terraform.lock.hcl` when the provider version changes, and
lock for every platform CI and developers use:

```bash
terraform providers lock -platform=linux_amd64 -platform=linux_arm64 \
  -platform=darwin_amd64 -platform=darwin_arm64
```

## Fonts

The UI uses the commercial Telegraf font, which must never be committed to this public repo.
Upload new font files to the build assets bucket:

```bash
gcloud storage cp Telegraf-*.otf gs://ai2-skiff2-olmo-eval-build-assets/fonts/
```

The deploy workflow downloads them into `dashboard/ui/public/fonts/` before building the UI image.
