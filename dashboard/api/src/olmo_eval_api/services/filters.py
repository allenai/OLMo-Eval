"""Runs filters shared by /api/runs and /api/runs/facets (spec 4.3).

Repeated values of one key are OR-ed; different keys are AND-ed. Globs use ``*``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Annotated, Any

from fastapi import Query

from olmo_eval_api.errors import bad_request
from olmo_eval_api.services.queries import VISIBLE

DATE_WINDOWS = {
    "24h": timedelta(days=1),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
}
STATUSES = {"running", "complete", "partial", "failed", "uploading"}


def glob_to_like(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped.replace("*", "%")


def like_contains(value: str) -> str:
    return f"%{glob_to_like(value)}%"


@dataclass
class RunFilters:
    model: list[str] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    family: list[str] = field(default_factory=list)
    task: list[str] = field(default_factory=list)
    suite: list[str] = field(default_factory=list)
    suite_coverage: str = "full"
    user: list[str] = field(default_factory=list)
    group: list[str] = field(default_factory=list)
    launch: list[str] = field(default_factory=list)
    workspace: list[str] = field(default_factory=list)
    tag: list[str] = field(default_factory=list)
    not_tag: list[str] = field(default_factory=list)
    status: list[str] = field(default_factory=list)
    after: datetime | None = None
    before: datetime | None = None
    date: str | None = None
    commit: list[str] = field(default_factory=list)
    step_min: int | None = None
    step_max: int | None = None
    has_failures: bool | None = None
    run_id: list[str] = field(default_factory=list)
    q: str | None = None


def run_filters(
    model: Annotated[list[str], Query()] = [],  # noqa: B006
    series: Annotated[list[str], Query()] = [],  # noqa: B006
    family: Annotated[list[str], Query()] = [],  # noqa: B006
    task: Annotated[list[str], Query()] = [],  # noqa: B006
    suite: Annotated[list[str], Query()] = [],  # noqa: B006
    suite_coverage: Annotated[str, Query(pattern="^(full|any)$")] = "full",
    user: Annotated[list[str], Query()] = [],  # noqa: B006
    group: Annotated[list[str], Query()] = [],  # noqa: B006
    launch: Annotated[list[str], Query()] = [],  # noqa: B006
    workspace: Annotated[list[str], Query()] = [],  # noqa: B006
    tag: Annotated[list[str], Query()] = [],  # noqa: B006
    not_tag: Annotated[list[str], Query()] = [],  # noqa: B006
    status: Annotated[list[str], Query()] = [],  # noqa: B006
    after: datetime | None = None,
    before: datetime | None = None,
    date: Annotated[str | None, Query(pattern="^(24h|7d|30d|90d)$")] = None,
    commit: Annotated[list[str], Query()] = [],  # noqa: B006
    step_min: int | None = None,
    step_max: int | None = None,
    has_failures: bool | None = None,
    run_id: Annotated[list[str], Query()] = [],  # noqa: B006
    q: str | None = None,
) -> RunFilters:
    bad = set(status) - STATUSES
    if bad:
        raise bad_request(f"unknown status: {', '.join(sorted(bad))}")
    return RunFilters(
        model=model,
        series=series,
        family=family,
        task=task,
        suite=suite,
        suite_coverage=suite_coverage,
        user=user,
        group=group,
        launch=launch,
        workspace=workspace,
        tag=tag,
        not_tag=not_tag,
        status=status,
        after=after,
        before=before,
        date=date,
        commit=commit,
        step_min=step_min,
        step_max=step_max,
        has_failures=has_failures,
        run_id=run_id,
        q=q,
    )


FACET_FILTER = {
    "user": "user",
    "family": "family",
    "group": "group",
    "workspace": "workspace",
    "tag": "tag",
    "status": "status",
    "model": "model",
}


def build_where(
    f: RunFilters, me: str, *, exclude: str | None = None
) -> tuple[str, dict[str, Any]]:
    """SQL condition over ``runs r JOIN models m`` and its parameters."""
    clauses = [VISIBLE]
    params: dict[str, Any] = {}
    n = 0

    def p(value: Any) -> str:
        nonlocal n
        n += 1
        params[f"f{n}"] = value
        return f":f{n}"

    def ors(items: list[str]) -> None:
        if items:
            clauses.append("(" + " OR ".join(items) + ")")

    def active(name: str) -> bool:
        return exclude != name and bool(getattr(f, name))

    if active("model"):
        ors([f"r.model_name ILIKE {p(glob_to_like(v))}" for v in f.model])
    if f.series:
        clauses.append(f"m.series = ANY({p(f.series)})")
    if active("family"):
        clauses.append(f"m.family = ANY({p(f.family)})")
    if f.task:
        ors(
            [
                "EXISTS (SELECT 1 FROM task_results t WHERE t.run_id = r.run_id "
                f"AND t.task_name ILIKE {p(glob_to_like(v))})"
                for v in f.task
            ]
        )
    if f.suite:
        items = []
        for v in f.suite:
            if f.suite_coverage == "any":
                items.append(
                    "EXISTS (SELECT 1 FROM task_results t WHERE t.run_id = r.run_id AND "
                    "t.task_name IN (SELECT jsonb_array_elements(d.children) ->> 'name' FROM "
                    "(SELECT children FROM suite_defs WHERE suite_name = "
                    f"{p(v)} ORDER BY last_seen_at DESC LIMIT 1) d))"
                )
            else:
                items.append(
                    "EXISTS (SELECT 1 FROM suite_results s WHERE s.run_id = r.run_id "
                    f"AND s.suite_name = {p(v)})"
                )
        ors(items)
    if active("user"):
        users = [me if v == "me" else v for v in f.user]
        ors(
            [
                f"r.author = ANY({p(users)})",
                f"split_part(r.uploaded_by, '@', 1) = ANY({p(users)})",
            ]
        )
    if active("group"):
        clauses.append(f"r.experiment_group = ANY({p(f.group)})")
    if f.launch:
        clauses.append(f"r.launch_id = ANY({p(f.launch)})")
    if active("workspace"):
        clauses.append(f"r.beaker_workspace = ANY({p(f.workspace)})")
    if active("tag"):
        ors([f"r.tags @> ARRAY[{p(v)}]::text[]" for v in f.tag])
    for v in f.not_tag:
        clauses.append(f"NOT (r.tags @> ARRAY[{p(v)}]::text[])")
    if active("status"):
        items = []
        plain = [s for s in f.status if s != "uploading"]
        if plain:
            items.append(f"(r.status = ANY({p(plain)}) AND r.upload_state = 'complete')")
            if "running" in plain:
                items.append("r.status = 'running'")
        if "uploading" in f.status:
            items.append("(r.status <> 'running' AND r.upload_state = 'uploading')")
        ors(items)
    if f.after is not None:
        clauses.append(f"r.created_at >= {p(f.after)}")
    if f.before is not None:
        clauses.append(f"r.created_at < {p(f.before)}")
    if f.date:
        clauses.append(f"r.created_at >= now() - CAST({p(DATE_WINDOWS[f.date])} AS interval)")
    if f.commit:
        ors([f"r.git_commit LIKE {p(glob_to_like(v.lower()) + '%')}" for v in f.commit])
    if f.step_min is not None:
        clauses.append(f"m.step >= {p(f.step_min)}")
    if f.step_max is not None:
        clauses.append(f"m.step <= {p(f.step_max)}")
    if f.has_failures is not None:
        cond = (
            "(r.num_failed_tasks > 0 OR EXISTS (SELECT 1 FROM task_results t "
            "WHERE t.run_id = r.run_id AND t.instances_failed > 0))"
        )
        clauses.append(cond if f.has_failures else f"NOT {cond}")
    if f.run_id:
        clauses.append(f"r.run_id = ANY({p(f.run_id)})")
    if f.q:
        for token in f.q.lower().split():
            clauses.append(f"r.search_text LIKE {p(like_contains(token))}")
    if exclude != "status" and "uploading" in f.status:
        # Uploading runs are hidden by default; asking for them makes them visible.
        clauses[0] = f"({VISIBLE} OR r.upload_state = 'uploading')"
    return " AND ".join(clauses), params
