"""The Python and SQL builders of runs.search_text agree."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from sqlalchemy import text

from olmo_eval_api.services.search_text import SEARCH_TEXT_SQL
from tests.factories import run_payload


@pytest.mark.parametrize(
    ("group", "author", "tags"),
    [
        ("Group-A", "Alice", ["Tag1", "x:y"]),
        (None, "bob", []),
        ("", "", []),  # empty strings are skipped like nulls in both builders
    ],
)
async def test_python_and_sql_search_text_agree(
    client: httpx.AsyncClient, session: Any, group: str | None, author: str, tags: list[str]
) -> None:
    run_id = "search000001"
    body = run_payload(run_id=run_id, model_name="Org/Model-7B", group=group, tags=tags)
    body["run"]["author"] = author
    assert (await client.put(f"/v1/runs/{run_id}", json=body)).status_code == 200
    python_text, sql_text = (
        await session.execute(
            text(f"SELECT r.search_text, {SEARCH_TEXT_SQL} FROM runs r WHERE r.run_id = :r"),
            {"r": run_id},
        )
    ).one()
    assert python_text == sql_text
    assert "org/model-7b" in python_text
