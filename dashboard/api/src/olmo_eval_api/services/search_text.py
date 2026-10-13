"""The lowercase text that run search matches (``runs.search_text``).

Ingest builds it in Python from the ORM object; the dashboard rebuilds it in SQL after tag
edits. tests/test_search_text.py checks that the two agree.
"""

from __future__ import annotations

from olmo_eval_api.db.models import Run

SEARCH_TEXT_SQL = (
    "lower(concat_ws(' ', nullif(r.experiment_name, ''), nullif(r.experiment_group, ''), "
    "nullif(r.model_name, ''), nullif(array_to_string(r.tags, ' '), ''), nullif(r.author, ''), "
    "nullif(r.run_id, '')))"
)


def build_search_text(run: Run) -> str:
    parts = [
        run.experiment_name,
        run.experiment_group,
        run.model_name,
        " ".join(run.tags or []),
        run.author,
        run.run_id,
    ]
    return " ".join(p for p in parts if p).lower()
