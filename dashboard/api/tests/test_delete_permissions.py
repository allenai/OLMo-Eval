"""Who may delete a run or change its author, and the uploader account allowlist default."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from olmo_eval_api.auth.google_token import Principal
from olmo_eval_api.services.ingest import can_delete, may_change_author, run_time
from olmo_eval_api.settings import UPLOADER_SERVICE_ACCOUNT, Settings

ALICE = Principal(email="alice@allenai.org", principal_type="user", exp=0)
UPLOADER = Principal(email=UPLOADER_SERVICE_ACCOUNT, principal_type="service_account", exp=0)


@pytest.mark.parametrize(
    ("principal", "uploaded_by", "author", "allowed"),
    [
        (ALICE, "alice@allenai.org", "alice", True),
        (ALICE, "bob@allenai.org", "alice", False),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, "alice", True),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, "bob", False),
        (ALICE, UPLOADER_SERVICE_ACCOUNT, None, False),
        (UPLOADER, UPLOADER_SERVICE_ACCOUNT, "alice", False),
        (UPLOADER, "alice@allenai.org", "alice", False),
    ],
)
def test_can_delete(
    principal: Principal, uploaded_by: str, author: str | None, allowed: bool
) -> None:
    assert can_delete(principal, uploaded_by, author) is allowed


@pytest.mark.parametrize(
    ("principal", "uploaded_by", "allowed"),
    [
        (ALICE, "alice@allenai.org", True),
        (ALICE, "bob@allenai.org", False),
        # The author of a service-account upload is what can_delete trusts, so it is fixed.
        (ALICE, UPLOADER_SERVICE_ACCOUNT, False),
        (UPLOADER, UPLOADER_SERVICE_ACCOUNT, False),
    ],
)
def test_may_change_author(principal: Principal, uploaded_by: str, allowed: bool) -> None:
    assert may_change_author(principal, uploaded_by) is allowed


def test_uploader_is_allowlisted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INGEST_ALLOWED_SERVICE_ACCOUNTS", raising=False)
    assert Settings().ingest_allowed_service_accounts == [UPLOADER_SERVICE_ACCOUNT]


def test_run_time_is_the_start_unless_missing_or_in_the_future() -> None:
    received = datetime(2026, 10, 7, 12, tzinfo=UTC)
    started = received - timedelta(days=3)
    assert run_time(started, received) == started
    assert run_time(None, received) == received
    assert run_time(received + timedelta(hours=1), received) == received
    assert run_time(started.replace(tzinfo=None), received) == started
