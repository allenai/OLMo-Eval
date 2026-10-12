"""Reading the uploader key from Secret Manager and syncing it to Beaker."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olmo_eval.launch.beaker.uploader import (
    UPLOADER_SECRET_NAME,
    UploaderKeyError,
    ensure_uploader_secret,
    fetch_uploader_key,
)


def _session_returning(status: int, body: dict | None = None, text: str = "") -> MagicMock:
    response = SimpleNamespace(status_code=status, json=lambda: body, text=text)
    session = MagicMock()
    session.get.return_value = response
    return session


def _fetch(session: MagicMock) -> str:
    creds = MagicMock()
    with (
        patch("google.auth.default", return_value=(creds, None)),
        patch(
            "google.auth.transport.requests.AuthorizedSession", return_value=session
        ) as authorized_session,
    ):
        key = fetch_uploader_key()
    # Secret Manager bills the secret's project, so the caller's quota project is left alone.
    authorized_session.assert_called_once_with(creds)
    creds.with_quota_project.assert_not_called()
    return key


def test_fetch_decodes_the_latest_version() -> None:
    data = base64.b64encode(b'{"type": "service_account"}').decode()
    session = _session_returning(200, {"payload": {"data": data}})
    assert _fetch(session) == '{"type": "service_account"}'
    assert session.get.call_args.args[0].endswith(
        "/secrets/olmo-eval-uploader-key/versions/latest:access"
    )


def test_fetch_denied_names_the_secret() -> None:
    with pytest.raises(UploaderKeyError, match="olmo-eval-uploader-key.*HTTP 403"):
        _fetch(_session_returning(403, text="denied"))


@pytest.mark.parametrize(("current", "writes"), [(None, 1), ("old-key", 1), ("new-key", 0)])
def test_secret_is_written_only_when_the_key_changed(current: str | None, writes: int) -> None:
    client = MagicMock()
    if current is None:
        client.secret.get.side_effect = Exception("not found")
    else:
        client.secret.read.return_value = current
    with patch("beaker.Beaker.from_env", return_value=client):
        assert ensure_uploader_secret("ai2/ws", key="new-key") == UPLOADER_SECRET_NAME
    assert client.secret.write.call_count == writes
    if writes:
        client.secret.write.assert_called_with(UPLOADER_SECRET_NAME, "new-key")
