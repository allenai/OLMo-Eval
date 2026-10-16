"""Ingest token verification with a mocked tokeninfo endpoint, and dashboard identity."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest

from olmo_eval_api.auth import google_token
from olmo_eval_api.auth.google_token import TokenVerifier
from olmo_eval_api.errors import ApiError
from olmo_eval_api.log_config import configure_logging
from olmo_eval_api.settings import Settings


class FakeTokenInfo:
    def __init__(self) -> None:
        self.tokens: dict[str, tuple[int, dict[str, Any]]] = {}
        self.calls = 0
        self.fail = False

    def add(
        self,
        token: str,
        email: str | None,
        *,
        verified: bool = True,
        ttl: int = 3600,
        scope: str = "openid https://www.googleapis.com/auth/userinfo.email "
        "https://www.googleapis.com/auth/cloud-platform",
    ) -> None:
        body: dict[str, Any] = {
            "exp": str(int(time.time()) + ttl),
            "expires_in": str(ttl),
            "scope": scope,
        }
        if email is not None:
            body["email"] = email
            body["email_verified"] = "true" if verified else "false"
        self.tokens[token] = (200, body)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail:
            raise httpx.ConnectError("down")
        assert request.method == "POST" and "access_token" not in str(request.url)
        token = parse_qs(request.content.decode()).get("access_token", [""])[0]
        status, body = self.tokens.get(token, (400, {"error": "invalid_token"}))
        return httpx.Response(status, json=body)


def make_verifier(fake: FakeTokenInfo, **overrides: Any) -> TokenVerifier:
    values: dict[str, Any] = {
        "skiff_env": "prod",
        "ingest_allowed_service_accounts": "ci@proj.iam.gserviceaccount.com",
        **overrides,
    }
    settings = Settings(**values)
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return TokenVerifier(settings, client=client)


async def test_user_accepted_and_cached() -> None:
    fake = FakeTokenInfo()
    fake.add("good", "Alice@AllenAI.org")
    verifier = make_verifier(fake)
    principal = await verifier.verify("good")
    assert principal.email == "alice@allenai.org"
    assert principal.principal_type == "user"
    assert 3500 < principal.expires_in <= 3600
    await verifier.verify("good")
    assert fake.calls == 1


async def test_token_never_reaches_the_logs(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")
    try:
        fake = FakeTokenInfo()
        fake.add("ya29.secret-token-value", "alice@allenai.org")
        await make_verifier(fake).verify("ya29.secret-token-value")
        with pytest.raises(ApiError):
            await make_verifier(fake).verify("ya29.unknown-token-value")
    finally:
        configure_logging("INFO")
    out = capsys.readouterr()
    assert "secret-token-value" not in out.out + out.err
    assert "unknown-token-value" not in out.out + out.err


async def test_user_token_needs_cloud_platform_scope() -> None:
    # A token that a "Sign in with Google" site got with only the email scope must not work.
    fake = FakeTokenInfo()
    fake.add(
        "signin",
        "alice@allenai.org",
        scope="openid email https://www.googleapis.com/auth/userinfo.email",
    )
    fake.add(
        "sa",
        "ci@proj.iam.gserviceaccount.com",
        scope="https://www.googleapis.com/auth/userinfo.email",
    )
    verifier = make_verifier(fake)
    with pytest.raises(ApiError) as exc:
        await verifier.verify("signin")
    assert exc.value.status_code == 401
    assert "cloud-platform" in exc.value.message
    # Allowlisted service accounts are matched by email alone.
    assert (await verifier.verify("sa")).principal_type == "service_account"


async def test_service_account_allowlist() -> None:
    fake = FakeTokenInfo()
    fake.add("sa", "ci@proj.iam.gserviceaccount.com")
    fake.add("other-sa", "rogue@proj.iam.gserviceaccount.com")
    verifier = make_verifier(fake)
    principal = await verifier.verify("sa")
    assert principal.principal_type == "service_account"
    with pytest.raises(ApiError) as exc:
        await verifier.verify("other-sa")
    assert exc.value.status_code == 403


async def test_service_account_in_allowed_domain_is_not_a_user() -> None:
    fake = FakeTokenInfo()
    fake.add("sa", "bot@allenai.org.gserviceaccount.com")
    with pytest.raises(ApiError) as exc:
        await make_verifier(fake).verify("sa")
    assert exc.value.status_code == 403


async def test_wrong_domain_rejected_and_negative_cached() -> None:
    fake = FakeTokenInfo()
    fake.add("gmail", "someone@gmail.com")
    verifier = make_verifier(fake)
    for _ in range(2):
        with pytest.raises(ApiError) as exc:
            await verifier.verify("gmail")
        assert exc.value.status_code == 403
        assert "someone@gmail.com is not allowed" in exc.value.message
    assert fake.calls == 1


async def test_missing_email_scope() -> None:
    fake = FakeTokenInfo()
    fake.add("noscope", None)
    with pytest.raises(ApiError) as exc:
        await make_verifier(fake).verify("noscope")
    assert exc.value.status_code == 401
    assert "userinfo.email" in exc.value.message


async def test_unverified_email_rejected() -> None:
    fake = FakeTokenInfo()
    fake.add("unverified", "bob@allenai.org", verified=False)
    with pytest.raises(ApiError) as exc:
        await make_verifier(fake).verify("unverified")
    assert exc.value.status_code == 401


async def test_invalid_and_expired_tokens() -> None:
    fake = FakeTokenInfo()
    fake.add("expired", "bob@allenai.org", ttl=-10)
    verifier = make_verifier(fake)
    for token in ("nope", "expired"):
        with pytest.raises(ApiError) as exc:
            await verifier.verify(token)
        assert exc.value.status_code == 401
        assert exc.value.message == "invalid or expired Google access token"
        assert exc.value.headers == {"WWW-Authenticate": "Bearer"}


async def test_tokeninfo_unreachable() -> None:
    fake = FakeTokenInfo()
    fake.fail = True
    with pytest.raises(ApiError) as exc:
        await make_verifier(fake).verify("any")
    assert exc.value.status_code == 503


@pytest.mark.parametrize("status", [403, 429, 502])
async def test_tokeninfo_errors_are_not_cached_as_invalid(status: int) -> None:
    fake = FakeTokenInfo()
    fake.add("good", "alice@allenai.org")
    verifier = make_verifier(fake)
    fake.tokens["good"] = (status, {"error": "rate limited"})
    with pytest.raises(ApiError) as exc:
        await verifier.verify("good")
    assert exc.value.status_code == 503
    fake.add("good", "alice@allenai.org")
    assert (await verifier.verify("good")).email == "alice@allenai.org"


async def test_malformed_tokens_skip_tokeninfo() -> None:
    fake = FakeTokenInfo()
    verifier = make_verifier(fake)
    for token in ("x" * 5000, "has space", "ya29.\u00e9"):
        with pytest.raises(ApiError) as exc:
            await verifier.verify(token)
        assert exc.value.status_code == 401
    assert fake.calls == 0


async def test_rejections_cannot_evict_accepted_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(google_token, "CACHE_MAX", 3)
    fake = FakeTokenInfo()
    fake.add("good", "alice@allenai.org")
    verifier = make_verifier(fake)
    await verifier.verify("good")
    for i in range(10):
        with pytest.raises(ApiError):
            await verifier.verify(f"junk{i}")
    calls = fake.calls
    await verifier.verify("good")
    assert fake.calls == calls


async def test_dev_tokens_only_in_local() -> None:
    fake = FakeTokenInfo()
    prod = make_verifier(fake)
    with pytest.raises(ApiError):
        await prod.verify("dev:alice@allenai.org")
    local = make_verifier(fake, skiff_env="local", ingest_dev_auth=True)
    assert (await local.verify("dev:alice@allenai.org")).email == "alice@allenai.org"


async def test_ingest_endpoints_require_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/whoami", headers={"X-Olmo-Eval-Token": ""})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_bearer_header_accepted_and_custom_header_wins(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/v1/whoami",
        headers={"X-Olmo-Eval-Token": "", "Authorization": "Bearer dev:bob@allenai.org"},
    )
    assert response.json()["email"] == "bob@allenai.org"
    response = await client.get(
        "/v1/whoami",
        headers={
            "X-Olmo-Eval-Token": "dev:carol@allenai.org",
            "Authorization": "Bearer dev:bob@allenai.org",
        },
    )
    assert response.json()["email"] == "carol@allenai.org"


async def test_ingest_forbidden_domain(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/whoami", headers={"X-Olmo-Eval-Token": "dev:x@gmail.com"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


async def test_mocked_tokeninfo_through_app(app: Any, client: httpx.AsyncClient) -> None:
    fake = FakeTokenInfo()
    fake.add("real-token", "dana@allenai.org")
    app.state.token_verifier = make_verifier(fake, skiff_env="local")
    response = await client.get("/v1/whoami", headers={"X-Olmo-Eval-Token": "real-token"})
    assert response.status_code == 200
    assert response.json()["email"] == "dana@allenai.org"
    response = await client.get("/v1/whoami", headers={"X-Olmo-Eval-Token": "bogus"})
    assert response.status_code == 401


async def test_dashboard_identity(client: httpx.AsyncClient) -> None:
    body = (await client.get("/api/me")).json()
    assert body == {"email": "tester@allenai.org", "username": "tester", "dev_mode": False}
    body = (await client.get("/api/me", headers={"X-Goog-Authenticated-User-Email": ""})).json()
    assert body["dev_mode"] is True and body["email"] == "dev@allenai.org"


async def test_dashboard_csrf(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/views",
        json={"name": "x", "page": "runs", "query": "", "shared": False},
        headers={"X-Requested-With": ""},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
