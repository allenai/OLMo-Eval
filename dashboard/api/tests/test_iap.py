"""IAP JWT verification in dashboard mode outside local development."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from google.auth import crypt, jwt

from olmo_eval_api.auth.iap import IAP_ISSUER, IapVerifier
from olmo_eval_api.main import create_app
from olmo_eval_api.settings import Settings

AUDIENCE = "/projects/354333262681/locations/us-west1/services/ui"


class Keys:
    def __init__(self) -> None:
        self.private: dict[str, Any] = {}
        self.fetches = 0

    def add(self, kid: str) -> None:
        self.private[kid] = ec.generate_private_key(ec.SECP256R1())

    async def fetch(self) -> dict[str, str]:
        self.fetches += 1
        return {
            kid: key.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode()
            for kid, key in self.private.items()
        }

    def token(self, kid: str = "k1", **claims: Any) -> str:
        pem = self.private[kid].private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        now = int(time.time())
        payload = {
            "iss": IAP_ISSUER,
            "aud": AUDIENCE,
            "sub": "accounts.google.com:123",
            "email": "pat@allenai.org",
            "iat": now,
            "exp": now + 600,
            **claims,
        }
        signer = crypt.ES256Signer.from_string(pem, key_id=kid)
        return jwt.encode(signer, payload).decode()


@pytest.fixture
def keys() -> Keys:
    k = Keys()
    k.add("k1")
    return k


@pytest.fixture
async def dash(keys: Keys) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        api_mode="dashboard",
        skiff_env="prod",
        k_service="ui",
        db_url="postgresql+asyncpg://nobody@127.0.0.1:1/none",  # /api/me never connects
        storage_backend="local",
    )
    assert settings.verify_iap and settings.iap_audience == AUDIENCE
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.iap_verifier = IapVerifier(settings.iap_audience, fetch_certs=keys.fetch)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def jwt_header(token: str) -> dict[str, str]:
    return {"X-Goog-IAP-JWT-Assertion": token}


async def test_valid_assertion(dash: httpx.AsyncClient, keys: Keys) -> None:
    response = await dash.get("/api/me", headers=jwt_header(keys.token(email="Pat@AllenAI.org")))
    assert response.status_code == 200, response.text
    assert response.json() == {"email": "pat@allenai.org", "username": "pat", "dev_mode": False}


async def test_plain_email_header_is_not_trusted(dash: httpx.AsyncClient) -> None:
    headers = {"X-Goog-Authenticated-User-Email": "accounts.google.com:pat@allenai.org"}
    assert (await dash.get("/api/me", headers=headers)).status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "/projects/354333262681/locations/us-west1/services/other"},
        {"iss": "https://accounts.google.com"},
        {"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
        {"email": None},
    ],
)
async def test_invalid_assertions(dash: httpx.AsyncClient, keys: Keys, claims: Any) -> None:
    response = await dash.get("/api/me", headers=jwt_header(keys.token(**claims)))
    assert response.status_code == 401


async def test_forged_signature(dash: httpx.AsyncClient, keys: Keys) -> None:
    other = Keys()
    other.add("k1")  # same key id, different key
    assert (await dash.get("/api/me", headers=jwt_header(other.token()))).status_code == 401


async def test_other_domains_are_refused(dash: httpx.AsyncClient, keys: Keys) -> None:
    response = await dash.get("/api/me", headers=jwt_header(keys.token(email="eve@gmail.com")))
    assert response.status_code == 403


async def test_rotated_key_is_fetched(
    dash: httpx.AsyncClient, keys: Keys, monkeypatch: Any
) -> None:
    from olmo_eval_api.auth import iap

    assert (await dash.get("/api/me", headers=jwt_header(keys.token()))).status_code == 200
    assert keys.fetches == 1
    monkeypatch.setattr(iap, "CERTS_MIN_REFRESH_S", 0.0)
    keys.add("k2")
    assert (await dash.get("/api/me", headers=jwt_header(keys.token("k2")))).status_code == 200
    assert keys.fetches == 2
