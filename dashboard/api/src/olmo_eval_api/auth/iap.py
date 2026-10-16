"""Dashboard identity from IAP (spec 4.1).

IAP authenticates every request to the ui service and adds two headers: the plain
``X-Goog-Authenticated-User-Email: accounts.google.com:user@allenai.org`` and the signed
``X-Goog-IAP-JWT-Assertion``. Outside local development (``Settings.verify_iap``) the dashboard
trusts only the signed one: it checks the ES256 signature against IAP's public keys, the issuer,
the audience (``/projects/<number>/locations/<region>/services/<K_SERVICE>`` for IAP on Cloud
Run, see https://cloud.google.com/iap/docs/signed-headers-howto) and the expiry, and then
requires an allowed email domain. That keeps a request that bypasses IAP (for example a
misconfigured ingress) from choosing its identity with a header.

In local development the plain header is read if present, else the dev user is used.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import orjson
from fastapi import Request
from google.auth import exceptions as gexc
from google.auth import jwt

from olmo_eval_api.errors import ApiError
from olmo_eval_api.log_config import principal_var
from olmo_eval_api.settings import Settings

logger = logging.getLogger(__name__)

IAP_EMAIL_HEADER = "x-goog-authenticated-user-email"
IAP_JWT_HEADER = "x-goog-iap-jwt-assertion"
IAP_ISSUER = "https://cloud.google.com/iap"
IAP_CERTS_URL = "https://www.gstatic.com/iap/verify/public_key"
CERTS_TTL_S = 3600.0
# An unknown key ID refetches the keys (IAP rotates them), at most this often.
CERTS_MIN_REFRESH_S = 60.0
CLOCK_SKEW_S = 30

CertsFetcher = Callable[[], Awaitable[Mapping[str, str]]]


@dataclass(frozen=True)
class DashboardUser:
    email: str
    dev_mode: bool

    @property
    def username(self) -> str:
        return self.email.split("@", 1)[0]


class IapVerifier:
    """Verifies IAP JWTs. ``fetch_certs`` returns {key id: PEM public key}; tests inject it."""

    def __init__(self, audience: str, fetch_certs: CertsFetcher | None = None) -> None:
        self.audience = audience
        self._fetch = fetch_certs or self._fetch_from_google
        self._client: httpx.AsyncClient | None = None
        self._certs: Mapping[str, str] = {}
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _fetch_from_google(self) -> Mapping[str, str]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        response = await self._client.get(IAP_CERTS_URL)
        response.raise_for_status()
        return response.json()

    def _needs_fetch(self, kid: str | None) -> bool:
        age = time.monotonic() - self._fetched_at
        return age > CERTS_TTL_S or (kid not in self._certs and age > CERTS_MIN_REFRESH_S)

    async def _keys(self, kid: str | None) -> Mapping[str, str]:
        if self._needs_fetch(kid):
            async with self._lock:
                if self._needs_fetch(kid):  # another request may have fetched meanwhile
                    try:
                        self._certs = dict(await self._fetch())
                        self._fetched_at = time.monotonic()
                    except Exception as exc:
                        logger.warning("could not fetch IAP public keys: %s", exc)
                        if not self._certs:
                            raise ApiError(503, "could not verify the IAP identity") from exc
        return self._certs

    async def verify(self, token: str) -> str:
        """The verified email in the token. Raises ApiError 401 when the token is invalid."""
        if not self.audience:
            logger.error("IAP audience is unknown: set K_SERVICE or IAP_AUDIENCE")
            raise ApiError(503, "IAP verification is not configured")
        certs = await self._keys(_key_id(token))
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                certs=dict(certs),
                audience=self.audience,
                clock_skew_in_seconds=CLOCK_SKEW_S,
            )
        except (gexc.GoogleAuthError, ValueError) as exc:
            raise ApiError(401, "invalid IAP assertion") from exc
        if claims.get("iss") != IAP_ISSUER:
            raise ApiError(401, "invalid IAP assertion issuer")
        email = claims.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise ApiError(401, "IAP assertion has no email")
        return email


def _key_id(token: str) -> str | None:
    try:
        header = token.split(".", 1)[0]
        kid = orjson.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4))).get("kid")
    except Exception:
        return None
    return kid if isinstance(kid, str) else None


def _check_domain(settings: Settings, email: str) -> None:
    if email.rsplit("@", 1)[-1] not in settings.dashboard_allowed_domains:
        raise ApiError(403, f"{email} may not use the dashboard")


async def current_user(request: Request) -> DashboardUser:
    settings: Settings = request.app.state.settings
    if settings.verify_iap:
        token = request.headers.get(IAP_JWT_HEADER)
        if not token:
            raise ApiError(401, "missing IAP identity")
        verifier: IapVerifier = request.app.state.iap_verifier
        email = (await verifier.verify(token)).strip().lower()
        _check_domain(settings, email)
        user = DashboardUser(email=email, dev_mode=False)
    else:
        raw = request.headers.get(IAP_EMAIL_HEADER)
        if raw:
            email = raw.split(":", 1)[1] if ":" in raw else raw
            user = DashboardUser(email=email.strip().lower(), dev_mode=False)
            _check_domain(settings, user.email)
        elif settings.is_local:
            user = DashboardUser(email=settings.dashboard_dev_user.lower(), dev_mode=True)
        else:
            raise ApiError(401, "missing IAP identity")
    principal_var.set(user.email)
    return user
