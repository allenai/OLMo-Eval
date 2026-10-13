"""Google OAuth access-token verification for the ingest service (spec 3.1).

The ingest service is reachable without IAP login, so every request must carry a Google access
token. The token is checked with Google's tokeninfo endpoint and the verified email is matched
against the allowed domains and service accounts. Results are cached in process.

Access tokens carry no audience we can check, so a token that some other Google sign-in client
received (for example a website using "Sign in with Google" with the email scope) would also
pass tokeninfo. User tokens must therefore include the cloud-platform scope: a holder of such a
token already has the user's full Google Cloud access, so accepting it grants nothing new.
gcloud application-default credentials and the olmo-eval client both request that scope.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import httpx
from fastapi import Request

from olmo_eval_api.errors import ApiError
from olmo_eval_api.log_config import principal_var
from olmo_eval_api.settings import Settings

logger = logging.getLogger(__name__)

TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
TOKEN_HEADER = "x-olmo-eval-token"
REQUIRED_USER_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# Short, so a revoked token stops working soon after revocation.
POSITIVE_TTL_S = 60.0
NEGATIVE_TTL_S = 30.0
CACHE_MAX = 10_000
# tokeninfo answers 400 for a malformed, unknown or expired token. Any other failure, such as
# rate limiting, says nothing about the token.
INVALID_TOKEN_STATUSES = frozenset({400, 401})
MAX_TOKEN_LENGTH = 4096
_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~+/=")

PrincipalType = Literal["user", "service_account"]


@dataclass(frozen=True)
class Principal:
    email: str
    principal_type: PrincipalType
    exp: float

    @property
    def expires_in(self) -> int:
        return max(0, int(self.exp - time.time()))


@dataclass(frozen=True)
class _Rejection:
    status_code: int
    message: str


@dataclass(frozen=True)
class TokenInfo:
    """What tokeninfo reported for a token, or None when Google rejected it."""

    email: str | None
    email_verified: bool
    exp: float
    scopes: frozenset[str] = frozenset()


class TokenVerifier:
    """Verifies tokens with tokeninfo and applies the allowlist, with a TTL cache."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        # Rejections live in their own cache so a flood of bad tokens cannot evict good ones.
        self._accepted: OrderedDict[str, tuple[float, Principal | _Rejection]] = OrderedDict()
        self._rejected: OrderedDict[str, tuple[float, Principal | _Rejection]] = OrderedDict()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=5.0)
        return self._client

    async def fetch_tokeninfo(self, token: str) -> TokenInfo | None:
        """Call Google's tokeninfo. Returns None when the token is invalid or expired."""
        try:
            # POST keeps the token out of URLs, which HTTP clients and proxies tend to log.
            response = await self._http().post(TOKENINFO_URL, data={"access_token": token})
        except httpx.HTTPError as exc:
            logger.warning("tokeninfo unreachable: %s", exc)
            raise ApiError(503, "could not verify the access token; try again") from exc
        if response.status_code in INVALID_TOKEN_STATUSES:
            return None
        if response.status_code != 200:
            logger.warning("tokeninfo returned HTTP %s", response.status_code)
            raise ApiError(503, "could not verify the access token; try again")
        data = response.json()
        try:
            exp = float(data.get("exp", 0))
        except (TypeError, ValueError):
            exp = 0.0
        return TokenInfo(
            email=data.get("email"),
            email_verified=str(data.get("email_verified", "")).lower() == "true",
            exp=exp,
            scopes=frozenset(str(data.get("scope") or "").split()),
        )

    def _authorize(self, email: str, exp: float) -> Principal | _Rejection:
        email = email.lower()
        if email.endswith(".gserviceaccount.com"):
            if email in self.settings.ingest_allowed_service_accounts:
                return Principal(email=email, principal_type="service_account", exp=exp)
            return _Rejection(403, f"{email} is not allowed to upload results")
        domain = email.rsplit("@", 1)[-1]
        if domain in self.settings.ingest_allowed_domains:
            return Principal(email=email, principal_type="user", exp=exp)
        return _Rejection(403, f"{email} is not allowed to upload results")

    async def _verify_uncached(self, token: str) -> Principal | _Rejection:
        if self.settings.dev_auth_enabled and token.startswith("dev:"):
            return self._authorize(token[4:], time.time() + 3600)
        if len(token) > MAX_TOKEN_LENGTH or not set(token) <= _TOKEN_CHARS:
            return _Rejection(401, "invalid or expired Google access token")
        info = await self.fetch_tokeninfo(token)
        if info is None or info.exp <= time.time():
            return _Rejection(401, "invalid or expired Google access token")
        if not info.email:
            return _Rejection(
                401,
                "token lacks the userinfo.email scope; run `gcloud auth application-default login`",
            )
        if not info.email_verified:
            return _Rejection(401, "the token's email address is not verified")
        result = self._authorize(info.email, info.exp)
        if (
            isinstance(result, Principal)
            and result.principal_type == "user"
            and REQUIRED_USER_SCOPE not in info.scopes
        ):
            return _Rejection(
                401,
                "token lacks the cloud-platform scope; run `gcloud auth application-default login`",
            )
        return result

    async def verify(self, token: str) -> Principal:
        key = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        cached = self._accepted.get(key) or self._rejected.get(key)
        if cached is not None and cached[0] > now:
            result = cached[1]
        else:
            result = await self._verify_uncached(token)
            if isinstance(result, Principal):
                cache, valid_until = self._accepted, min(result.exp, now + POSITIVE_TTL_S)
            else:
                cache, valid_until = self._rejected, now + NEGATIVE_TTL_S
            cache[key] = (valid_until, result)
            cache.move_to_end(key)
            while len(cache) > CACHE_MAX:
                cache.popitem(last=False)
        if isinstance(result, _Rejection):
            raise ApiError(result.status_code, result.message, headers=_www_authenticate(result))
        return result


def _www_authenticate(rejection: _Rejection) -> dict[str, str] | None:
    return {"WWW-Authenticate": "Bearer"} if rejection.status_code == 401 else None


def extract_token(request: Request) -> str | None:
    token = request.headers.get(TOKEN_HEADER)
    if token:
        return token.strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return None


async def require_principal(request: Request) -> Principal:
    """FastAPI dependency for every ingest endpoint except /health."""
    token = extract_token(request)
    if not token:
        raise ApiError(
            401,
            "missing Google access token (send X-Olmo-Eval-Token)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    verifier: TokenVerifier = request.app.state.token_verifier
    principal = await verifier.verify(token)
    principal_var.set(principal.email)
    return principal
