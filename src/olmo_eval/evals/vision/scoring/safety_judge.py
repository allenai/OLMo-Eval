"""Cached OpenAI judge shared by the multimodal safety benchmarks.

MM-SafetyBench, SIUO and USB all grade free-form responses with an OpenAI
model. Each benchmark supplies its own verbatim judge prompt, model and
decoding settings; this module only makes the call, optionally attaching the
benchmark image the way the official scripts do (the raw file as a base64 data
URL), retries transient failures with randomized exponential backoff, and caches
the reply on disk so a re-run is free.

A call that never returns usable text yields ``None``; each benchmark records
that as a judge error and leaves the example out of its rates, rather than
counting it as safe or unsafe. Unusable credentials raise instead, since no
retry can succeed. The cache dir comes from ``SAFETY_JUDGE_CACHE_DIR`` or a
fresh per-process temp dir; ``OPENAI_API_KEY`` is needed on cache misses.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 30.0

_PROCESS_CACHE_DIR: list[str] = []
_ASYNC_CLIENTS: dict[str, Any] = {}

_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def default_cache_dir() -> str:
    """Judge-response cache dir: env override or a fresh process-local temp dir."""
    env_dir = os.environ.get("SAFETY_JUDGE_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_CACHE_DIR:
        _PROCESS_CACHE_DIR.append(tempfile.mkdtemp(prefix="safety-judge-cache-"))
    return _PROCESS_CACHE_DIR[0]


def _get_client() -> Any:
    if "default" not in _ASYNC_CLIENTS:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the safety judge on a cache miss.")
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai") from None
        _ASYNC_CLIENTS["default"] = AsyncOpenAI(api_key=api_key)
    return _ASYNC_CLIENTS["default"]


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (401, 403):
        return True
    return type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")


def _retry_delay(attempt: int) -> float:
    """Randomized exponential backoff, in seconds, for judge attempt ``attempt``."""
    return min(_RETRY_MAX_DELAY, _RETRY_BASE_DELAY * 2 ** (attempt - 1)) * (0.5 + random.random())


def image_data_url(image_path: str) -> str:
    """The image file as a base64 data URL, typed by its extension (JPEG otherwise)."""
    mime = _MIME_TYPES.get(Path(image_path).suffix.lower(), "image/jpeg")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _cache_key(model: str, prompt: str, image_path: str | None, params: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model}\x00{prompt}\x00".encode())
    if image_path is not None:
        with open(image_path, "rb") as f:
            digest.update(hashlib.sha256(f.read()).hexdigest().encode())
    digest.update(b"\x00" + json.dumps(params, sort_keys=True).encode())
    return digest.hexdigest()


async def judge(
    prompt: str,
    *,
    model: str,
    max_tokens: int | None,
    temperature: float = 0.0,
    image_path: str | None = None,
    image_first: bool = False,
    system: str | None = None,
    response_format: dict[str, Any] | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    cache_dir: str | None = None,
    max_attempts: int = 5,
) -> str | None:
    """One judge call; the reply text, or ``None`` when every attempt failed.

    ``image_path`` attaches that file to the user turn, after the prompt text
    unless ``image_first`` is set. ``system`` adds a system turn, and
    ``response_format`` is passed through (a JSON schema for structured output,
    whose reply text is then the JSON). ``max_tokens=None`` leaves the cap to the API.
    """
    params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "image_first": image_first,
    }
    for name, value in (
        ("system", system),
        ("response_format", response_format),
        ("top_p", top_p),
        ("seed", seed),
    ):
        if value is not None:
            params[name] = value
    cache_dir = cache_dir or default_cache_dir()
    key = _cache_key(model, prompt, image_path, params)
    cache_file = Path(cache_dir) / f"{key}-v1.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)["content"]

    if image_path is not None:
        parts: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ]
        content: Any = parts[::-1] if image_first else parts
    else:
        content = prompt

    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    kwargs: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    for name in ("response_format", "top_p", "seed"):
        if name in params:
            kwargs[name] = params[name]

    client = _get_client()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            completion = await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            text = completion.choices[0].message.content
            if text and text.strip():
                break
            # A refusal or an empty reply is not transient for a deterministic call.
            logger.warning(
                "Safety judge %s returned no content (finish_reason=%s)",
                model,
                completion.choices[0].finish_reason,
            )
            return None
        except Exception as e:
            if _is_auth_error(e):
                raise
            last_error = e
            logger.warning("Safety judge error (attempt %d/%d): %s", attempt, max_attempts, e)
            if attempt < max_attempts:
                await asyncio.sleep(_retry_delay(attempt))
    else:
        logger.warning("Safety judge failed after %d attempts: %s", max_attempts, last_error)
        return None

    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(".tmp", prefix=f"{key}-v1.json", text=True, dir=cache_dir)
    os.close(fd)
    with open(tmp, "w") as f:
        json.dump({"model": model, "content": text}, f)
    os.rename(tmp, str(cache_file))
    return text


async def gather_bounded(coros, limit: int) -> list:
    """Await ``coros`` with at most ``limit`` in flight, preserving order."""
    semaphore = asyncio.Semaphore(limit)

    async def _run(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_run(c) for c in coros))
