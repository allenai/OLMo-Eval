"""Download-once cache for benchmark files that live outside the Hugging Face Hub.

Files are stored under ``$OLMO_EVAL_DOWNLOAD_CACHE`` (default ``$HF_HOME/olmo-eval-downloads``,
with ``HF_HOME`` defaulting to ``~/.cache/huggingface``), at a path that includes the pinned
revision in the URL, so a cached file is never stale. Writes go to a temporary file in the same
directory and are renamed into place, so concurrent workers never read a partial file.
"""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

__all__ = ["download_cache_dir", "cached_download"]


def download_cache_dir() -> Path:
    """The root of the download cache."""
    override = os.environ.get("OLMO_EVAL_DOWNLOAD_CACHE")
    if override:
        return Path(override)
    hf_home = os.environ.get("HF_HOME", os.path.join("~", ".cache", "huggingface"))
    return Path(os.path.expanduser(hf_home)) / "olmo-eval-downloads"


def cached_download(url: str, relpath: str, *, retries: int = 3, timeout: float = 60.0) -> Path:
    """Return the local copy of ``url``, downloading it to ``<cache>/<relpath>`` on first use.

    :param relpath: Cache-relative destination. Include the pinned revision so that a change
        of revision downloads afresh.
    :raises OSError: If the download still fails after ``retries`` attempts.
    """
    target = download_cache_dir() / relpath
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
            fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            return target
        except OSError:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")
