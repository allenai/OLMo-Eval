"""Setup and launch support for the reference retrieval server."""

from pathlib import Path


def upstream_directory() -> Path:
    """Return the bundled reference source and assets."""
    return Path(__file__).resolve().parent / "upstream"
