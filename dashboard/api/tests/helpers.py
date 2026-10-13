"""Paths and loaders for the shared contract files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "contract"
EXAMPLES_DIR = CONTRACT_DIR / "examples"


def load_example(relative: str) -> Any:
    return json.loads((EXAMPLES_DIR / relative).read_text())


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((CONTRACT_DIR / name).read_text())
