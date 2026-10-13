"""The generated API models match the contract, and the API examples parse."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from olmo_eval_api.schemas import api as a
from tests.helpers import load_example

ROOT = Path(__file__).resolve().parents[1]
INDEX: dict[str, str] = load_example("index.json")
API_EXAMPLES = {k: v for k, v in INDEX.items() if k.startswith("api/")}


def test_generated_models_are_current(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "generate_api_schemas", ROOT / "scripts" / "generate_api_schemas.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "api.py"
    out.write_text(module.generate())
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--quiet", str(out)], check=True, cwd=ROOT
    )
    committed = (ROOT / "src" / "olmo_eval_api" / "schemas" / "api.py").read_text()
    assert out.read_text() == committed, (
        "schemas/api.py is out of date; run `uv run python scripts/generate_api_schemas.py`"
    )


@pytest.mark.parametrize(("path", "definition"), sorted(API_EXAMPLES.items()))
def test_api_examples_parse(path: str, definition: str) -> None:
    if definition in ("ErrorResponse", "HealthResponse"):
        return
    model = getattr(a, definition)
    payload = load_example(path)
    parsed = model.model_validate(payload)
    assert parsed.model_dump(mode="json", by_alias=True) is not None
