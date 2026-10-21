"""Tests for running external evals with an OpenRouter key in place of an OpenAI key."""

from __future__ import annotations

import pytest

from olmo_eval.evals.external.benchmarks.asta.args import AstaArgs
from olmo_eval.evals.external.benchmarks.asta.eval import AstaExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar.args import DeepScholarArgs
from olmo_eval.evals.external.benchmarks.deepscholar.eval import DeepScholarExternalEval
from olmo_eval.evals.external.benchmarks.tau2.eval import Tau2Args, Tau2ExternalEval

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.fixture
def openrouter_only(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")


@pytest.fixture
def openai_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.mark.usefixtures("openrouter_only")
def test_sandbox_env_points_openai_sdk_at_openrouter() -> None:
    env = Tau2ExternalEval()._build_env_vars()
    assert env == {
        "OPENROUTER_API_KEY": "or-key",
        "OPENAI_API_KEY": "or-key",
        "OPENAI_BASE_URL": OPENROUTER_BASE_URL,
    }


@pytest.mark.usefixtures("openai_only")
def test_sandbox_env_with_openai_key() -> None:
    assert Tau2ExternalEval()._build_env_vars() == {"OPENAI_API_KEY": "sk-openai"}


def test_sandbox_env_without_keys_names_both(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"OPENAI_API_KEY \(or OPENROUTER_API_KEY\)"):
        Tau2ExternalEval()._build_env_vars()


@pytest.mark.usefixtures("openrouter_only")
def test_tau2_routes_openai_models_through_openrouter() -> None:
    command = Tau2ExternalEval()._build_run_command(
        "openai/gpt-4o", "http://localhost:8000/v1", False, Tau2Args.from_dict({})
    )
    assert "--agent-llm openrouter/openai/gpt-4o" in command
    assert "--user-llm openrouter/openai/gpt-4o-mini" in command


@pytest.mark.usefixtures("openrouter_only")
def test_tau2_local_agent_is_unchanged() -> None:
    command = Tau2ExternalEval()._build_run_command(
        "my-model", "http://10.0.0.1:8000/v1", True, Tau2Args.from_dict({})
    )
    assert "--agent-llm hosted_vllm/my-model" in command


@pytest.mark.usefixtures("openai_only")
def test_tau2_keeps_openai_names_with_openai_key() -> None:
    command = Tau2ExternalEval()._build_run_command(
        "gpt-4o", "http://localhost:8000/v1", False, Tau2Args.from_dict({})
    )
    assert "--agent-llm gpt-4o " in command
    assert "--user-llm gpt-4o-mini" in command


@pytest.mark.usefixtures("openrouter_only")
def test_deepscholar_judge_routes_through_openrouter() -> None:
    command = DeepScholarExternalEval()._build_eval_command(DeepScholarArgs.from_dict({}))
    assert "--model-name openrouter/openai/gpt-4o" in command


@pytest.mark.usefixtures("openai_only")
def test_deepscholar_judge_keeps_openai_name_with_openai_key() -> None:
    command = DeepScholarExternalEval()._build_eval_command(DeepScholarArgs.from_dict({}))
    assert "--model-name gpt-4o" in command


@pytest.mark.usefixtures("openrouter_only")
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "openai/gpt-4o"),
        ("gpt-4-0613", "openai/gpt-4"),
        ("gpt-5.5-2026-04-23", "openai/gpt-5.5"),
    ],
)
def test_deepscholar_model_under_test_routes_through_openrouter(model: str, expected: str) -> None:
    lm = DeepScholarExternalEval()._build_lm_config(
        model, "http://localhost:8000/v1", False, DeepScholarArgs.from_dict({})
    )
    assert lm["model"] == expected


@pytest.mark.usefixtures("openrouter_only")
def test_deepscholar_local_model_is_unchanged() -> None:
    args = DeepScholarArgs.from_dict({})
    lm = DeepScholarExternalEval()._build_lm_config(
        "my-model", "http://10.0.0.1:8000/v1", True, args
    )
    assert lm["model"] == f"{args.local_model_prefix}/my-model"


@pytest.mark.usefixtures("openai_only")
def test_deepscholar_model_under_test_keeps_openai_name_with_openai_key() -> None:
    lm = DeepScholarExternalEval()._build_lm_config(
        "gpt-4-0613", "http://localhost:8000/v1", False, DeepScholarArgs.from_dict({})
    )
    assert lm["model"] == "gpt-4-0613"


@pytest.mark.usefixtures("openrouter_only")
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-4o", "openrouter/openai/gpt-4o"),
        ("openai/o3-mini-2025-01-31", "openrouter/openai/o3-mini"),
    ],
)
def test_asta_model_under_test_routes_through_openrouter(model: str, expected: str) -> None:
    command = AstaExternalEval()._build_run_command(
        model, "http://localhost:8000/v1", False, AstaArgs.from_dict({})
    )
    assert f"--model {expected}" in command


@pytest.mark.usefixtures("openrouter_only")
def test_asta_local_model_is_unchanged() -> None:
    command = AstaExternalEval()._build_run_command(
        "my-model", "http://10.0.0.1:8000/v1", True, AstaArgs.from_dict({})
    )
    assert "openai-api/vllm/my-model" in command


@pytest.mark.usefixtures("openai_only")
def test_asta_keeps_openai_name_with_openai_key() -> None:
    command = AstaExternalEval()._build_run_command(
        "openai/o3-mini-2025-01-31", "http://localhost:8000/v1", False, AstaArgs.from_dict({})
    )
    assert "--model openai/o3-mini-2025-01-31" in command
