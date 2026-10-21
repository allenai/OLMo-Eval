import asyncio
import base64
import importlib.util
import inspect
import json
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from olmo_eval.common.execution.environment import ExecutionResult
from olmo_eval.evals.external.base import SandboxedExternalEval
from olmo_eval.evals.external.benchmarks.deepscholar import citation_lookup_patch
from olmo_eval.evals.external.benchmarks.deepscholar.args import (
    PRIMARY_METRICS,
    DeepScholarArgs,
)
from olmo_eval.evals.external.benchmarks.deepscholar.eval import (
    _HALF_SCALE_METRICS,
    LOTUS_REF,
    DeepScholarExternalEval,
)
from olmo_eval.evals.external.benchmarks.deepscholar.result_parser import (
    parse_aggregate_csv,
    parse_per_query_csv,
)
from olmo_eval.evals.external.benchmarks.deepscholar.sandbox_search_shim import (
    map_s2_paper,
    s2_search_rows,
)
from olmo_eval.harness.sandbox.config import SandboxConfig, SandboxMode


def test_output_directories_are_bind_mounted(tmp_path: Path) -> None:
    evaluator = DeepScholarExternalEval()
    base_config = SandboxConfig(image="test", mode=SandboxMode.DOCKER)

    with mock.patch.object(
        SandboxedExternalEval, "_create_sandbox_config", return_value=base_config
    ):
        config = evaluator._create_sandbox_config("podman", str(tmp_path))

    destination = tmp_path / "deepscholar_results"
    assert config.inject_swerex
    assert config.volumes == (
        (str(destination / "generation"), evaluator._gen_dir),
        (str(destination / "evaluation"), evaluator._eval_dir),
    )
    assert (destination / "generation").is_dir()
    assert (destination / "evaluation").is_dir()
    assert stat.S_IMODE((destination / "generation").stat().st_mode) == 0o777
    assert stat.S_IMODE((destination / "evaluation").stat().st_mode) == 0o777


def test_setup_pins_lotus_revision() -> None:
    evaluator = DeepScholarExternalEval()

    assert any(LOTUS_REF in command for command in evaluator.setup_command)
    assert evaluator.required_secrets == ("OPENAI_API_KEY", "OPENALEX_API_KEY")


def test_organization_patch_is_written_to_upstream_evaluator() -> None:
    evaluator = DeepScholarExternalEval()
    executor = mock.Mock()
    executor.execute_command = mock.AsyncMock(return_value=ExecutionResult(success=True))

    asyncio.run(evaluator._write_organization_eval_patch(executor))

    awaited_call = executor.execute_command.await_args
    assert awaited_call is not None
    command = awaited_call.args[0]
    assert "base64 -d" in command
    assert command.endswith("> /workspace/deepscholar-bench/eval/evaluator/organization.py")


def test_citation_patch_is_applied_to_upstream_utils() -> None:
    evaluator = DeepScholarExternalEval()
    executor = mock.Mock()
    executor.execute_command = mock.AsyncMock(return_value=ExecutionResult(success=True))

    asyncio.run(evaluator._patch_citation_lookup(executor))

    awaited_call = executor.execute_command.await_args
    assert awaited_call is not None
    command = awaited_call.args[0]
    assert "base64 -d | python3 -" in command
    assert command.endswith("/workspace/deepscholar-bench/eval/utils.py")


def test_organization_patch_scores_normalized_lotus_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_path = (
        Path(inspect.getfile(DeepScholarExternalEval)).parent / "organization_eval_patch.py"
    )
    fake_lotus = types.ModuleType("lotus")
    fake_evaluator = types.ModuleType("evaluator")
    fake_evaluator.__dict__.update(
        Evaluator=object,
        EvaluationFunction=SimpleNamespace(ORGANIZATION=SimpleNamespace(value="organization")),
    )
    fake_parsers = types.ModuleType("parsers")
    fake_parsers.__dict__["Parser"] = object
    module_name = "test_deepscholar_organization_eval_patch"
    spec = importlib.util.spec_from_file_location(module_name, patch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    captured_kwargs: list[dict[str, object]] = []

    def pairwise_judge(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        captured_kwargs.append(kwargs)
        result = frame.copy()
        if kwargs["col1"] == "related_work_a":
            result["raw_output_judge_forward_0"] = ["A", "B", " A ", "B"]
        else:
            result["raw_output_judge_reverse_0"] = ["B", "A", "A", "B"]
        return result

    class FakeParser:
        def __init__(self, index: int):
            self.index = index

        def get_folder_info(self, include_related_works_section: bool) -> dict[str, str]:
            assert include_related_works_section
            return {
                "folder_path": f"/generation/{self.index}",
                "paper_title": f"paper {self.index}",
                "paper_abstract": "abstract",
                "generated_related_works_section": "generated",
                "related_works_section": "reference",
            }

    with mock.patch.dict(
        sys.modules,
        {
            "lotus": fake_lotus,
            "evaluator": fake_evaluator,
            "parsers": fake_parsers,
        },
    ):
        monkeypatch.setattr(pd.DataFrame, "pairwise_judge", pairwise_judge, raising=False)
        spec.loader.exec_module(module)

    result = module.OrganizationEvaluator().calculate([FakeParser(i) for i in range(4)])

    assert result["organization_v1"].tolist() == [1, 0, 1, 0]
    assert result["organization_v2"].tolist() == [1, 0, 0, 1]
    assert result["organization"].tolist() == [1.0, 0.0, 0.5, 0.5]
    assert result["organization_raw_v1"].tolist() == ["A", "B", " A ", "B"]
    assert result["organization_raw_v2"].tolist() == ["B", "A", "A", "B"]
    assert len(captured_kwargs) == 2
    assert [kwargs["col1"] for kwargs in captured_kwargs] == [
        "related_work_a",
        "related_work_b",
    ]
    assert all(kwargs["n_trials"] == 1 for kwargs in captured_kwargs)
    assert all(kwargs["permute_cols"] is False for kwargs in captured_kwargs)
    assert all(kwargs["return_raw_outputs"] is True for kwargs in captured_kwargs)
    assert all("response_format" not in kwargs for kwargs in captured_kwargs)
    instruction = str(captured_kwargs[0]["judge_instruction"])
    assert "Return exactly the choice token" in instruction
    assert "not return json" in instruction.lower()


def test_organization_patch_rejects_non_exact_judge_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_path = (
        Path(inspect.getfile(DeepScholarExternalEval)).parent / "organization_eval_patch.py"
    )
    fake_lotus = types.ModuleType("lotus")
    fake_evaluator = types.ModuleType("evaluator")
    fake_evaluator.__dict__.update(
        Evaluator=object,
        EvaluationFunction=SimpleNamespace(ORGANIZATION=SimpleNamespace(value="organization")),
    )
    fake_parsers = types.ModuleType("parsers")
    fake_parsers.__dict__["Parser"] = object
    module_name = "test_deepscholar_organization_eval_patch_invalid"
    spec = importlib.util.spec_from_file_location(module_name, patch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    def pairwise_judge(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        result = frame.copy()
        suffix = str(kwargs["suffix"])
        result[f"raw_output{suffix}_0"] = ["A because it flows better"]
        return result

    parser = mock.Mock()
    parser.get_folder_info.return_value = {
        "folder_path": "/generation/0",
        "paper_title": "paper",
        "paper_abstract": "abstract",
        "generated_related_works_section": "generated",
        "related_works_section": "reference",
    }

    with mock.patch.dict(
        sys.modules,
        {
            "lotus": fake_lotus,
            "evaluator": fake_evaluator,
            "parsers": fake_parsers,
        },
    ):
        monkeypatch.setattr(pd.DataFrame, "pairwise_judge", pairwise_judge, raising=False)
        spec.loader.exec_module(module)

    with pytest.raises(ValueError, match="non-A/B output for 1/1 rows"):
        module.OrganizationEvaluator().calculate([parser])


def test_citation_patch_authenticates_retries_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "utils.py"
    source_path.write_text(
        "import os\nimport requests\nimport time\n\n"
        "def jaccard_similarity(left, right):\n"
        "    left_tokens = set(left.lower().split())\n"
        "    right_tokens = set(right.lower().split())\n"
        "    union = left_tokens | right_tokens\n"
        "    return len(left_tokens & right_tokens) / len(union) if union else 0\n\n\n"
        + citation_lookup_patch._FUNCTION_OLD
    )
    monkeypatch.setattr(sys, "argv", ["citation_lookup_patch.py", str(source_path)])
    citation_lookup_patch.main()

    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)

    rate_limited = mock.Mock(status_code=429, headers={"Retry-After": "0"})
    success = mock.Mock(status_code=200, headers={})
    success.json.return_value = {
        "results": [{"display_name": "A useful paper", "cited_by_count": 42}]
    }
    fake_get = mock.Mock(side_effect=[rate_limited, success])
    fake_time = SimpleNamespace(
        monotonic=mock.Mock(side_effect=[1.0, 1.1, 1.2, 1.3]), sleep=mock.Mock()
    )
    namespace["requests"] = SimpleNamespace(
        get=fake_get,
        RequestException=Exception,
    )
    namespace["time"] = fake_time
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    monkeypatch.setenv("OPENALEX_EMAIL", "researcher@example.org")

    lookup = namespace["get_citation_count_from_title"]
    assert callable(lookup)
    assert lookup("A useful paper?") == 42
    assert lookup("A useful paper?") == 42
    assert fake_get.call_count == 2
    assert fake_get.call_args.args == ("https://api.openalex.org/works",)
    assert fake_get.call_args.kwargs["params"] == {
        "search": '"A useful paper"',
        "api_key": "secret",
        "per_page": 1,
        "select": "display_name,cited_by_count",
        "mailto": "researcher@example.org",
    }


def test_citation_patch_does_not_retry_or_leak_key_on_http_400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "utils.py"
    source_path.write_text(
        "import os\nimport requests\nimport time\n\n"
        "def jaccard_similarity(left, right):\n"
        "    return 1.0\n\n\n" + citation_lookup_patch._FUNCTION_OLD
    )
    monkeypatch.setattr(sys, "argv", ["citation_lookup_patch.py", str(source_path)])
    citation_lookup_patch.main()

    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)

    api_key = "do-not-leak"
    bad_request = SimpleNamespace(
        status_code=400,
        headers={},
        text=f"invalid wildcard request using {api_key}",
    )
    fake_get = mock.Mock(return_value=bad_request)
    namespace["requests"] = SimpleNamespace(
        get=fake_get,
        RequestException=Exception,
    )
    namespace["time"] = SimpleNamespace(monotonic=mock.Mock(return_value=1.0), sleep=mock.Mock())
    monkeypatch.setenv("OPENALEX_API_KEY", api_key)

    lookup = namespace["get_citation_count_from_title"]
    assert callable(lookup)
    with pytest.raises(RuntimeError, match="OpenAlex HTTP 400") as error:
        lookup("Can Language Models Learn to Listen?")

    assert fake_get.call_count == 1
    assert fake_get.call_args.kwargs["params"]["search"] == (
        '"Can Language Models Learn to Listen"'
    )
    assert api_key not in str(error.value)
    assert "<redacted>" in str(error.value)


def test_citation_patch_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "utils.py"
    source_path.write_text(
        "import os\nimport requests\nimport time\n\n"
        "def jaccard_similarity(left, right):\n"
        "    return 1.0\n\n\n" + citation_lookup_patch._FUNCTION_OLD
    )
    monkeypatch.setattr(sys, "argv", ["citation_lookup_patch.py", str(source_path)])
    citation_lookup_patch.main()
    namespace: dict[str, object] = {}
    exec(compile(source_path.read_text(), str(source_path), "exec"), namespace)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    lookup = namespace["get_citation_count_from_title"]
    assert callable(lookup)
    with pytest.raises(RuntimeError, match="OPENALEX_API_KEY is required"):
        lookup("A useful paper")


def test_all_metrics_argument_expands_to_primary_metrics() -> None:
    assert DeepScholarArgs.from_dict({"evals": "all"}).evals == list(PRIMARY_METRICS)


def test_locality_is_decided_from_actual_base_url() -> None:
    # execute() passes `base_url or ""` so a provider without a base_url is treated
    # as an external API model (no forced-local health check), not mislabeled local
    # by the localhost fallback used for the provider URL.
    evaluator = DeepScholarExternalEval()

    external = SimpleNamespace(base_url=None)
    assert evaluator._is_local_provider(external, external.base_url or "") is False

    local = SimpleNamespace(base_url="http://localhost:8000/v1")
    assert evaluator._is_local_provider(local, local.base_url or "") is True

    # A provider managing its own server is still local via the _server check even
    # when it has not yet assigned a base_url.
    managed = SimpleNamespace(base_url=None, _server=object())
    assert evaluator._is_local_provider(managed, managed.base_url or "") is True


def test_config_preserves_final_generation_token_budget() -> None:
    evaluator = DeepScholarExternalEval()
    executor = mock.Mock()
    executor.execute_command = mock.AsyncMock(return_value=ExecutionResult(success=True))

    asyncio.run(
        evaluator._write_config(
            executor,
            "test-model",
            "http://host.containers.internal:8000/v1",
            True,
            DeepScholarArgs(max_tokens=10000, stage_max_tokens=4096),
        )
    )

    command = executor.execute_command.await_args.args[0]
    encoded = command.split("'", 2)[1]
    config = json.loads(base64.b64decode(encoded))
    assert config["lm"]["max_tokens"] == 10000
    assert config["generation_lm"]["max_tokens"] == 10000
    assert config["generation_lm"] == config["lm"]


def test_s2_mapping_prefers_scorable_arxiv_metadata() -> None:
    row = map_s2_paper(
        {
            "title": "A paper",
            "abstract": "An abstract",
            "authors": [{"name": "Ada"}, {"name": "Grace"}],
            "year": 2024,
            "externalIds": {"ArXiv": "2401.01234", "DOI": "10.1/example"},
        },
        "test query",
    )

    assert row["url"] == "https://arxiv.org/abs/2401.01234"
    assert row["id"] == "2401.01234"
    assert row["date"] == "2024-01-01"
    assert row["authors"] == "Ada, Grace"
    assert row["query"] == "test query"


def test_s2_search_sends_key_and_recovers_from_rate_limit() -> None:
    rate_limited = mock.Mock(status_code=429)
    success = mock.Mock(status_code=200)
    success.json.return_value = {
        "data": [
            {
                "title": "A paper",
                "abstract": "An abstract",
                "publicationDate": "2024-01-02",
                "externalIds": {"ArXiv": "2401.01234"},
            }
        ]
    }

    with (
        mock.patch("requests.get", side_effect=[rate_limited, success]) as get,
        mock.patch("time.sleep"),
    ):
        rows = s2_search_rows("test query", 10, api_key="secret", budget_sec=5)

    assert len(rows) == 1
    assert get.call_count == 2
    assert get.call_args.kwargs["headers"] == {"x-api-key": "secret"}
    assert get.call_args.kwargs["timeout"] == (5, 15)
    success.raise_for_status.assert_called_once_with()


def test_s2_search_excludes_same_year_when_only_year_is_known() -> None:
    response = mock.Mock(status_code=200)
    response.json.return_value = {
        "data": [
            {"title": "Unknown date", "year": 2024},
            {"title": "Known earlier year", "year": 2023},
            {"title": "Known earlier date", "publicationDate": "2024-05-31"},
        ]
    }

    with mock.patch("requests.get", return_value=response):
        rows = s2_search_rows(
            "test query",
            10,
            end_date=pd.Timestamp("2024-06-01"),
            api_key="secret",
            budget_sec=5,
        )

    assert [row["title"] for row in rows] == ["Known earlier year", "Known earlier date"]
    assert rows[0]["date"] == "2023-01-01"


def test_generation_runs_once_and_counts_only_scorable_artifacts(tmp_path: Path) -> None:
    evaluator = DeepScholarExternalEval()
    generation = tmp_path / "deepscholar_results" / "generation"
    complete = generation / "0"
    incomplete = generation / "1"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    for filename in ("final_report.md", "intro.md", "paper.csv"):
        (complete / filename).write_text("complete")
    (incomplete / "final_report.md").write_text("not enough")
    (generation / "summary.json").write_text(
        json.dumps([{"status": "success"}, {"status": "success"}])
    )

    executor = mock.Mock()
    executor.execute_command = mock.AsyncMock(
        side_effect=[
            ExecutionResult(success=True, output="63\n"),
            ExecutionResult(success=True, output="generation output"),
        ]
    )

    counts = asyncio.run(
        evaluator._run_generation(
            executor,
            DeepScholarArgs(limit=2, search_backend="s2"),
            [],
            str(tmp_path),
        )
    )

    assert counts == (1, 2, True)
    generation_commands = [
        call.args[0]
        for call in executor.execute_command.await_args_list
        if "--output-folder" in call.args[0]
    ]
    assert len(generation_commands) == 1
    assert "--start-idx 0 --end-idx 2" in generation_commands[0]
    assert "Chunk" not in generation_commands[0]


def test_metric_parsers_skip_invalid_values() -> None:
    text = (
        "folder_path,cite_p\n"
        "/workspace/outputs/generation/1,0.25\n"
        "/workspace/outputs/generation/2,nan\n"
        "/workspace/outputs/generation/3,invalid\n"
    )

    assert parse_per_query_csv(text, "cite_p") == {"1": 0.25}
    assert parse_aggregate_csv("baseline_name,cite_p\ndeepscholar_base,nan\n", "cite_p") is None


def test_missing_requested_metric_fails_the_eval() -> None:
    evaluator = DeepScholarExternalEval()
    aggregate_files = {
        "cite_p/aggregated_results.csv": "baseline_name,cite_p\ndeepscholar_base,0.5\n"
    }
    query_files = {
        "cite_p/deepscholar_base.csv": ("folder_path,cite_p\n/workspace/generation/0,0.5\n")
    }

    with mock.patch.object(
        evaluator,
        "_read_dir",
        new=mock.AsyncMock(side_effect=[aggregate_files, query_files]),
    ):
        result = asyncio.run(
            evaluator._extract_results(
                mock.Mock(),
                "raw output",
                0,
                n_success=1,
                n_total=1,
                requested_metrics=("cite_p", "organization"),
            )
        )

    assert result.success is False
    assert result.error == "Missing requested eval metrics: organization (eval phase exited 0)"


@pytest.mark.parametrize("metric", PRIMARY_METRICS)
def test_fixed_metrics_use_requested_query_denominator(metric: str) -> None:
    evaluator = DeepScholarExternalEval()

    # _HALF_SCALE_METRICS arrive on upstream's 0-2 scale and are halved by
    # _extract_results. Feed those doubled so every metric lands at the same 0-1
    # value post-normalization (0.5 aggregate, 0.25/0.75 per query) - keeping the
    # uniform expectations below and failing if the halving regresses.
    def _agg(name: str) -> str:
        value = 1.0 if name in _HALF_SCALE_METRICS else 0.5
        return f"baseline_name,{name}\ndeepscholar_base,{value}\n"

    def _query(name: str) -> str:
        lo, hi = ("0.5", "1.5") if name in _HALF_SCALE_METRICS else ("0.25", "0.75")
        return f"folder_path,{name}\n/workspace/generation/0,{lo}\n/workspace/generation/1,{hi}\n"

    aggregate_files = {f"{name}/aggregated_results.csv": _agg(name) for name in PRIMARY_METRICS}
    query_files = {f"{name}/deepscholar_base.csv": _query(name) for name in PRIMARY_METRICS}

    with mock.patch.object(
        evaluator,
        "_read_dir",
        new=mock.AsyncMock(side_effect=[aggregate_files, query_files]),
    ):
        result = asyncio.run(
            evaluator._extract_results(
                mock.Mock(),
                "raw output",
                0,
                n_success=2,
                n_total=3,
                generation_ok=True,
                requested_metrics=PRIMARY_METRICS,
            )
        )

    assert result.metrics[metric] == pytest.approx(0.5)
    assert result.metrics[f"{metric}_fixed"] == pytest.approx(1 / 3)
    assert next(iter(result.metrics)) == "geomean_fixed"
    assert result.metrics["geomean"] == pytest.approx(0.5)
    assert result.metrics["geomean_fixed"] == pytest.approx(1 / 3)
    assert result.metadata["queries_requested"] == 3
    assert result.metadata["queries_generated"] == 2
    assert result.metadata["queries_scored"] == 2
    assert result.metadata["num_tasks"] == 3
    assert result.metadata["generation_complete"] is False
