"""Tests for the Tau2 external evaluation."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from parameterized import parameterized

from olmo_eval.evals.external.benchmarks.tau2.eval import (
    TAU2_COMMIT,
    TAU2_REPOSITORY,
    TAU2_RESULTS_FILE,
    TAU2_SUCCESS_TOLERANCE,
    Tau2Args,
    Tau2ExternalEval,
)


def _make_results(
    rewards_by_task: dict[str, list[float]],
    *,
    domain: str = "airline",
    termination_reasons: dict[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    trial_counts = {len(rewards) for rewards in rewards_by_task.values()}
    if len(trial_counts) != 1:
        raise ValueError("test fixture requires the same number of trials per task")
    num_trials = trial_counts.pop()
    simulations = []
    for task_id, rewards in rewards_by_task.items():
        for trial, reward in enumerate(rewards):
            simulations.append(
                {
                    "id": f"{task_id}-{trial}",
                    "task_id": task_id,
                    "trial": trial,
                    "reward_info": {
                        "reward": reward,
                        "reward_breakdown": {"DB": reward},
                        "reward_basis": ["DB"],
                    },
                    "termination_reason": (termination_reasons or {}).get(
                        (task_id, trial), "user_stop"
                    ),
                    "duration": 2.0,
                    "agent_cost": 0.1,
                    "user_cost": 0.2,
                }
            )
    return {
        "info": {
            "git_commit": TAU2_COMMIT,
            "num_trials": num_trials,
            "max_steps": 30,
            "max_errors": 10,
            "seed": 17,
            "environment_info": {"domain_name": domain},
            "agent_info": {"implementation": "llm_agent", "llm": "hosted_vllm/model"},
            "user_info": {"implementation": "user_simulator", "llm": "gpt-4o-mini"},
        },
        "tasks": [{"id": task_id} for task_id in rewards_by_task],
        "simulations": simulations,
    }


class FakeExecutor:
    def __init__(self, *, success: bool = True, output: str = "") -> None:
        self.result = SimpleNamespace(success=success, output=output)
        self.commands: list[str] = []

    async def execute_command(self, command: str, **_kwargs: Any) -> SimpleNamespace:
        self.commands.append(command)
        return self.result


class TestTau2Args(unittest.TestCase):
    def test_from_dict_parses_task_ids(self) -> None:
        args = Tau2Args.from_dict({"task_ids": "one, two", "num_trials": "2"})
        self.assertEqual(args.task_ids, ["one", "two"])
        self.assertEqual(args.num_trials, 2)

    @parameterized.expand(
        [
            ("domain", "banking"),
            ("num_trials", 0),
            ("max_steps", 0),
            ("max_concurrency", -1),
            ("max_tokens", 0),
            ("max_model_len", -1),
            ("num_tasks", 0),
            ("max_errors", -1),
            ("seed", -1),
            ("temperature", -0.1),
            ("user_temperature", -0.1),
        ]
    )
    def test_rejects_invalid_argument(self, name: str, value: Any) -> None:
        with self.assertRaises(ValueError):
            Tau2Args(**{name: value})

    @parameterized.expand([([],), ([""],), (["task", "task"],), (("task",),)])
    def test_rejects_invalid_task_ids(self, task_ids: Any) -> None:
        with self.assertRaises(ValueError):
            Tau2Args(task_ids=task_ids)


class TestTau2ResultValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Tau2ExternalEval()

    def _validate(
        self, data: dict[str, Any], *, num_trials: int = 2, domain: str = "airline"
    ) -> dict[str, list[dict[str, Any]]]:
        return self.evaluator._validate_result_inventory(
            data, Tau2Args(num_trials=num_trials, domain=domain)
        )

    def test_partial_rewards_do_not_become_successes(self) -> None:
        inventory = self._validate(_make_results({"task": [0.5, 0.5]}))
        self.assertEqual(
            self.evaluator._compute_pass_k_metrics(inventory, num_trials=2),
            {"pass^1": 0.0, "pass^2": 0.0},
        )

    def test_success_tolerance_matches_tau2(self) -> None:
        data = _make_results(
            {
                "task": [
                    1.0 - TAU2_SUCCESS_TOLERANCE,
                    1.0 + TAU2_SUCCESS_TOLERANCE,
                    1.0 - 2 * TAU2_SUCCESS_TOLERANCE,
                    1.0 + 2 * TAU2_SUCCESS_TOLERANCE,
                ]
            }
        )
        inventory = self._validate(data, num_trials=4)
        metrics = self.evaluator._compute_pass_k_metrics(inventory, num_trials=4)
        self.assertEqual(metrics["pass^1"], 0.5)
        self.assertAlmostEqual(metrics["pass^2"], 1 / 6)
        self.assertEqual(metrics["pass^3"], 0.0)
        self.assertEqual(metrics["pass^4"], 0.0)

    def test_pass_k_uses_each_tasks_exact_trials(self) -> None:
        data = _make_results({"one": [1.0, 1.0, 0.0], "two": [1.0, 0.0, 0.0]})
        inventory = self._validate(data, num_trials=3)
        metrics = self.evaluator._compute_pass_k_metrics(inventory, num_trials=3)
        self.assertEqual(metrics["pass^1"], 0.5)
        self.assertAlmostEqual(metrics["pass^2"], 1 / 6)
        self.assertEqual(metrics["pass^3"], 0.0)

    def test_rejects_duplicate_task_ids(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["tasks"].append({"id": "task"})
        with self.assertRaisesRegex(ValueError, "duplicate task IDs"):
            self._validate(data)

    def test_rejects_duplicate_simulation_ids(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"][1]["id"] = data["simulations"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate simulation ID"):
            self._validate(data)

    def test_rejects_duplicate_task_trial_slots(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"][1]["trial"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate task/trial slot"):
            self._validate(data)

    def test_rejects_missing_task_trial_slots(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"].pop()
        with self.assertRaisesRegex(ValueError, "missing task/trial slots"):
            self._validate(data)

    def test_rejects_unexpected_task_id(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"][0]["task_id"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "unexpected task ID"):
            self._validate(data)

    def test_rejects_out_of_range_trial(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"][1]["trial"] = 2
        with self.assertRaisesRegex(ValueError, "out of range"):
            self._validate(data)

    def test_rejects_nonfinite_reward(self) -> None:
        data = _make_results({"task": [1.0, math.nan]})
        with self.assertRaisesRegex(ValueError, "must be finite"):
            self._validate(data)

    @parameterized.expand(
        [
            ("git_commit", "other", "benchmark commit mismatch"),
            ("num_trials", 3, "num_trials mismatch"),
            ("max_steps", 31, "max_steps mismatch"),
        ]
    )
    def test_rejects_inconsistent_run_info(
        self, field: str, value: Any, expected_error: str
    ) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["info"][field] = value
        with self.assertRaisesRegex(ValueError, expected_error):
            self._validate(data)

    def test_rejects_inconsistent_domain(self) -> None:
        data = _make_results({"task": [1.0, 0.0]}, domain="retail")
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            self._validate(data)

    def test_rejects_inconsistent_seed_when_requested(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["info"]["seed"] = 18
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            self.evaluator._validate_result_inventory(data, Tau2Args(num_trials=2, seed=17))

    def test_rejects_inconsistent_max_errors_when_requested(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["info"]["max_errors"] = 11
        with self.assertRaisesRegex(ValueError, "max_errors mismatch"):
            self.evaluator._validate_result_inventory(data, Tau2Args(num_trials=2, max_errors=10))

    def test_rejects_unknown_termination_reason(self) -> None:
        data = _make_results({"task": [1.0, 0.0]})
        data["simulations"][0]["termination_reason"] = "mystery"
        with self.assertRaisesRegex(ValueError, "termination_reason.*invalid"):
            self._validate(data)

    def test_predictions_separate_success_rate_from_average_reward(self) -> None:
        data = _make_results(
            {"task": [0.5, 0.5]},
            termination_reasons={("task", 1): "agent_error"},
        )
        inventory = self._validate(data)
        prediction = self.evaluator._build_predictions(data["tasks"], inventory)[0]
        metrics = prediction["instance_metrics"]
        self.assertEqual(metrics["success_rate"]["external"], 0.0)
        self.assertEqual(metrics["average_reward"]["external"], 0.5)
        self.assertEqual(metrics["error_rate"]["external"], 0.5)
        self.assertEqual([trial["trial"] for trial in prediction["trials"]], [0, 1])


class TestTau2ResultExtraction(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.evaluator = Tau2ExternalEval()
        self.args = Tau2Args(domain="airline", num_trials=2, seed=17)
        self.data = _make_results(
            {"one": [1.0, 0.0], "two": [1.0, 1.0]},
            termination_reasons={("one", 1): "max_steps"},
        )

    async def test_extracts_one_exact_file_and_records_provenance(self) -> None:
        executor = FakeExecutor(output=json.dumps(self.data))
        with tempfile.TemporaryDirectory() as output_dir:
            result = await self.evaluator._extract_results(
                executor,
                raw_output="raw",
                exit_code=0,
                tau2_args=self.args,
                model_name="allenai/Olmo-3-7B-Instruct",
                output_dir=output_dir,
            )
            trajectories_path = Path(output_dir) / "tau2_trajectories.json"
            self.assertTrue(trajectories_path.is_file())
            self.assertEqual(json.loads(trajectories_path.read_text()), self.data)

        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.metrics, {"pass^1": 0.75, "pass^2": 0.5})
        self.assertEqual(
            executor.commands,
            [f"cat /workspace/tau2-bench/{TAU2_RESULTS_FILE}"],
        )
        self.assertEqual(result.metadata["benchmark_repository"], TAU2_REPOSITORY)
        self.assertEqual(result.metadata["benchmark_commit"], TAU2_COMMIT)
        self.assertEqual(result.metadata["model_name"], "allenai/Olmo-3-7B-Instruct")
        self.assertEqual(result.metadata["expected_simulations"], 4)
        self.assertEqual(result.metadata["observed_simulations"], 4)
        self.assertEqual(result.metadata["successful_simulations"], 3)
        self.assertEqual(result.metadata["termination_counts"], {"max_steps": 1, "user_stop": 3})

    async def test_nonzero_exit_fails_without_reading_results(self) -> None:
        executor = FakeExecutor(output=json.dumps(self.data))
        result = await self.evaluator._extract_results(
            executor,
            raw_output="raw",
            exit_code=2,
            tau2_args=self.args,
            model_name="model",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tau2 exited with code 2")
        self.assertEqual(executor.commands, [])

    async def test_metadata_uses_effective_tau2_defaults(self) -> None:
        data = copy.deepcopy(self.data)
        data["info"]["seed"] = 300
        data["info"]["max_errors"] = 10

        result = await self.evaluator._extract_results(
            FakeExecutor(output=json.dumps(data)),
            raw_output="raw",
            exit_code=0,
            tau2_args=Tau2Args(domain="airline", num_trials=2),
            model_name="model",
        )

        self.assertTrue(result.success)
        self.assertEqual(result.metadata["seed"], 300)
        self.assertEqual(result.metadata["max_errors"], 10)

    async def test_missing_results_file_fails(self) -> None:
        result = await self.evaluator._extract_results(
            FakeExecutor(success=False),
            raw_output="raw",
            exit_code=0,
            tau2_args=self.args,
            model_name="model",
        )
        self.assertFalse(result.success)
        self.assertIn("missing or unreadable", result.error or "")

    async def test_malformed_results_file_fails(self) -> None:
        result = await self.evaluator._extract_results(
            FakeExecutor(output="not-json"),
            raw_output="raw",
            exit_code=0,
            tau2_args=self.args,
            model_name="model",
        )
        self.assertFalse(result.success)
        self.assertIn("Invalid Tau2 results JSON", result.error or "")

    async def test_incomplete_results_file_fails(self) -> None:
        data = copy.deepcopy(self.data)
        data["simulations"].pop()
        result = await self.evaluator._extract_results(
            FakeExecutor(output=json.dumps(data)),
            raw_output="raw",
            exit_code=0,
            tau2_args=self.args,
            model_name="model",
        )
        self.assertFalse(result.success)
        self.assertIn("missing task/trial slots", result.error or "")


if __name__ == "__main__":
    unittest.main()
