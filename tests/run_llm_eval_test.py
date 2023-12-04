import json
import os
import subprocess

from rjsonnet import evaluate_file

from llm_eval.common.testing import LLMEvalTestCase


class TestRunLLMEvalScript(LLMEvalTestCase):
    def test_script_with_config(self):
        config_file = self.PROJECT_ROOT / "configs" / "run_lm_eval_example.jsonnet"

        _ = subprocess.run(
            ["python", "-m", "llm_eval.run_lm_eval", "--config_file", config_file],
            capture_output=True,
            text=True,
        )

        config = json.loads(evaluate_file(str(config_file)))

        assert os.path.exists(config["metrics_file"])
        assert os.path.exists(config["full_output_file"])

        with open(config["metrics_file"]) as f:
            metrics = json.load(f)

        assert "metrics" in metrics
        assert len(metrics["metrics"]) == len(config["task"])

        os.remove(config["metrics_file"])
        os.remove(config["full_output_file"])

    def test_script_with_args(self):
        predictions_file = self.TEST_DIR / "predicions.jsonl"
        metrics_file = self.TEST_DIR / "metrics.jsonl"

        _ = subprocess.run(
            [
                "python",
                "-m",
                "llm_eval.run_lm_eval",
                "--model",
                "lm::pretrained=EleutherAI/pythia-160m,revision=step140000",
                "--task",
                "arc_challenge",
                "arc_easy",
                "--split",
                "validation",
                "--full_output_file",
                predictions_file,
                "--metrics_file",
                metrics_file,
                "--model_max_length",
                "2048",
                "--max_batch_tokens",
                "4096",
                "--num_recorded_inputs",
                "3",
                "--num_shots",
                "0",
                "--limit",
                "10",
            ],
            capture_output=True,
            text=True,
        )

        assert os.path.exists(str(metrics_file))
        assert os.path.exists(str(predictions_file))

        with open(metrics_file) as f:
            metrics = json.load(f)

        assert "metrics" in metrics
        assert len(metrics["metrics"]) == 2
