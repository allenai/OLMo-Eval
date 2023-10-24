from catwalk.model import Model
from catwalk.models import MODELS

from llm_eval.common.testing import LLMEvalTestCase
from llm_eval.steps.run_catwalk import (
    ConstructCatwalkModel,
    ConstructTaskDict,
    PredictAndCalculateMetricsStep,
)


class TestConstructTaskDict(LLMEvalTestCase):
    def test_step(self):
        step = ConstructTaskDict()
        result = step.run(task_name="arc_easy", task_rename="my-arc-task", my_kwarg=23)
        assert isinstance(result, dict)
        assert result["name"] == "my-arc-task"
        assert "task_obj" in result
        assert result["my_kwarg"] == 23
        assert "unconditioned_prompt" in result

    def test_step_updated_metrics(self):
        step = ConstructTaskDict()
        result = step.run(
            task_name="arc_easy", task_rename="my-arc-task", primary_metric="my-metric-name"
        )
        assert isinstance(result, dict)
        assert result["name"] == "my-arc-task"
        assert "task_obj" in result
        assert (
            result["task_obj"].metrics["rc_metrics"].keywords["primary_metric"] == "my-metric-name"
        )

    def test_step_updated_prompt(self):
        step = ConstructTaskDict()
        result = step.run(task_name="arc_easy", unconditioned_prompt="What is the answer?")
        assert isinstance(result, dict)
        assert result["name"] == "arc_easy"
        assert "unconditioned_prompt" in result
        assert result["unconditioned_prompt"] == "What is the answer?"

    def test_step_data_files(self):
        step = ConstructTaskDict()
        result = step.run(
            task_name="ppl_custom",
            task_rename="ppl_c4_100_domains",
            keep_instance_fields=["orig_file_name", "source", "subdomain"],
            files=["path_to_data/val"],
        )
        assert isinstance(result, dict)
        assert "files" in result
        assert "keep_instance_fields" in result


class TestConstructCatwalkModel(LLMEvalTestCase):
    def test_step(self):
        step = ConstructCatwalkModel()
        result = step.run(model_path="sshleifer/tiny-gpt2")
        assert isinstance(result, Model)
        assert result.pretrained_model_name_or_path == "sshleifer/tiny-gpt2"
        assert "lm::pretrained=sshleifer-tiny-gpt2" in MODELS


class TestPredictAndCalculateMetricsStep(LLMEvalTestCase):
    def test_step(self):
        model = ConstructCatwalkModel().run(model_path="sshleifer/tiny-gpt2")
        task_dict = ConstructTaskDict().run(task_name="boolq")

        step = PredictAndCalculateMetricsStep()
        num_instances = 3
        result = step.run(model=model, task_dict=task_dict, model_max_length=5, limit=num_instances)

        assert isinstance(result, dict)

        for key in [
            "task",
            "task_options",
            "model_kwargs",
            "metrics",
            "num_instances",
            "processing_time",
            "instance_predictions",
        ]:
            assert key in result

        assert len(result["instance_predictions"]) == num_instances == result["num_instances"]
