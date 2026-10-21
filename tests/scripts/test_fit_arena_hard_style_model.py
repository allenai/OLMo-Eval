"""Exercise the Arena-Hard style-model fit on small synthetic data without network access."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from olmo_eval.evals.tasks.arena_hard import HARD_PROMPT, StyleModel, sigmoid

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/internal/fit_arena_hard_style_model.py"


@pytest.fixture
def fitter():
    spec = importlib.util.spec_from_file_location("fit_arena_hard_style_model", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def answer_record(model: str, uid: str, answer: str, tokens: int) -> dict:
    return {
        "uid": uid,
        "model": model,
        "messages": [{"role": "assistant", "content": {"answer": answer}}],
        "metadata": {
            "token_len": tokens,
            "header_count": {"h1": 0},
            "list_count": {"ordered": 0, "unordered": 0},
            "bold_count": {"**": 0, "__": 0},
        },
    }


# numpy on macOS's Accelerate BLAS flags overflow in finite matmuls; the fit is exact.
@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_fit_joint_recovers_strengths_and_style_coefficients(fitter):
    rng = np.random.default_rng(0)
    strengths = {"model-a": 1.0, "model-b": -0.5}
    coefs = np.array([1.2, 0.1, -0.3, 0.05])
    models = [name for name in strengths for _ in range(200)]
    features = rng.normal(scale=[0.4, 0.01, 0.02, 0.02], size=(len(models), 4))
    standardized = (features - features.mean(axis=0)) / features.std(axis=0, ddof=1)
    logits = np.array([strengths[name] for name in models]) + standardized @ coefs
    # Expected outcomes stand in for sampled ones, so the fit recovers the truth exactly.
    outcomes = 1.0 / (1.0 + np.exp(-logits))

    win_rates, style_model = fitter.fit_joint(models, features, outcomes)

    for name, strength in strengths.items():
        assert win_rates[name] == pytest.approx(sigmoid(strength), abs=1e-6)
    assert style_model.coefs == pytest.approx(tuple(coefs), abs=1e-6)
    assert style_model.means == pytest.approx(tuple(features.mean(axis=0)))


def test_load_battles_skips_judgments_without_a_known_verdict(fitter, tmp_path, capsys):
    baseline = HARD_PROMPT.baseline_model
    write_jsonl(
        tmp_path / "model_answer" / f"{baseline}.jsonl",
        [answer_record(baseline, uid, "base", tokens=10) for uid in ("q1", "q2", "q3")],
    )
    write_jsonl(
        tmp_path / "model_answer" / "model-a.jsonl",
        [answer_record("org/model-a", uid, "mine", tokens=30) for uid in ("q1", "q2", "q3")],
    )

    def judgment(uid: str, games: list, category: str = "hard_prompt") -> dict:
        return {"uid": uid, "category": category, "model": "org/model-a", "games": games}

    write_jsonl(
        tmp_path / "model_judgment" / "gpt-4.1" / "model-a.jsonl",
        [
            judgment("q1", [{"score": "A>B"}, {"score": "A>>B"}]),
            judgment("q2", [None, {"score": "A>B"}]),
            judgment("q3", [{"score": "A>>>B"}, {"score": "A>B"}]),
            judgment("q1", [{"score": "A>B"}, {"score": "A>B"}], category="creative_writing"),
        ],
    )

    style = fitter.load_style(tmp_path)
    models, features, outcomes = fitter.load_battles(tmp_path, HARD_PROMPT, style)

    assert models == ["model-a"] * 4
    assert outcomes.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert features[0, 0] == pytest.approx((30 - 10) / (30 + 10))
    assert "{'no verdict': 1, 'unknown verdict': 1}" in capsys.readouterr().out


def test_style_model_literal_round_trips(fitter):
    style_model = StyleModel(
        means=(0.0318839, 0.00184698), stds=(0.356366, 0.00459669), coefs=(1.18056, -0.1984)
    )
    literal = fitter._format_style_model(style_model)
    assert eval(literal, {"StyleModel": StyleModel}) == style_model
