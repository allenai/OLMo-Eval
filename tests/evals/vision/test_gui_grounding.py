"""ScreenSpot-v2 / -Pro, OSWorld-G and Point-Bench: scoring rules, metrics, prompts, instances."""

from __future__ import annotations

import json

import numpy as np
import pytest

import olmo_eval.evals  # noqa: F401  (registration side effect)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common.registry import get_task
from olmo_eval.evals.vision.benchmarks import os_world_g, point_bench, screen_spot
from olmo_eval.evals.vision.scoring.gui_grounding import (
    ClickAccuracyMetric,
    ClickScorer,
    MacroClickAccuracyMetric,
    PointCountRateMetric,
    point_in_polygon,
    point_in_rect,
    rect_from_xywh,
)
from olmo_eval.evals.vision.scoring.point_bench import (
    PointBenchAccuracyMetric,
    PointBenchScorer,
    point_in_mask,
    score_point_bench,
)

STAGE1 = {"prompt_templates": "none", "system_prompt_style": "style_and_length_v2"}


def _points(*pts: tuple[int, int]) -> str:
    """A points answer in thousandths of the image (html-v2)."""
    coords = " ".join(f"{i + 1} {x:03d} {y:03d}" for i, (x, y) in enumerate(pts))
    return f'<points coords="1 {coords}">x</points>'


def _response(instance: Instance, text: str, scorer) -> Response:
    output = LMOutput(text=text)
    score = scorer.score(instance, output)
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, prompt=""),
        outputs=[output],
        scores={scorer.name: score},
    )


class TestGeometry:
    def test_rect_from_xywh_and_inclusive_edges(self):
        rect = rect_from_xywh([10, 20, 30, 40])
        assert rect == {"type": "rect", "xyxy": [10.0, 20.0, 40.0, 60.0]}
        assert point_in_rect(10, 20, rect["xyxy"]) and point_in_rect(40, 60, rect["xyxy"])
        assert not point_in_rect(40.01, 30, rect["xyxy"])

    def test_polygon_handles_a_concave_shape(self):
        # an L: the notch at (15, 5) is outside
        l_shape = [0, 0, 10, 0, 10, 10, 20, 10, 20, 20, 0, 20]
        assert point_in_polygon(5, 5, l_shape)
        assert point_in_polygon(15, 15, l_shape)
        assert not point_in_polygon(15, 5, l_shape)

    def test_two_vertex_polygon_contains_nothing(self):
        # the form OSWorld-G's polygons take on the Hub mirror; why the official JSON is used
        assert not point_in_polygon(5, 5, [0, 0, 10, 10])


class TestClickScorer:
    def _instance(self, target, **meta):
        return Instance(
            question="q",
            gold_answer=None,
            metadata={"image_size": (1000, 500), "target": target, **meta},
        )

    def test_first_point_decides(self):
        inst = self._instance(rect_from_xywh([100, 100, 100, 100]))
        scorer = ClickScorer()
        assert scorer.score(inst, LMOutput(text=_points((150, 300)))) == 1.0  # (150, 150) px
        assert scorer.score(inst, LMOutput(text=_points((900, 900), (150, 300)))) == 0.0
        out = LMOutput(text=_points((150, 300), (900, 900)))
        assert scorer.score(inst, out) == 1.0
        assert out.metadata["click_result"] == {"n_points": 2}

    def test_no_point_is_wrong_except_for_a_refusal(self):
        scorer = ClickScorer()
        assert (
            scorer.score(self._instance(rect_from_xywh([0, 0, 10, 10])), LMOutput(text="none"))
            == 0.0
        )
        refusal = self._instance({"type": "refusal"})
        assert scorer.score(refusal, LMOutput(text="There are none.")) == 1.0
        assert scorer.score(refusal, LMOutput(text=_points((500, 500)))) == 0.0

    def test_polygon_target(self):
        inst = self._instance({"type": "polygon", "coords": [0, 0, 200, 0, 200, 200, 0, 200]})
        assert ClickScorer().score(inst, LMOutput(text=_points((100, 200)))) == 1.0  # (100, 100)
        assert ClickScorer().score(inst, LMOutput(text=_points((300, 200)))) == 0.0


class TestClickMetrics:
    def _responses(self):
        scorer = ClickScorer()
        rect = rect_from_xywh([0, 0, 500, 250])
        rows = [
            ("desktop", ["a"], _points((100, 100))),  # hit
            ("desktop", ["a", "b"], "nothing"),  # miss, zero points
            ("web", ["b"], _points((100, 100), (200, 200))),  # hit, two points
            ("web", [], _points((900, 900))),  # miss
            ("mobile", ["a"], _points((100, 100))),  # hit
        ]
        return [
            _response(
                Instance(
                    question="q",
                    gold_answer=None,
                    metadata={"image_size": (1000, 500), "target": rect, "kind": k, "groups": g},
                ),
                text,
                scorer,
            )
            for k, g, text in rows
        ], scorer

    def test_overall_group_macro_and_rates(self):
        responses, scorer = self._responses()
        assert ClickAccuracyMetric(name="accuracy", scorer=scorer).compute(responses) == 3 / 5
        desktop = ClickAccuracyMetric(name="d", scorer=scorer, field="kind", value="desktop")
        assert desktop.compute(responses) == 0.5
        assert desktop.compute_instance(responses[2]) is None
        group_b = ClickAccuracyMetric(name="b", scorer=scorer, field="groups", value="b")
        assert group_b.compute(responses) == 0.5  # list membership
        macro = MacroClickAccuracyMetric(
            name="average", scorer=scorer, field="kind", values=("desktop", "web", "mobile")
        )
        assert macro.compute(responses) == pytest.approx((0.5 + 0.5 + 1.0) / 3)
        zero = PointCountRateMetric(name="z", scorer=scorer, mode="zero")
        multi = PointCountRateMetric(name="m", scorer=scorer, mode="multiple")
        assert zero.compute(responses) == 1 / 5 and multi.compute(responses) == 1 / 5


class TestPointBenchRule:
    MASK = np.zeros((100, 200), dtype=bool)
    MASK[10:40, 10:60] = True  # region A
    MASK[60:90, 120:180] = True  # region B

    def test_non_counting_scores_only_the_first_point(self):
        first_in = score_point_bench([(20, 20), (199, 99)], self.MASK, (200, 100), "spatial", None)
        assert first_in["official"] and not first_in["all_points"]
        first_out = score_point_bench([(199, 99), (20, 20)], self.MASK, (200, 100), "spatial", None)
        assert not first_out["official"]

    def test_counting_needs_the_count_and_every_point_inside(self):
        assert score_point_bench([(20, 20), (130, 70)], self.MASK, (200, 100), "counting", 2)[
            "official"
        ]
        assert not score_point_bench([(20, 20)], self.MASK, (200, 100), "counting", 2)["official"]
        assert not score_point_bench(
            [(20, 20), (130, 70), (5, 5)], self.MASK, (200, 100), "counting", 3
        )["official"]

    def test_no_point_fails(self):
        result = score_point_bench([], self.MASK, (200, 100), "reasoning", None)
        assert not result["official"] and not result["all_points"] and result["n_points"] == 0

    def test_bounds_and_truncation(self):
        assert point_in_mask(10.9, 10.9, self.MASK, 200, 100)
        assert not point_in_mask(-0.5, 20, self.MASK, 200, 100)
        assert not point_in_mask(20, 100, self.MASK, 200, 100)
        # a mask smaller than its image never contains the extra pixels
        assert not point_in_mask(150, 20, self.MASK[:, :100], 200, 100)

    def test_scorer_and_category_average(self, tmp_path):
        from PIL import Image

        mask_path = tmp_path / "m.png"
        Image.fromarray((self.MASK * 255).astype(np.uint8)).save(mask_path)
        scorer = PointBenchScorer()
        categories = ("affordable", "spatial", "reasoning", "steerable", "counting")
        responses = []
        for i, category in enumerate(categories):
            meta = {
                "image_size": (200, 100),
                "mask_path": str(mask_path),
                "category": category,
                "count": 2 if category == "counting" else None,
            }
            # (100, 200)/1000 of 200x100 = pixel (20, 20): inside A; counting gets one point only
            text = _points((100, 200)) if i < 4 else _points((100, 200))
            responses.append(
                _response(Instance(question="q", gold_answer=None, metadata=meta), text, scorer)
            )
        avg = PointBenchAccuracyMetric(name="average", scorer=scorer)
        assert avg.compute(responses) == pytest.approx(4 / 5)
        counting = PointBenchAccuracyMetric(name="c", scorer=scorer, category="counting")
        assert counting.compute(responses) == 0.0
        loose = PointBenchAccuracyMetric(name="all", scorer=scorer, rule="all_points")
        assert loose.compute(responses) == 1.0  # mm_olmo's rule ignores the count

    def test_steerable_hint_matches_the_official_text(self):
        hint = point_bench.steerable_hint([{"x": 50.0, "y": 25.0}], 640, 480)
        assert hint == (
            "\nThe image contains an existing original point at pixel coordinates: [320.0, 120.0]."
            "\nThe query refers to this existing point."
        )
        assert point_bench.steerable_hint([], 640, 480) == ""


class TestPromptsAndRegistration:
    @pytest.mark.parametrize(
        "name", ["screen_spot_v2", "screen_spot_pro", "os_world_g", "os_world_g_refined"]
    )
    def test_gui_tasks_follow_the_prompt_family(self, name):
        assert get_task(name).gui_question("open the menu") == "Click open the menu"
        assert get_task(name, STAGE1).gui_question("open the menu") == "gui_point: open the menu"

    def test_point_bench_uses_the_pointing_style(self):
        assert (
            get_task("point_bench", STAGE1).apply_family_prefix("Point to x.")
            == "pointing: Point to x."
        )

    def test_pointing_suites(self):
        assert set(get_suite("pointing").tasks) == {
            "pixmo_points_eval",
            "sa_co_gold_subset",
            "point_bench",
        }
        assert set(get_suite("pointing_mp").tasks) == {
            "pixmo_points_eval_mp",
            "sa_co_gold_subset_mp",
            "sa_co_gold_point_4k_mp",
            "point_bench",
        }
        assert set(get_suite("gui_pointing").tasks) == {
            "screen_spot_v2",
            "screen_spot_pro",
            "os_world_g",
        }


def _png(path, size=(200, 100)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


class TestInstances:
    def test_screen_spot_v2(self, tmp_path, monkeypatch):
        root = tmp_path / "ScreenSpot-v2"
        for kind in ("desktop", "web", "mobile"):
            row = {
                "img_filename": f"{kind}.png",
                "bbox": [1, 2, 3, 4],
                "instruction": f"go {kind}",
                "data_type": "icon",
                "data_source": "x",
            }
            root.mkdir(parents=True, exist_ok=True)
            (root / f"screenspot_{kind}_v2.json").write_text(json.dumps([row]))
            _png(root / "screenspotv2_image" / f"{kind}.png")
        monkeypatch.setattr(screen_spot, "torch_datasets_dir", lambda: tmp_path)
        insts = list(get_task("screen_spot_v2", STAGE1).instances)
        assert [i.metadata["kind"] for i in insts] == ["desktop", "web", "mobile"]
        assert insts[0].question == "gui_point: go desktop"
        assert insts[0].metadata["target"] == {"type": "rect", "xyxy": [1.0, 2.0, 4.0, 6.0]}
        assert insts[0].metadata["image_size"] == (200, 100)
        assert insts[0].metadata["kind_type"] == "desktop_icon"

    def test_screen_spot_pro(self, tmp_path, monkeypatch):
        root = tmp_path / "ScreenSpotPro"
        (root / "annotations").mkdir(parents=True)
        row = {
            "img_filename": "a.png",
            "bbox": [1, 2, 3, 4],
            "img_size": [3840, 2160],
            "instruction": "do",
            "group": "Dev",
            "application": "vscode",
            "platform": "macos",
            "ui_type": "text",
            "id": "x",
        }
        (root / "annotations" / "vscode.json").write_text(json.dumps([row]))
        monkeypatch.setattr(screen_spot, "torch_datasets_dir", lambda: tmp_path)
        (inst,) = list(get_task("screen_spot_pro").instances)
        assert inst.question == "Click do" and inst.metadata["image_size"] == (3840, 2160)
        assert inst.metadata["target"]["xyxy"] == [1.0, 2.0, 3.0, 4.0]
        assert inst.metadata["group_type"] == "Dev_text"

    def test_os_world_g(self, tmp_path, monkeypatch):
        items = [
            {
                "id": "a-0",
                "image_path": "a.png",
                "image_size": [1920, 1080],
                "instruction": "open",
                "box_type": "bbox",
                "box_coordinates": [1, 2, 3, 4],
                "GUI_types": ["Button"],
            },
            {
                "id": "a-1",
                "image_path": "a.png",
                "image_size": [1920, 1080],
                "instruction": "drag",
                "box_type": "polygon",
                "box_coordinates": [0, 0, 5, 0, 5, 5],
                "GUI_types": [],
            },
            {
                "id": "a-2",
                "image_path": "a.png",
                "image_size": [1920, 1080],
                "instruction": "upload",
                "box_type": "refusal",
                "box_coordinates": [-1, -1, -1, -1],
                "GUI_types": [],
            },
        ]
        files = {
            "OSWorld-G.json": json.dumps(items),
            "classification_result.json": json.dumps(
                {"classified": {"element_recognition": [items[0]], "refusal": [items[2]]}}
            ),
        }
        for name, text in files.items():
            (tmp_path / name).write_text(text)
        _png(tmp_path / "images" / "a.png")
        monkeypatch.setattr(os_world_g, "_download", lambda name: str(tmp_path / name))
        insts = list(get_task("os_world_g").instances)
        assert [i.metadata["target"]["type"] for i in insts] == ["rect", "polygon", "refusal"]
        assert [i.metadata["has_target"] for i in insts] == ["yes", "yes", "no"]
        assert (
            insts[0].metadata["groups"] == ["element_recognition"]
            and insts[1].metadata["groups"] == []
        )

    def test_point_bench(self, tmp_path, monkeypatch):
        import huggingface_hub

        items = [
            {
                "image_filename": "s.jpg",
                "user_input": "Point to the cup.",
                "category": "steerable",
                "mask_filename": "s_mask.png",
                "timestamp": "t",
            },
            {
                "image_filename": "c.jpg",
                "user_input": "Point to all cups.",
                "category": "counting",
                "mask_filename": "c_mask.png",
                "timestamp": "t",
                "count": 3,
            },
        ]
        (tmp_path / "data.json").write_text(json.dumps(items))
        (tmp_path / "pixmo_metadata.csv").write_text(
            'image_filename,image_url,label,points\ns.jpg,u,cup,"[{""x"": 50, ""y"": 50}]"\n'
        )
        root = tmp_path / "point_arena"
        for item in items:
            _png(root / "selected_images" / item["category"] / item["image_filename"])
        monkeypatch.setattr(
            huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(tmp_path / name)
        )
        monkeypatch.setattr(point_bench, "torch_datasets_dir", lambda: tmp_path)
        steer, count = list(get_task("point_bench").instances)
        assert steer.question == (
            "Point to the cup.\nThe image contains an existing original point at pixel "
            "coordinates: [100.0, 50.0].\nThe query refers to this existing point."
        )
        assert count.question == "Point to all cups." and count.metadata["count"] == 3
        assert count.metadata["mask_path"].endswith("selected_masks/counting/c_mask.png")
