"""CI-safe tests that image-QA / dense-caption tasks attach the image to the request.

No data files, model, GPU, or network: a tiny in-memory PIL image stands in for
the real dataset image, and tasks are constructed with a minimal ``TaskConfig``.
These guard ``format_request`` populating ``LMRequest.images`` (and leaving it
``None`` for imageless instances, so text tasks are unaffected).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from olmo_eval.common.types import Instance, RequestType
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.vision.tasks.image_qa import ImageQATask

Image = pytest.importorskip("PIL.Image")


class _DummyImageQATask(ImageQATask):
    """Minimal concrete ImageQATask (no data) for exercising format_request."""

    def _build_instances(self) -> Iterator[Instance]:
        return iter(())


def _tiny_image():
    return Image.new("RGB", (8, 8))


def test_image_qa_format_request_attaches_pil_image():
    task = _DummyImageQATask(TaskConfig(name="dummy_image_qa"))
    img = _tiny_image()
    instance = Instance(question="chart_qa: What is the value?", metadata={"image": img})

    request = task.format_request(instance)

    assert request.request_type == RequestType.CHAT
    assert request.images == (img,)
    # The message content stays the plain question text (prompt parity unaffected).
    assert request.messages[0]["content"] == "chart_qa: What is the value?"


def test_image_qa_format_request_attaches_path_image(tmp_path):
    task = _DummyImageQATask(TaskConfig(name="dummy_image_qa"))
    img_path = tmp_path / "img.png"
    _tiny_image().save(img_path)
    instance = Instance(question="vqa2: what color?", metadata={"image_path": str(img_path)})

    request = task.format_request(instance)

    assert request.images is not None
    assert len(request.images) == 1


def test_image_qa_format_request_no_image_is_none():
    task = _DummyImageQATask(TaskConfig(name="dummy_image_qa"))
    instance = Instance(question="a text-only question", metadata={})

    request = task.format_request(instance)

    assert request.images is None


def test_dense_caption_format_request_attaches_image():
    from olmo_eval.evals.vision.benchmarks.dense_caption import DenseCaptionEval

    task = DenseCaptionEval(TaskConfig(name="dense_caption"))
    img = _tiny_image()
    instance = Instance(question="Describe this image.", metadata={"image": img})

    request = task.format_request(instance)

    assert request.images == (img,)
    assert request.messages[0]["content"] == "Describe this image."
