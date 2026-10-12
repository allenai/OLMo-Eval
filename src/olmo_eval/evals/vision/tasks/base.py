"""Shared mechanics for the vision benchmark tasks.

Every vision family caches its instances and builds a single-turn CHAT request
with images attached; only how instances are built and which images attach
differ. Subclasses implement ``_build_instances`` and, when
one image is not the right attachment, override ``_attach_images``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.evals.tasks.common.base import Task

__all__ = ["VisionTask"]


class VisionTask(Task):
    """Base class for the vision benchmark tasks."""

    #: Image decoding is needed to build instances, whichever provider runs them.
    dependencies = ["pillow"]

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            params = self.config.sampling_params
            if params is not None and params.num_samples > 1:
                # The metric families read one output per example, and mm_olmo scores a
                # single greedy sample; averaging or maxing samples would be neither.
                raise ValueError(
                    f"{self.config.name} scores one sample per example; "
                    "num_samples > 1 is not supported"
                )
            # `config.limit` is left to the runner, which draws a seeded random sample
            # exactly as it does for the text tasks.
            self._instances_cache = list(self._build_instances())
        yield from self._instances_cache

    @abstractmethod
    def _build_instances(self) -> Iterator[Instance]:
        """Yield all instances for ``self.config.split``."""
        ...

    def _attach_images(self, instance: Instance) -> tuple[Any, ...] | None:
        """The images to send with ``instance``; ``None`` sends a text-only request.

        Entries are lazy references — a path or a picklable zero-arg callable — so
        preparing requests never decodes pixels in the parent process; providers
        resolve them (``olmo_eval.common.images.resolve_images``) at inference time.
        """
        image = instance.metadata.get("image")
        if image is not None:
            return (image,)
        path = instance.metadata.get("image_path")
        return (str(path),) if path is not None else None

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=self._attach_images(instance),
        )
