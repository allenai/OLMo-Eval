"""Resolution of lazy image references carried on ``LMRequest.images``.

An images entry may be a PIL image, a filesystem path (``str`` / ``Path``), or a
zero-argument callable returning a PIL image. Tasks attach the lazy forms so the
runner never decodes pixels in the parent process or pickles them through the
worker queue; providers resolve entries with :func:`resolve_images` right before
preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_image(entry: Any) -> Any:
    """Resolve one images entry to a PIL image, or a list of them.

    A callable may return a whole image list (the multi-image tasks attach one
    lazy loader for the example's list); its items are resolved recursively.
    Returns ``None`` unchanged.
    """
    if entry is None:
        return None
    if callable(entry):
        entry = entry()
    if isinstance(entry, (list, tuple)):
        return [resolve_image(item) for item in entry]
    if isinstance(entry, (str, Path)):
        from PIL import Image

        entry = Image.open(entry)
    return entry


def resolve_images(images: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    """Resolve a request's images tuple, flattening lists and dropping empty slots.

    A benchmark with a variable image count per example (MMMU-Pro interleaves up
    to seven) attaches one lazy reference per slot, and the multi-image tasks
    attach a single lazy loader for the whole list. Unused slots resolve to
    ``None`` and must not reach the provider as holes in the image list.
    """
    if not images:
        return None
    resolved: list[Any] = []
    for entry in images:
        item = resolve_image(entry)
        if isinstance(item, (list, tuple)):
            resolved.extend(image for image in item if image is not None)
        elif item is not None:
            resolved.append(item)
    return tuple(resolved) or None
