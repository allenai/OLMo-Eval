"""Utilities for building derived sandbox images with swe-rex."""

from __future__ import annotations

import hashlib
import logging
import shlex
import shutil
import subprocess
import time
from typing import Any

from olmo_eval.common.config import get_infra_config

logger = logging.getLogger(__name__)

# Cache resolved image names to avoid redundant registry checks/builds
# when multiple executors share the same config.
_resolved_images: dict[str, str] = {}

# Pinned uv image used to bootstrap Python and install swe-rex in derived images
UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.7"

# Version bump this when changing the Dockerfile to invalidate cached images
SWEREX_IMAGE_VERSION = "20260505.1"


class _Deadline:
    """Runs subprocesses against a shared time budget, killing any that run past it."""

    def __init__(self, timeout: float | None, description: str) -> None:
        self._end = None if timeout is None else time.monotonic() + timeout
        self._timeout = timeout
        self._description = description

    def run(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        remaining = None if self._end is None else self._end - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"{self._description} exceeded {self._timeout:g}s")
        try:
            return subprocess.run(cmd, timeout=remaining, **kwargs)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"{self._description} exceeded {self._timeout:g}s during: {' '.join(cmd[:3])}"
            ) from e


def _remote_image_exists(container_runtime: str, image: str) -> bool:
    """Check if a remote image exists in a registry.

    Uses multiple methods since podman manifest inspect doesn't always
    work with private registries and credential helpers.
    """
    # skopeo (shipped with podman) supports credential helpers and only
    # fetches the manifest — no layer downloads.
    if shutil.which("skopeo"):
        result = subprocess.run(
            ["skopeo", "inspect", "--raw", f"docker://{image}"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True

    # gcloud can check GCP Artifact Registry directly
    if shutil.which("gcloud") and "pkg.dev" in image:
        result = subprocess.run(
            ["gcloud", "artifacts", "docker", "images", "describe", image, "--quiet"],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True

    # Last resort: container runtime manifest inspect
    try:
        result = subprocess.run(
            [container_runtime, "manifest", "inspect", image],
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def _swerex_tag_hash(base_image: str, dockerfile_extra: tuple[str, ...] = ()) -> str:
    """Deterministic tag for the derived image, from its content inputs."""
    extra_hash = ":".join(dockerfile_extra) if dockerfile_extra else ""
    hash_input = f"{base_image}:{UV_IMAGE}:{SWEREX_IMAGE_VERSION}:{extra_hash}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]


def remove_swerex_image(
    base_image: str,
    container_runtime: str = "docker",
    dockerfile_extra: tuple[str, ...] = (),
) -> None:
    """Remove a derived swe-rex image and its base image from local storage.

    Use this to free disk when many single-use images pass through one host.
    Registry copies are left alone, and failures are logged rather than raised.
    """
    tag_hash = _swerex_tag_hash(base_image, dockerfile_extra)
    resolved = _resolved_images.pop(tag_hash, None)
    names = {f"swerex-{tag_hash}:latest", base_image}
    if resolved:
        names.add(resolved)
    # One name per call, so a name that is already gone does not block the others.
    for name in sorted(names):
        try:
            result = subprocess.run(
                [container_runtime, "rmi", "--force", name], capture_output=True
            )
        except OSError as e:
            logger.debug(f"Could not remove image {name}: {e}")
            continue
        if result.returncode != 0:
            stderr = result.stderr.decode() if result.stderr else ""
            logger.debug(f"Could not remove image {name}: {stderr.strip()}")


def get_swerex_image(
    base_image: str,
    container_runtime: str = "docker",
    dockerfile_extra: tuple[str, ...] = (),
    require_registry: bool = False,
    timeout: float | None = None,
) -> str:
    """Build a derived image with Python and swe-rex pre-installed.

    Checks local cache first, then registry (if configured), then builds and pushes.
    Results are cached by content hash so repeated calls for the same image are free.

    Args:
        base_image: The base container image.
        container_runtime: Container runtime (docker or podman).
        dockerfile_extra: Additional Dockerfile commands to inject.
        require_registry: If True, requires SWEREX_REGISTRY and returns registry URL.
            Use for Modal which needs remote-accessible images.
        timeout: Seconds allowed for resolving the image, covering every pull,
            build and push it runs. None means no limit.

    Returns:
        The derived image name with swe-rex installed. If require_registry=True,
        returns the registry URL.

    Raises:
        ValueError: If require_registry=True but SWEREX_REGISTRY is not set.
        RuntimeError: If image build or push fails.
        TimeoutError: If resolving the image takes longer than ``timeout``.
    """
    tag_hash = _swerex_tag_hash(base_image, dockerfile_extra)

    if tag_hash in _resolved_images:
        logger.debug(f"Using cached resolution for swerex-{tag_hash}")
        return _resolved_images[tag_hash]

    result = _resolve_swerex_image(
        base_image,
        container_runtime,
        dockerfile_extra,
        require_registry,
        tag_hash,
        _Deadline(timeout, f"Resolving swerex image for {base_image}"),
    )
    _resolved_images[tag_hash] = result
    return result


def _resolve_swerex_image(
    base_image: str,
    container_runtime: str,
    dockerfile_extra: tuple[str, ...],
    require_registry: bool,
    tag_hash: str,
    deadline: _Deadline,
) -> str:
    """Core image resolution logic — called once per unique image hash."""
    config = get_infra_config()
    registry = config.swerex_registry

    if require_registry and not registry:
        raise ValueError(
            "SWEREX_REGISTRY required for Modal deployments with inject_swerex=True. "
            "Set to your registry URL (e.g., 'us-docker.pkg.dev/project/repo')."
        )

    # Local image name
    local_image = f"swerex-{tag_hash}:latest"

    # Registry image name (if registry configured)
    registry_image = f"{registry}/swerex-{tag_hash}:latest" if registry else None

    # Check if image exists locally
    result = deadline.run(
        [container_runtime, "image", "inspect", local_image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.debug(f"Using cached swerex image: {local_image}")
        if require_registry:
            assert registry_image is not None  # Guaranteed by earlier check
            # Check if the image already exists in the registry before pushing
            if _remote_image_exists(container_runtime, registry_image):
                logger.debug(f"Registry image already exists: {registry_image}")
                return registry_image
            # Not in registry yet — tag and push
            deadline.run(
                [container_runtime, "tag", local_image, registry_image],
                capture_output=True,
            )
            push_result = deadline.run(
                [container_runtime, "push", registry_image], capture_output=True
            )
            if push_result.returncode != 0:
                stderr = push_result.stderr.decode() if push_result.stderr else ""
                raise RuntimeError(f"Failed to push to registry: {stderr}")
            logger.debug(f"Pushed swerex image to registry: {registry_image}")
            return registry_image
        return local_image

    logger.info(f"Local image {local_image} not found, checking registry...")

    # Try to use image from registry (if configured)
    if registry and registry_image:
        if require_registry:
            # For Modal: just verify the image exists in the registry without pulling.
            # Modal pulls directly from the registry, so a local copy is unnecessary.
            if _remote_image_exists(container_runtime, registry_image):
                logger.info(f"Registry image exists: {registry_image}")
                return registry_image
            logger.info(f"Registry image {registry_image} not found, will build and push")
        else:
            # For Docker/Podman: pull so we have a local copy
            result = deadline.run(
                [container_runtime, "pull", registry_image],
                capture_output=True,
            )
            if result.returncode == 0:
                deadline.run(
                    [container_runtime, "tag", registry_image, local_image],
                    capture_output=True,
                )
                logger.info(f"Pulled swerex image from registry: {registry_image}")
                return local_image
            stderr = result.stderr.decode() if result.stderr else "unknown error"
            logger.warning(f"Registry pull failed for {registry_image}: {stderr}")

    # Build the image with Python (via uv venv), swe-rex, curl, and git.
    # Seed pip into the venv because Modal adds a small builder layer on top of
    # registry images and expects `python -m pip` to work inside the image.
    logger.info(f"Building swerex image from {base_image}...")

    extra_lines = "\n".join(dockerfile_extra) if dockerfile_extra else ""

    dockerfile = f"""\
FROM {base_image}
USER root
# Disable apt sandboxing to avoid setgroups/setegid errors in rootless containers
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-disable-sandbox
RUN apt-get update && \\
    apt-get install -y --no-install-recommends curl git ca-certificates && \\
    rm -rf /var/lib/apt/lists/*
COPY --from={UV_IMAGE} /uv /uvx /usr/local/bin/
RUN uv venv /root/venv --python 3.12 --seed && \\
    uv pip install --python /root/venv/bin/python --no-cache-dir swe-rex
{extra_lines}
ENV VIRTUAL_ENV="/root/venv"
ENV PATH="/root/venv/bin:$PATH"
"""

    result = deadline.run(
        [container_runtime, "build", "-t", local_image, "-"],
        input=dockerfile.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"Failed to build swerex image: {stderr}")

    logger.info(f"Built swerex image: {local_image}")

    if registry and registry_image:
        logger.debug(f"Pushing swerex image to registry: {registry_image}")
        deadline.run([container_runtime, "tag", local_image, registry_image], capture_output=True)
        push_result = deadline.run([container_runtime, "push", registry_image], capture_output=True)
        if push_result.returncode == 0:
            logger.debug(f"Pushed swerex image to registry: {registry_image}")
            if require_registry:
                return registry_image
        else:
            stderr = push_result.stderr.decode() if push_result.stderr else ""
            if require_registry:
                raise RuntimeError(f"Failed to push to registry: {stderr}")
            logger.warning(f"Failed to push to registry (using local image): {stderr}")

    # When require_registry=True, we've already returned above on success or raised on failure
    # So reaching here with require_registry=True means something unexpected happened
    if require_registry:
        raise RuntimeError("Failed to push image to registry (unexpected state)")

    return local_image


def dependencies_to_dockerfile_extra(dependencies: tuple[str, ...]) -> tuple[str, ...]:
    """Convert package specs to Dockerfile RUN commands for sandbox images.

    Installs into the /root/venv created by get_swerex_image() using uv.

    Args:
        dependencies: Package specs (e.g., ("numpy", "pandas>=2.0")).

    Returns:
        Tuple of Dockerfile command strings, or empty tuple if no dependencies.
    """
    if not dependencies:
        return ()
    pkgs = " ".join(shlex.quote(dep) for dep in dependencies)
    return (f"RUN uv pip install --python /root/venv/bin/python --no-cache {pkgs}",)
