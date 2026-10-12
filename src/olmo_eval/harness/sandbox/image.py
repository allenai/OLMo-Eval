"""Utilities for building derived sandbox images with swe-rex."""

from __future__ import annotations

import hashlib
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from olmo_eval.common.config import get_infra_config

logger = logging.getLogger(__name__)

# Cache resolved image names to avoid redundant registry checks/builds
# when multiple executors share the same config.
_resolved_images: dict[str, str] = {}

# Pinned uv image used to bootstrap Python and install swe-rex in derived images
UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.7"

# Version bump this when changing the Dockerfile to invalidate cached images
SWEREX_IMAGE_VERSION = "20261008.1"


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
    result = subprocess.run(
        [container_runtime, "manifest", "inspect", image],
        capture_output=True,
    )
    return result.returncode == 0


def _image_platform(container_runtime: str, image: str) -> str | None:
    """Return the ``os/arch`` an image was built for, pulling it if needed.

    A derived image has to be built for its base image's platform, which on a
    host of another architecture is not the platform a build defaults to.
    """
    inspect = [
        container_runtime,
        "image",
        "inspect",
        image,
        "--format",
        "{{.Os}}/{{.Architecture}}",
    ]
    result = subprocess.run(inspect, capture_output=True)
    if result.returncode != 0:
        pull = subprocess.run([container_runtime, "pull", image], capture_output=True)
        if pull.returncode != 0:
            stderr = pull.stderr.decode() if pull.stderr else ""
            logger.warning(f"Could not pull {image} to read its platform: {stderr.strip()}")
            return None
        result = subprocess.run(inspect, capture_output=True)
        if result.returncode != 0:
            return None
    platform = result.stdout.decode().strip()
    return platform if "/" in platform else None


def _image_user(container_runtime: str, image: str) -> str | None:
    """Return the user an image is configured to run as, if any."""
    result = subprocess.run(
        [container_runtime, "image", "inspect", image, "--format", "{{.Config.User}}"],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    user = result.stdout.decode().strip()
    return user or None


def _image_exists(container_runtime: str, image: str) -> bool:
    """Return whether an image is present in the local store."""
    result = subprocess.run([container_runtime, "image", "inspect", image], capture_output=True)
    return result.returncode == 0


def build_context_hash(context_dir: Path) -> str:
    """Hash a build context's file names and contents.

    The hash names the image built from the context, so a changed Dockerfile or
    data file produces a new image while an unchanged context reuses the old one.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(context_dir).rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(context_dir)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def build_task_image(
    context_dir: Path,
    container_runtime: str,
    name: str,
    timeout: float | None = None,
) -> str:
    """Build an image from a directory holding a Dockerfile, reusing a cached one.

    Args:
        context_dir: Directory with the Dockerfile and build context.
        container_runtime: Container runtime (docker or podman).
        name: Short name for the image; the tag carries the context hash.
        timeout: Seconds to allow the build before failing.

    Returns:
        The built image's name and tag.

    Raises:
        RuntimeError: If the build fails or exceeds the timeout.
    """
    context_dir = Path(context_dir)
    if not (context_dir / "Dockerfile").is_file():
        raise RuntimeError(f"No Dockerfile in build context {context_dir}")
    image = f"{name}:{build_context_hash(context_dir)}"
    if _image_exists(container_runtime, image):
        logger.debug(f"Using cached task image: {image}")
        return image

    logger.info(f"Building task image {image} from {context_dir}...")
    try:
        result = subprocess.run(
            [container_runtime, "build", "-t", image, str(context_dir)],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Building {image} exceeded {timeout}s") from e
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        raise RuntimeError(f"Failed to build {image}: {stderr[-4000:]}")
    logger.info(f"Built task image: {image}")
    return image


def get_swerex_image(
    base_image: str,
    container_runtime: str = "docker",
    dockerfile_extra: tuple[str, ...] = (),
    require_registry: bool = False,
    pristine: bool = False,
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
        pristine: If True, leave the base image's packages, PATH and environment
            untouched so commands run in the task's own environment. The swe-rex
            server is installed into a private virtualenv and exposed through a
            single launcher symlink. Use for benchmarks whose tasks depend on the
            exact contents of their image.

    Returns:
        The derived image name with swe-rex installed. If require_registry=True,
        returns the registry URL.

    Raises:
        ValueError: If require_registry=True but SWEREX_REGISTRY is not set.
        RuntimeError: If image build or push fails.
    """
    # Deterministic tag from content inputs
    extra_hash = ":".join(dockerfile_extra) if dockerfile_extra else ""
    hash_input = f"{base_image}:{UV_IMAGE}:{SWEREX_IMAGE_VERSION}:{extra_hash}"
    if pristine:
        hash_input += ":pristine"
    tag_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    if tag_hash in _resolved_images:
        logger.debug(f"Using cached resolution for swerex-{tag_hash}")
        return _resolved_images[tag_hash]

    result = _resolve_swerex_image(
        base_image, container_runtime, dockerfile_extra, require_registry, tag_hash, pristine
    )
    _resolved_images[tag_hash] = result
    return result


#: Where the pristine variant keeps the swe-rex virtualenv. It sits outside any
#: home directory so a base image's non-root user can run the server from it.
PRISTINE_VENV = "/opt/swerex/venv"


def build_swerex_dockerfile(
    base_image: str,
    dockerfile_extra: tuple[str, ...] = (),
    pristine: bool = False,
    user: str | None = None,
) -> str:
    """Render the Dockerfile that layers swe-rex onto a base image.

    The default variant installs curl, git and certificates, creates a Python
    virtualenv and puts it first on PATH so ``python`` and ``pip`` resolve to it.
    The pristine variant only adds the virtualenv and a launcher symlink for the
    swe-rex server, so the image's own interpreters, packages and environment
    are what commands see, and it hands control back to the base image's user.

    Args:
        base_image: The base container image.
        dockerfile_extra: Additional Dockerfile commands to append.
        pristine: Whether to leave the base image's environment untouched.
        user: The base image's configured user, restored at the end of a
            pristine build when it is not root.

    Returns:
        The Dockerfile contents.
    """
    extra_lines = "\n".join(dockerfile_extra) if dockerfile_extra else ""

    if pristine:
        # swe-rex starts its server as ``swerex-remote`` found on PATH. A symlink in
        # the standard binary directories satisfies that lookup without putting the
        # virtualenv itself on PATH; the script's shebang points at the venv python.
        # uv is only needed to build the venv and is removed again so the image's
        # own tools are the ones commands find.
        user_line = f"USER {user}\n" if user and user not in ("root", "0") else ""
        return f"""\
FROM {base_image}
USER root
COPY --from={UV_IMAGE} /uv /usr/local/bin/swerex-uv
RUN swerex-uv venv {PRISTINE_VENV} --python 3.12 --seed && \\
    swerex-uv pip install --python {PRISTINE_VENV}/bin/python --no-cache-dir swe-rex && \\
    chmod -R a+rX /opt/swerex && \\
    ln -sf {PRISTINE_VENV}/bin/swerex-remote /usr/local/bin/swerex-remote && \\
    ln -sf {PRISTINE_VENV}/bin/swerex-remote /usr/bin/swerex-remote && \\
    rm /usr/local/bin/swerex-uv
{extra_lines}
{user_line}"""

    # Seed pip into the venv because Modal adds a small builder layer on top of
    # registry images and expects `python -m pip` to work inside the image.
    return f"""\
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


def _resolve_swerex_image(
    base_image: str,
    container_runtime: str,
    dockerfile_extra: tuple[str, ...],
    require_registry: bool,
    tag_hash: str,
    pristine: bool = False,
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
    result = subprocess.run(
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
            subprocess.run(
                [container_runtime, "tag", local_image, registry_image],
                capture_output=True,
            )
            push_result = subprocess.run(
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
            result = subprocess.run(
                [container_runtime, "pull", registry_image],
                capture_output=True,
            )
            if result.returncode == 0:
                subprocess.run(
                    [container_runtime, "tag", registry_image, local_image],
                    capture_output=True,
                )
                logger.info(f"Pulled swerex image from registry: {registry_image}")
                return local_image
            stderr = result.stderr.decode() if result.stderr else "unknown error"
            logger.warning(f"Registry pull failed for {registry_image}: {stderr}")

    logger.info(f"Building swerex image from {base_image}...")

    platform = _image_platform(container_runtime, base_image)
    user = _image_user(container_runtime, base_image) if pristine else None
    dockerfile = build_swerex_dockerfile(base_image, dockerfile_extra, pristine, user=user)

    build_cmd = [container_runtime, "build", "-t", local_image]
    if platform:
        build_cmd.append(f"--platform={platform}")
    build_cmd.append("-")
    result = subprocess.run(build_cmd, input=dockerfile.encode(), capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"Failed to build swerex image: {stderr}")

    logger.info(f"Built swerex image: {local_image}")

    if registry and registry_image:
        logger.debug(f"Pushing swerex image to registry: {registry_image}")
        subprocess.run([container_runtime, "tag", local_image, registry_image], capture_output=True)
        push_result = subprocess.run(
            [container_runtime, "push", registry_image], capture_output=True
        )
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
