"""Common secret handling for Beaker jobs.

Provides utilities to retrieve local secrets (Hugging Face token, Weights & Biases key)
and store them as user-scoped Beaker secrets.

The Hugging Face token is copied to Beaker only when the job needs authenticated
Hugging Face access: ``--hf-token`` was passed, a task declares ``HF_TOKEN`` in its
required secrets, or a model or dataset is gated or private. An existing
``<user>_HF_TOKEN`` secret in the workspace is always injected. See ``plan_hf_token``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from beaker import Beaker

log = logging.getLogger(__name__)

__all__ = [
    "COMMON_SECRET_NAMES",
    "HF_TOKEN_ENV",
    "HfAccessCheck",
    "HfTokenPlan",
    "beaker_token_secret_name",
    "check_hf_access",
    "get_local_hf_token",
    "get_local_wandb_api_key",
    "ensure_common_secrets",
    "ensure_task_secrets",
    "plan_hf_token",
    "secret_exists",
    "write_secret",
]

HF_TOKEN_ENV = "HF_TOKEN"
COMMON_SECRET_NAMES: frozenset[str] = frozenset({HF_TOKEN_ENV})


def get_local_hf_token() -> str | None:
    """Retrieve HuggingFace token from the local environment.

    Checks (in order):
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
    3. ~/.huggingface/token file (huggingface-cli login)
    4. ~/.cache/huggingface/token file (older location)

    Returns:
        HuggingFace token if found, None otherwise.
    """
    # Check environment variables first
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug("Found HF_TOKEN in environment")
        return token

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        log.debug("Found HUGGING_FACE_HUB_TOKEN in environment")
        return token

    # Check token files
    token_paths = [
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]

    for token_path in token_paths:
        if token_path.exists():
            try:
                token = token_path.read_text().strip()
                if token:
                    log.debug(f"Found HF token in {token_path}")
                    return token
            except Exception as e:
                log.warning(f"Could not read {token_path}: {e}")

    return None


def get_local_wandb_api_key() -> str | None:
    """Retrieve Weights & Biases API key from the local environment.

    Checks (in order):
    1. WANDB_API_KEY environment variable
    2. ~/.netrc file (wandb login)

    Returns:
        WANDB API key if found, None otherwise.
    """
    # Check environment variable first
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        log.debug("Found WANDB_API_KEY in environment")
        return api_key

    # Check netrc file
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            import netrc

            nrc = netrc.netrc(str(netrc_path))
            auth = nrc.authenticators("api.wandb.ai")
            if auth:
                # netrc returns (login, account, password) - API key is the password
                api_key = auth[2]
                if api_key:
                    log.debug("Found WANDB API key in ~/.netrc")
                    return api_key
        except Exception as e:
            log.warning(f"Could not read ~/.netrc for wandb credentials: {e}")

    return None


def _get_beaker_username(client: Beaker) -> str:
    """Get the current Beaker username.

    Args:
        client: Beaker client instance.

    Returns:
        The username of the authenticated Beaker account.
    """
    return client.user_name


def _write_secret_if_needed(
    client: Beaker,
    name: str,
    value: str,
    overwrite: bool,
) -> bool:
    """Write a secret to Beaker if it doesn't exist or overwrite is True.

    Args:
        client: Beaker client instance.
        name: Secret name.
        value: Secret value.
        overwrite: Whether to overwrite existing secrets.

    Returns:
        True if the secret was written, False if it already existed.
    """
    try:
        existing = client.secret.get(name)
        if existing and not overwrite:
            log.debug(f"Secret {name} already exists, skipping")
            return False
    except Exception:
        pass  # Secret doesn't exist

    client.secret.write(name, value)
    log.info(f"Wrote secret {name} to Beaker workspace")
    return True


def beaker_token_secret_name(username: str) -> str:
    """Name of the user-scoped secret holding a Beaker token for in-job status updates."""
    return f"{username}_BEAKER_TOKEN"


def secret_exists(client: Beaker, name: str, workspace: str | None = None) -> bool:
    """Return whether a Beaker secret exists in ``workspace`` (default: the client's)."""
    from beaker.exceptions import BeakerSecretNotFound

    ws = client.workspace.get(workspace) if workspace else None
    try:
        client.secret.get(name, workspace=ws)
    except BeakerSecretNotFound:
        return False
    return True


def write_secret(client: Beaker, name: str, value: str, workspace: str | None = None) -> None:
    """Write a Beaker secret to ``workspace`` (default: the client's)."""
    ws = client.workspace.get(workspace) if workspace else None
    client.secret.write(name, value, workspace=ws)
    log.info(f"Wrote secret {name} to Beaker workspace {workspace or ''}".rstrip())


# Hugging Face repo ids look like "org/name". Anything else (paths, URLs, API model
# names) is not checked.
_HF_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

RepoAccess = Literal["public", "gated", "private", "missing", "unknown"]


def looks_like_hf_repo_id(value: str) -> bool:
    """Return whether ``value`` looks like a Hugging Face repo id rather than a path."""
    value = value.removeprefix("hf://")
    return bool(_HF_REPO_ID.match(value)) and not Path(value).exists()


@dataclass(frozen=True)
class HfAccessCheck:
    """Result of checking whether Hugging Face repos need authenticated access.

    Attributes:
        needs_token: Repos that are gated or private, as "model org/name (gated)".
        unchecked: Repos whose status could not be determined (network error, timeout).
    """

    needs_token: list[str] = field(default_factory=list)
    unchecked: list[str] = field(default_factory=list)


def _repo_access(repo_id: str, repo_type: str, token: str | None, timeout: float) -> RepoAccess:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    api = HfApi()
    try:
        info = api.repo_info(repo_id, repo_type=repo_type, timeout=timeout, token=False)
    except HFValidationError:
        return "missing"
    except RepositoryNotFoundError:
        # Anonymous lookups cannot tell a private repo from a missing one.
        if not token:
            return "missing"
        try:
            api.repo_info(repo_id, repo_type=repo_type, timeout=timeout, token=token)
        except RepositoryNotFoundError:
            return "missing"
        except Exception:
            return "unknown"
        return "private"
    except Exception as e:
        log.debug(f"Could not check Hugging Face {repo_type} {repo_id}: {e}")
        return "unknown"
    if info.private:
        return "private"
    if info.gated:
        return "gated"
    return "public"


def check_hf_access(
    repos: Iterable[tuple[str, str]],
    token: str | None = None,
    timeout: float = 5.0,
) -> HfAccessCheck:
    """Check which Hugging Face repos are gated or private.

    Each repo is looked up anonymously first. A repo that is not found anonymously is
    looked up again with ``token`` (when given) to detect private repos. Values that do
    not look like repo ids are skipped.

    Args:
        repos: (repo_id, repo_type) pairs, repo_type "model" or "dataset". A leading
            "hf://" is stripped.
        token: Local Hugging Face token, used only to detect private repos.
        timeout: Per-request timeout in seconds.
    """
    unique = sorted(
        {
            (repo_id.removeprefix("hf://"), repo_type)
            for repo_id, repo_type in repos
            if looks_like_hf_repo_id(repo_id)
        }
    )
    if not unique:
        return HfAccessCheck()
    with ThreadPoolExecutor(max_workers=min(8, len(unique))) as pool:
        results = list(pool.map(lambda r: _repo_access(r[0], r[1], token, timeout), unique))
    check = HfAccessCheck()
    for (repo_id, repo_type), access in zip(unique, results, strict=True):
        if access in ("gated", "private"):
            check.needs_token.append(f"{repo_type} {repo_id} ({access})")
        elif access == "unknown":
            check.unchecked.append(f"{repo_type} {repo_id}")
    return check


@dataclass(frozen=True)
class HfTokenPlan:
    """What to do with the Hugging Face token for a launch.

    Attributes:
        secret_name: The user-scoped Beaker secret, e.g. "alice_HF_TOKEN".
        inject: Mount the secret as HF_TOKEN in the job.
        copy_local: Write the local token to the secret before launching.
        message: One-line explanation for the user.
        warning: True when ``message`` is a warning.
        error: Set when the job needs a token but none is available.
    """

    secret_name: str
    inject: bool
    copy_local: bool = False
    message: str = ""
    warning: bool = False
    error: str | None = None


def plan_hf_token(
    *,
    username: str,
    mode: bool | None,
    secret_present: bool | None,
    required: bool,
    access: HfAccessCheck,
    have_local_token: bool,
) -> HfTokenPlan:
    """Decide whether to inject and copy the Hugging Face token.

    Args:
        username: Beaker username for the secret name.
        mode: ``--hf-token`` (True), ``--no-hf-token`` (False), or auto (None).
        secret_present: Whether ``<user>_HF_TOKEN`` exists in the workspace. None when
            not checked (dry run).
        required: A task, model, harness or external eval declares HF_TOKEN.
        access: Gated or private repos found by ``check_hf_access``.
        have_local_token: Whether a local Hugging Face token was found.
    """
    name = f"{username}_{HF_TOKEN_ENV}"
    reasons: list[str] = []
    if mode:
        reasons.append("--hf-token was passed")
    if required:
        reasons.append("a task or model declares HF_TOKEN in required_secrets")
    reasons.extend(access.needs_token)
    needed = bool(reasons)

    if mode is False:
        if needed:
            return HfTokenPlan(
                name,
                inject=False,
                message=f"--no-hf-token: not injecting HF_TOKEN, but {'; '.join(reasons[:3])}",
                warning=True,
            )
        return HfTokenPlan(name, inject=False, message="--no-hf-token: not injecting HF_TOKEN")

    if secret_present:
        return HfTokenPlan(name, inject=True, message=f"HF_TOKEN from existing secret {name}")

    if needed:
        why = "; ".join(reasons[:3])
        if secret_present is None:
            # Dry run: the workspace is not checked.
            note = "" if have_local_token else " (no local token found to copy)"
            return HfTokenPlan(
                name,
                inject=True,
                message=(
                    f"HF_TOKEN from secret {name}, copying the local token if the secret "
                    f"is missing{note} ({why})"
                ),
                warning=not have_local_token,
            )
        if have_local_token:
            return HfTokenPlan(
                name,
                inject=True,
                copy_local=True,
                message=f"Copying local Hugging Face token to secret {name} ({why})",
            )
        return HfTokenPlan(
            name,
            inject=False,
            error=(
                f"The job needs a Hugging Face token ({why}), but no local token was found "
                f"and secret {name} does not exist. Set HF_TOKEN or run `hf auth login`, "
                f"or create the secret with: beaker secret write {name} <token>"
            ),
        )

    if access.unchecked:
        message = (
            f"Could not check {', '.join(access.unchecked[:3])} on Hugging Face; not copying "
            "your HF token. Pass --hf-token if a gated or private model or dataset is used."
        )
        warning = True
    else:
        message = "HF_TOKEN not needed: models and datasets are public"
        warning = False
    if secret_present is None:
        message += f" (an existing secret {name} would still be injected)"
    return HfTokenPlan(name, inject=False, message=message, warning=warning)


def ensure_common_secrets(
    workspace: str,
) -> list[tuple[str, str]]:
    """Inject common optional secrets (WANDB_API_KEY) that already exist in the workspace.

    The user's ``<username>_WANDB_API_KEY`` secret is injected only if it is already in
    the workspace. It is never copied there from the local machine: nothing in olmo-eval
    reads it, and a personal key in a shared workspace is readable by every writer. A task
    that needs it declares WANDB_API_KEY in ``required_secrets`` instead.

    HF_TOKEN is handled separately by ``plan_hf_token``.

    Args:
        workspace: Beaker workspace the job runs in.

    Returns:
        (env_var_name, secret_name) tuples for secrets that exist, for example
        [("WANDB_API_KEY", "alice_WANDB_API_KEY")].
    """
    from beaker import Beaker

    client = Beaker.from_env(default_workspace=workspace)
    name = f"{_get_beaker_username(client)}_WANDB_API_KEY"
    try:
        found = secret_exists(client, name)
    except Exception as e:
        log.debug(f"Could not check for secret {name}: {e}")
        found = False
    return [("WANDB_API_KEY", name)] if found else []


def ensure_task_secrets(
    workspace: str,
    required_secrets: set[str],
) -> list[tuple[str, str]]:
    """Ensure task-required secrets exist in Beaker.

    Unlike ensure_common_secrets, this function DOES raise an error if
    any required secret is not found. Task-required secrets are mandatory
    for the evaluation to run correctly.

    Secrets are expected to be stored with a username prefix to prevent
    collisions between users in shared workspaces. For example, user "alice"
    requesting "S2_API_KEY" will look for secret "alice_S2_API_KEY".

    Args:
        workspace: Beaker workspace to check secrets in.
        required_secrets: Set of environment variable names that must exist
            as Beaker secrets.

    Returns:
        List of (env_var_name, secret_name) tuples.

    Raises:
        ValueError: If any required secret is not found in Beaker.
    """
    if not required_secrets:
        return []

    from beaker import Beaker
    from beaker.exceptions import BeakerSecretNotFound

    client = Beaker.from_env(default_workspace=workspace)
    username = _get_beaker_username(client)
    secrets: list[tuple[str, str]] = []
    missing: list[str] = []

    for env_var in sorted(required_secrets):
        secret_name = f"{username}_{env_var}"
        try:
            client.secret.get(secret_name)
            secrets.append((env_var, secret_name))
            log.debug(f"Found required secret {secret_name}")
        except BeakerSecretNotFound:
            missing.append(f"{env_var} (expected Beaker secret: {secret_name})")

    if missing:
        raise ValueError(
            "Missing required Beaker secrets:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nCreate these secrets with:\n"
            + "\n".join(f"  beaker secret write {username}_{s.split()[0]} <value>" for s in missing)
        )

    return secrets
