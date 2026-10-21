"""Option decorators for the run command."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

import click

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR, LOCAL_RESULT_DIR

F = TypeVar("F", bound=Callable[..., Any])


def default_output_dir() -> str:
    """The Beaker result dir on a Beaker job that has one, else the local result dir.

    Output under /tmp is lost when a Beaker container exits, so a job launched without
    ``-O`` (e.g. by gantry or a wrapper script) keeps its results in the result dataset.
    """
    if os.environ.get("BEAKER_JOB_ID") and os.path.isdir(BEAKER_RESULT_DIR):
        return BEAKER_RESULT_DIR
    return LOCAL_RESULT_DIR


def parallelism_options(func: F) -> F:  # noqa: UP047
    """Parallelism options."""

    @click.option(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    @click.option(
        "--parallelism",
        "-P",
        type=int,
        default=1,
        help="Number of model instances to run in parallel",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def upload_options(func: F) -> F:  # noqa: UP047
    """Dashboard upload options."""

    @click.option(
        "--upload/--no-upload",
        "upload",
        default=None,
        help=(
            "Upload results to the olmo-eval dashboard with your Google credentials "
            "(default: on; OLMO_EVAL_UPLOAD=0 turns it off)"
        ),
    )
    @click.option(
        "--api-url",
        default=None,
        help="Dashboard ingest service URL (default: $OLMO_EVAL_API_URL, else production)",
    )
    @click.option(
        "--tag",
        "tags",
        multiple=True,
        help="Label to attach to the uploaded run (repeatable)",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def experiment_options(func: F) -> F:  # noqa: UP047
    """Experiment metadata options."""

    @click.option(
        "--experiment-name",
        help="Human-readable experiment name (Beaker launches use the experiment name)",
    )
    @click.option(
        "--experiment-group",
        help="Experiment group for grouping related experiments",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def output_options(func: F) -> F:  # noqa: UP047
    """Output control options."""

    @click.option(
        "--output-dir",
        "-O",
        default=default_output_dir,
        help=(
            f"Output directory (default: {BEAKER_RESULT_DIR} on a Beaker job, "
            f"else {LOCAL_RESULT_DIR})"
        ),
    )
    @click.option(
        "--save-predictions/--no-save-predictions",
        "save_predictions",
        default=True,
        help="Save per-instance predictions to JSONL (default: enabled)",
    )
    @click.option(
        "--save-requests/--no-save-requests",
        "save_requests",
        default=True,
        help="Save per-instance requests to JSONL (default: enabled)",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Print config and exit without running",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def inspect_options(func: F) -> F:  # noqa: UP047
    """Debug and inspection options."""

    @click.option(
        "--debug-requests",
        is_flag=True,
        hidden=True,
        help="Log HTTP requests/responses to inference providers",
    )
    @click.option(
        "--debug-provider",
        is_flag=True,
        hidden=True,
        help="Enable verbose provider logging",
    )
    @click.option(
        "--inspect",
        is_flag=True,
        help="Enable all inspection flags (instance, formatted, tokens, request, response)",
    )
    @click.option(
        "--inspect-instance",
        is_flag=True,
        help="Print the first instance of each task before running evaluation",
    )
    @click.option(
        "--inspect-formatted",
        is_flag=True,
        help="Show formatted prompt (after template applied) before evaluation",
    )
    @click.option(
        "--inspect-tokens",
        is_flag=True,
        help="Show token array before evaluation",
    )
    @click.option(
        "--inspect-response",
        is_flag=True,
        help="Print the first response of each task after model generation",
    )
    @click.option(
        "--inspect-request",
        is_flag=True,
        help="Print the first request of each task before model generation",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)


def harness_options(func: F) -> F:  # noqa: UP047
    """Harness configuration options."""

    @click.option(
        "-H",
        "--harness",
        "harness_preset",
        type=str,
        default=None,
        help="Harness preset name (e.g., 'search') for tool/prompt configuration",
    )
    @click.option(
        "--harness-config",
        type=click.Path(exists=True),
        default=None,
        help="Path to harness config YAML/JSON file",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return cast(F, wrapper)
