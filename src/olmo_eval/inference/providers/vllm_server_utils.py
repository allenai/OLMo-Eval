"""vLLM OpenAI-compatible server management for agent tasks.

This module provides utilities to start and manage a vLLM server that
exposes an OpenAI-compatible API, enabling agent tasks to use any
HuggingFace or local model via the OpenAI Agents SDK.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from functools import cache
from typing import TYPE_CHECKING, Any

from olmo_eval.common.debug import is_debug_provider
from olmo_eval.inference.gpu_planner import resolve_visible_devices

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default timeout for server startup
DEFAULT_STARTUP_TIMEOUT = 300  # 5 minutes for large models
DEFAULT_HEALTH_CHECK_INTERVAL = 2  # seconds
_VLLM_INTERNAL_PORT_RANGE = (20000, 32767)


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _find_free_internal_port(exclude: set[int] | None = None) -> int:
    """Find a local base port for vLLM internals outside the ephemeral range."""
    import random

    excluded = exclude or set()
    start, end = _VLLM_INTERNAL_PORT_RANGE
    for port in random.sample(range(start, end + 1), end - start + 1):
        if port in excluded:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port

    return _find_free_port()


ProgressCallback = Callable[[str], None]


@cache
def _chat_template_has_function_tag(
    model_name: str, trust_remote_code: bool = False, revision: str | None = None
) -> bool | None:
    """Return whether the model chat template contains '<function=', or None if unavailable."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision=revision
        )
        get_chat_template = getattr(tokenizer, "get_chat_template", None)
        if callable(get_chat_template):
            chat_template = get_chat_template()
        else:
            chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            return None
        return "<function=" in str(chat_template)
    except Exception:
        return None


def _wait_for_server(
    url: str,
    timeout: float = DEFAULT_STARTUP_TIMEOUT,
    interval: float = DEFAULT_HEALTH_CHECK_INTERVAL,
    process: subprocess.Popen | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, Exception | None, str | None]:
    """Wait for vLLM server to be ready.

    Args:
        url: Base URL of the server (e.g., "http://localhost:8000/v1")
        timeout: Maximum time to wait in seconds
        interval: Time between health checks in seconds
        process: Optional subprocess to monitor for early exit
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (success, last_error, process_output):
        - success: True if server is ready, False if timeout exceeded or process died
        - last_error: The last error encountered, or None
        - process_output: Captured output if process died early, or None
    """
    import urllib.error
    import urllib.request

    health_url = url.rstrip("/").replace("/v1", "") + "/health"
    models_url = url.rstrip("/") + "/models"

    start_time = time.time()
    last_error: Exception | None = None
    last_progress_time = 0.0
    progress_interval = 10.0  # Log progress every 10 seconds

    if progress_callback:
        progress_callback(f"Waiting for vLLM server (0/{int(timeout)}s)...")

    while time.time() - start_time < timeout:
        elapsed = time.time() - start_time

        # Log progress periodically
        if progress_callback and elapsed - last_progress_time >= progress_interval:
            progress_callback(f"Waiting for vLLM server ({int(elapsed)}/{int(timeout)}s)...")
            last_progress_time = elapsed

        # Check if the subprocess has died
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None:
                # Process exited - use communicate() to get all output
                output = None
                try:
                    stdout, _ = process.communicate(timeout=5)
                    if stdout:
                        output = stdout.decode("utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"Failed to read process output: {e}")
                logger.error(f"vLLM server process exited with code {exit_code}")
                if output:
                    logger.error(f"vLLM server output:\n{output}")
                return False, RuntimeError(f"Process exited with code {exit_code}"), output

        try:
            # Try health endpoint first
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    if progress_callback:
                        progress_callback(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        try:
            # Fallback to models endpoint
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if response.status == 200:
                    if progress_callback:
                        progress_callback(f"vLLM server ready at {url}")
                    return True, None, None
        except urllib.error.URLError as e:
            last_error = e
        except Exception as e:
            last_error = e

        time.sleep(interval)

    logger.error(f"vLLM server failed to start within {timeout}s. Last error: {last_error}")
    return False, last_error, None


def _infer_tool_call_parser(
    model_name: str, *, trust_remote_code: bool = False, revision: str | None = None
) -> str:
    """Infer the appropriate tool call parser based on model name.

    Args:
        model_name: Model name or path.

    Returns:
        Tool call parser name for vLLM.
    """
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "llama3_json"
    if "mistral" in model_lower:
        return "mistral"
    if "olmo" in model_lower:
        return "olmo3"

    name_suggests_xml = any(k in model_lower for k in ("qwen3-coder", "qwen3.5", "qwen3.6"))
    tag = _chat_template_has_function_tag(model_name, trust_remote_code, revision)
    if tag:
        return "qwen3_coder"
    if tag is None and name_suggests_xml:
        return "qwen3_coder"
    if tag is False and name_suggests_xml:
        logger.warning(
            "Model name %s suggests XML-style tool calls but its chat template does not contain "
            "'<function='; using 'hermes'; pass tool_call_parser explicitly to override.",
            model_name,
        )
    # Default for Qwen (e.g. Qwen3-8B) and other models
    return "hermes"


def _get_vllm_python() -> str:
    """Get the Python interpreter to use for vLLM server.

    Checks VLLM_PYTHON env var first (for isolated venv setups),
    falls back to current interpreter.

    Returns:
        Path to Python interpreter.
    """
    return os.environ.get("VLLM_PYTHON", sys.executable)


def _get_olmo3_tool_template_path() -> str:
    """Get path to bundled OLMo3 tool chat template."""
    import importlib.resources

    return str(
        importlib.resources.files("olmo_eval.inference.templates")
        / "tool_chat_template_olmo3.jinja"
    )


# Kwargs that are used for deployment/setup, not vLLM server CLI arguments
_NON_VLLM_KWARGS = frozenset({"patch_olmo3_tool_parser", "add_bos_token"})


def _apply_olmo3_tool_parser_patch() -> None:
    """Apply OLMo3 tool parser patch at runtime.

    This patches vLLM's olmo3_tool_parser.py to handle JSON content in string
    arguments. The patch is idempotent (checks if already applied).

    See: https://github.com/vllm-project/vllm/issues/32534
    """
    import os
    from pathlib import Path

    from olmo_eval.inference.patches.olmo3_tool_parser_patch import (
        find_olmo3_parser,
        patch_parser,
    )

    # Determine which venv to patch based on VLLM_PYTHON env var
    vllm_python = os.environ.get("VLLM_PYTHON")
    venv_path = str(Path(vllm_python).parent.parent) if vllm_python else None

    parser_path = find_olmo3_parser(venv_path)
    if parser_path:
        patch_parser(parser_path)


def _build_server_command(
    model_name: str,
    port: int,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    dtype: str = "auto",
    tokenizer: str | None = None,
    enable_auto_tool_choice: bool = False,
    tool_call_parser: str | None = None,
    enable_prefix_caching: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
    enable_lora: bool = False,
    lora_modules: list[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Build the vLLM server command.

    Args:
        model_name: HuggingFace model ID or local path
        port: Port to serve on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use
        dtype: Data type for model weights
        tokenizer: Custom tokenizer (defaults to model_name)
        enable_auto_tool_choice: Enable automatic tool choice
        tool_call_parser: Parser for tool calls (auto-detected if not specified
            when enable_auto_tool_choice is True)
        enable_prefix_caching: Enable prefix caching for faster inference (default: True)
        chat_template_kwargs: Extra kwargs for chat template (e.g., {"enable_thinking": false}).
            These are applied at request time by the provider for broad vLLM compatibility,
            rather than being forwarded as server CLI flags.
        enable_lora: Enable LoRA (Low-Rank Adaptation) support for model loading
        lora_modules: List of LoRA module paths to load (format: "module_name=/path/to/lora").
            Only used if enable_lora is True.
        **kwargs: Additional vLLM server arguments. May include patch_olmo3_tool_parser
            which controls whether to use the custom OLMo3 chat template.

    Returns:
        Command list for subprocess
    """
    import json

    # Extract deployment kwargs before filtering
    patch_olmo3_tool_parser = kwargs.get("patch_olmo3_tool_parser", False)

    # Apply OLMo3 tool parser patch at runtime if requested
    # This patches vLLM to handle JSON content in string arguments
    if patch_olmo3_tool_parser:
        _apply_olmo3_tool_parser_patch()

    # Use VLLM_PYTHON env var if set (for isolated venv setups)
    python_executable = _get_vllm_python()

    cmd = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]

    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])

    if max_model_len:
        cmd.extend(["--max-model-len", str(max_model_len)])

    # Tool calling support
    if enable_auto_tool_choice:
        cmd.append("--enable-auto-tool-choice")
        # Auto-detect parser if not specified
        parser = tool_call_parser or _infer_tool_call_parser(
            tokenizer or model_name,
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
            revision=kwargs.get("revision"),
        )
        cmd.extend(["--tool-call-parser", parser])

        # OLMo3 requires custom chat template for tool calling (only when patch is applied)
        if parser == "olmo3" and patch_olmo3_tool_parser:
            cmd.extend(["--chat-template", _get_olmo3_tool_template_path()])

    # Prefix caching. vLLM defaults can vary by version, so pass an explicit
    # positive or negative flag instead of relying on omission semantics.
    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")

    # LoRA (Low-Rank Adaptation) support
    if enable_lora:
        cmd.append("--enable-lora")
        if lora_modules:
            # lora_modules should be a list of strings in format "name=/path/to/lora"
            cmd.extend(["--lora-modules"] + lora_modules)

    # Disable tqdm loading bar by default, enable with --debug-provider
    if is_debug_provider():
        cmd.append("--use-tqdm-on-load")
    else:
        cmd.append("--no-use-tqdm-on-load")

    # Add any extra kwargs as CLI args (filter out non-vLLM kwargs)
    for key, value in kwargs.items():
        if key in _NON_VLLM_KWARGS:
            continue
        if value is not None:
            arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cmd.append(arg_name)
            elif isinstance(value, (dict, list)):
                # JSON-encode complex values
                cmd.extend([arg_name, json.dumps(value)])
            else:
                cmd.extend([arg_name, str(value)])

    return cmd


class VLLMServerProcess:
    """Manages a vLLM server subprocess."""

    def __init__(
        self,
        model_name: str,
        port: int | None = None,
        tensor_parallel_size: int = 1,
        gpu_ids: list[int] | None = None,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        log_dir: str | None = None,
        owner: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the server manager.

        Args:
            model_name: HuggingFace model ID or local path
            port: Port to serve on (auto-assigned if None)
            tensor_parallel_size: Number of GPUs for tensor parallelism.
                If gpu_ids is provided, this is inferred from len(gpu_ids).
            gpu_ids: Specific GPU IDs to use. Sets CUDA_VISIBLE_DEVICES.
            startup_timeout: Maximum time to wait for server startup
            log_dir: Directory to write server logs to (if set, logs are persisted)
            owner: Owner identifier for log messages (e.g., worker ID)
            **kwargs: Additional arguments passed to vLLM server
        """
        from olmo_eval.common.logging import get_current_worker_id

        self.model_name = model_name
        self.port = port or _find_free_port()
        self.gpu_ids = gpu_ids
        # Infer tensor_parallel_size from gpu_ids if provided
        self.tensor_parallel_size = len(gpu_ids) if gpu_ids else tensor_parallel_size
        self.startup_timeout = startup_timeout
        self.log_dir = log_dir
        self.owner = owner or get_current_worker_id()
        self.server_kwargs = kwargs
        self._vllm_port = _find_free_internal_port(exclude={self.port})
        self._process: subprocess.Popen | None = None
        self._log_file: Any | None = None
        self._log_path: Any | None = None
        self._started = False

    @property
    def base_url(self) -> str:
        """Get the base URL for the OpenAI-compatible API."""
        return f"http://127.0.0.1:{self.port}/v1"

    def _log(self, level: int, msg: str) -> None:
        """Log a message with optional owner prefix."""
        if self.owner:
            logger.log(level, f"[{self.owner}] {msg}")
        else:
            logger.log(level, msg)

    def _read_log_output(self) -> str | None:
        """Read server logs, preferring the current server's log file."""
        if not self.log_dir:
            return None

        import pathlib

        candidate_paths: list[pathlib.Path] = []
        if self._log_path is not None:
            candidate_paths.append(self._log_path)

        log_dir = pathlib.Path(self.log_dir)
        if log_dir.exists():
            for path in sorted(
                log_dir.glob("vllm_server*.log"),
                key=lambda log_path: log_path.stat().st_mtime,
                reverse=True,
            ):
                if path not in candidate_paths:
                    candidate_paths.append(path)

        for log_path in candidate_paths:
            if log_path.exists():
                return log_path.read_text(errors="replace")

        return None

    def start(self, progress_callback: ProgressCallback | None = None) -> str:
        """Start the vLLM server.

        Args:
            progress_callback: Optional callback for progress updates during startup.

        Returns:
            The base URL for the OpenAI-compatible API

        Raises:
            RuntimeError: If server fails to start
        """
        if self._started:
            return self.base_url

        cmd = _build_server_command(
            model_name=self.model_name,
            port=self.port,
            tensor_parallel_size=self.tensor_parallel_size,
            **self.server_kwargs,
        )

        import shlex

        self._log(logging.INFO, f"Starting vLLM server: {shlex.join(cmd)}")
        if progress_callback:
            progress_callback(f"Starting vLLM server for {self.model_name}...")

        # Build environment for subprocess (child only, don't mutate parent)
        env = os.environ.copy()
        # VLLM_PYTHON can point at an isolated environment whose console tools
        # (for example ninja, used by FlashInfer JIT compilation) are installed
        # beside its Python executable. Make those tools visible to the child
        # without changing the launcher's environment.
        python_bin_dir = os.path.dirname(cmd[0])
        if python_bin_dir:
            python_bin_dir = os.path.abspath(python_bin_dir)
        inherited_path = env.get("PATH")
        env["PATH"] = os.pathsep.join(part for part in (python_bin_dir, inherited_path) if part)
        # VLLM_PYTHON is our own steering variable for selecting which Python
        # interpreter launches the server. Once the command is built, vLLM
        # itself does not need to see it, and forwarding it triggers a warning
        # because vLLM treats unknown VLLM_* variables as suspicious.
        env.pop("VLLM_PYTHON", None)
        # vLLM uses VLLM_PORT as the starting point for internal port
        # selection, including torch distributed rendezvous. Give each managed
        # server a low base port so concurrent cold starts do not collide in
        # the kernel ephemeral range.
        env["VLLM_PORT"] = str(self._vllm_port)
        if self.gpu_ids:
            # Invariant: gpu_ids are relative to the external CUDA_VISIBLE_DEVICES mask.
            # Only runner-side InferenceManager/aux servers reach this; workers pass
            # gpu_ids=None, avoiding a second composition that would mis-map GPUs.
            env["CUDA_VISIBLE_DEVICES"] = resolve_visible_devices(self.gpu_ids)

        # Allow extended max_model_len when user specifies it (e.g., with rope_scaling)
        # This is needed when max_model_len > model's max_position_embeddings
        if self.server_kwargs.get("max_model_len"):
            env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        # Enable verbose vLLM logging when debugging
        if is_debug_provider():
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            self._log(logging.INFO, "vLLM debug logging enabled (OLMO_EVAL_DEBUG_PROVIDER=1)")

        # Determine output handling:
        # 1. If log_dir is set, write to file for persistence
        # 2. If debugging, stream to stderr
        # 3. Otherwise capture for error reporting
        if self.log_dir:
            import pathlib

            self._log_path = pathlib.Path(self.log_dir) / f"vllm_server_{self.port}.log"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, "w")  # noqa: SIM115
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        elif is_debug_provider():
            self._process = subprocess.Popen(
                cmd,
                stdout=None,  # Inherit stdout
                stderr=None,  # Inherit stderr
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        else:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

        # Register cleanup on exit
        atexit.register(self.stop)

        # Wait for server to be ready
        success, last_error, process_output = _wait_for_server(
            self.base_url,
            timeout=self.startup_timeout,
            process=self._process,
            progress_callback=progress_callback,
        )
        if not success:
            # Stop the process first to avoid blocking on stdout.read()
            self.stop()

            # If log_dir is set, output went to file - read it from there
            if process_output is None and self.log_dir:
                with suppress(Exception):
                    process_output = self._read_log_output()

            # Log the captured output for debugging
            if process_output:
                self._log(logging.ERROR, f"vLLM server output:\n{process_output}")

            # Build a detailed error message
            error_msg = (
                f"vLLM server failed to start for model {self.model_name} "
                f"within {self.startup_timeout}s"
            )
            if last_error:
                error_msg += f". Error: {last_error}"
            if process_output:
                # Include a truncated version of the output in the exception
                max_output_len = 20000
                if len(process_output) > max_output_len:
                    truncated_output = "...[truncated]...\n" + process_output[-max_output_len:]
                else:
                    truncated_output = process_output
                error_msg += f"\n\nServer output:\n{truncated_output}"

            raise RuntimeError(error_msg)

        self._started = True
        return self.base_url

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        self._log(logging.INFO, "Stopping vLLM server...")

        try:
            # Try graceful shutdown first
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
                self._process.wait(timeout=5)

        except (ProcessLookupError, OSError):
            # Process already dead
            pass
        finally:
            self._process = None
            self._started = False
            atexit.unregister(self.stop)
            # Close log file if open
            if self._log_file is not None:
                with suppress(Exception):
                    self._log_file.close()
                self._log_file = None

        self._log(logging.INFO, "vLLM server stopped")

    def __enter__(self) -> str:
        """Context manager entry - start server and return URL."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - stop server."""
        self.stop()


@contextmanager
def vllm_server_context(
    model_name: str,
    port: int | None = None,
    tensor_parallel_size: int = 1,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    progress_callback: ProgressCallback | None = None,
    log_dir: str | None = None,
    **kwargs: Any,
) -> Generator[str, None, None]:
    """Context manager to start and stop a vLLM server.

    This is the recommended way to use a vLLM server for agent tasks.
    The server is automatically started and stopped.

    Args:
        model_name: HuggingFace model ID or local path
        port: Port to serve on (auto-assigned if None)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        startup_timeout: Maximum time to wait for server startup
        progress_callback: Optional callback for progress updates during startup
        log_dir: Directory to write server logs to (if set, logs are persisted)
        **kwargs: Additional arguments passed to vLLM server

    Yields:
        The base URL for the OpenAI-compatible API (e.g., "http://localhost:8000/v1")

    Example:
        with vllm_server_context("meta-llama/Llama-3.1-8B-Instruct") as url:
            # url is "http://localhost:<port>/v1"
            client = AsyncOpenAI(base_url=url, api_key="EMPTY")
            ...
    """
    server = VLLMServerProcess(
        model_name=model_name,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        startup_timeout=startup_timeout,
        log_dir=log_dir,
        **kwargs,
    )

    try:
        yield server.start(progress_callback=progress_callback)
    finally:
        server.stop()
