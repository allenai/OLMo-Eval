"""Tests for VLLMServerProcess startup and timeout handling."""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestVLLMServerStartupTimeout:
    """Tests for VLLMServerProcess startup timeout behavior."""

    @pytest.mark.anyio
    async def test_timeout_does_not_block_on_stdout(self):
        """Startup timeout terminates process before reading stdout to avoid blocking."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        # Create a mock process that never becomes healthy
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.stdout = MagicMock()
        mock_process.poll.return_value = None  # Process still running
        mock_process.wait.return_value = 0

        # Track if stop() is called before any stdout.read()
        call_order = []

        def track_killpg(*args):
            call_order.append("killpg")

        def track_read():
            call_order.append("read")
            # Simulate blocking read that would hang
            time.sleep(10)
            return b"output"

        mock_process.stdout.read.side_effect = track_read

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(False, "Connection refused", None),
            ),
            patch("atexit.register"),
            patch("atexit.unregister"),
            patch("os.killpg", side_effect=track_killpg),
            patch("os.getpgid", return_value=12345),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
                startup_timeout=1,  # Short timeout
            )

            with pytest.raises(RuntimeError, match="failed to start"):
                server.start()

            # Verify killpg was called (stop() uses killpg on Unix)
            # and stdout.read was never called (would have blocked)
            assert "killpg" in call_order
            assert "read" not in call_order

    @pytest.mark.anyio
    async def test_timeout_reads_log_file_after_stop(self, tmp_path):
        """After timeout, log file is read for error details."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        log_dir = tmp_path / "logs"
        log_dir.mkdir()

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0

        # Create a log file with error content
        log_file = log_dir / "vllm_server.log"
        log_content = "CUDA out of memory error"
        log_file.write_text(log_content)

        # Mock the log file handle
        mock_log_file = MagicMock()

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(False, "Connection refused", None),
            ),
            patch("builtins.open", return_value=mock_log_file),
            patch("atexit.register"),
            patch("atexit.unregister"),
            patch("os.killpg"),
            patch("os.getpgid", return_value=12345),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
                startup_timeout=1,
                log_dir=str(log_dir),
            )

            with pytest.raises(RuntimeError, match="failed to start") as exc_info:
                server.start()

            # Log content should be included in error message
            assert "CUDA out of memory" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_successful_startup_returns_url(self):
        """Successful startup returns the server URL."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ),
            patch("atexit.register"),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
            )

            url = server.start()
            assert url in ("http://localhost:8000/v1", "http://127.0.0.1:8000/v1")

    @pytest.mark.anyio
    async def test_vllm_python_selects_interpreter_but_not_child_env(self):
        """VLLM_PYTHON should steer our launcher without leaking into vLLM."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch.dict(
                "os.environ",
                {
                    "VLLM_PYTHON": "/opt/vllm-venv/bin/python",
                    "PATH": "/usr/local/bin:/usr/bin",
                },
                clear=True,
            ),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process) as mock_popen,
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ),
            patch("atexit.register"),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
            )

            server.start()

        cmd = mock_popen.call_args.args[0]
        env = mock_popen.call_args.kwargs["env"]
        assert cmd[0] == "/opt/vllm-venv/bin/python"
        assert "VLLM_PYTHON" not in env
        assert env["PATH"] == "/opt/vllm-venv/bin:/usr/local/bin:/usr/bin"
        assert env["VLLM_PORT"] == "23456"

    @pytest.mark.anyio
    async def test_sets_unique_internal_port_env(self):
        """Managed servers should pin vLLM's internal port selection base."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process) as mock_popen,
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ),
            patch("atexit.register"),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
            )

            server.start()

        cmd = mock_popen.call_args.args[0]
        env = mock_popen.call_args.kwargs["env"]
        assert "--distributed-init-method" not in cmd
        assert env["VLLM_PORT"] == "23456"

    @pytest.mark.anyio
    async def test_overrides_inherited_vllm_port_for_child(self):
        """Each managed server should get its own internal port base."""
        from olmo_eval.inference.providers.vllm_server_utils import VLLMServerProcess

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        with (
            patch.dict("os.environ", {"VLLM_PORT": "11111"}),
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._find_free_internal_port",
                return_value=23456,
            ),
            patch("subprocess.Popen", return_value=mock_process) as mock_popen,
            patch(
                "olmo_eval.inference.providers.vllm_server_utils._wait_for_server",
                return_value=(True, None, None),
            ),
            patch("atexit.register"),
        ):
            server = VLLMServerProcess(
                model_name="test-model",
                port=8000,
            )

            server.start()

        cmd = mock_popen.call_args.args[0]
        env = mock_popen.call_args.kwargs["env"]
        assert "--distributed-init-method" not in cmd
        assert env["VLLM_PORT"] == "23456"


class TestVLLMServerProviderStartup:
    """Tests for provider-side startup kwargs handling."""

    def test_provider_local_runtime_flags_do_not_leak_to_server_kwargs(self):
        """Provider-local runtime flags should not be forwarded as server CLI args."""
        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        mock_server = MagicMock()
        mock_server.base_url = "http://127.0.0.1:8000/v1"

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils.VLLMServerProcess",
                return_value=mock_server,
            ) as mock_server_cls,
            patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"),
        ):
            provider = VLLMServerProvider(
                "test-model",
                enforce_eager=True,
                add_bos_token=False,
                prompt_logprobs=1,
                completion_use_prompt_token_ids=True,
                completion_client_side_stop_trim=True,
                completion_sentencepiece_cleanup=True,
            )

        assert provider.base_url == "http://127.0.0.1:8000/v1"
        server_kwargs = mock_server_cls.call_args.kwargs
        assert server_kwargs["enforce_eager"] is True
        assert "add_bos_token" not in server_kwargs
        assert "prompt_logprobs" not in server_kwargs
        assert "completion_use_prompt_token_ids" not in server_kwargs
        assert "completion_client_side_stop_trim" not in server_kwargs
        assert "completion_sentencepiece_cleanup" not in server_kwargs

    def test_force_download_refreshes_cache_before_managed_server_startup(self):
        """Managed server startup should refresh HF cache without a vLLM CLI arg."""
        from unittest.mock import call

        from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

        mock_server = MagicMock()
        mock_server.base_url = "http://127.0.0.1:8000/v1"

        with (
            patch(
                "olmo_eval.inference.providers.vllm_server_utils.VLLMServerProcess",
                return_value=mock_server,
            ) as mock_server_cls,
            patch("olmo_eval.inference.providers.vllm_server.BeakerStatusReporter"),
            patch("olmo_eval.inference.providers.vllm_server.refresh_hf_cache") as refresh,
        ):
            provider = VLLMServerProvider(
                "org/model",
                tokenizer="org/tokenizer",
                revision="main",
                force_download=True,
                download_dir="/tmp/hf-cache",
            )

        assert provider.base_url == "http://127.0.0.1:8000/v1"
        assert refresh.call_args_list == [
            call(
                "org/model",
                revision="main",
                cache_dir="/tmp/hf-cache",
                token=None,
                force_download=True,
            ),
            call(
                "org/tokenizer",
                revision="main",
                cache_dir="/tmp/hf-cache",
                token=None,
                force_download=True,
            ),
        ]
        server_kwargs = mock_server_cls.call_args.kwargs
        assert server_kwargs["download_dir"] == "/tmp/hf-cache"
        assert "force_download" not in server_kwargs
