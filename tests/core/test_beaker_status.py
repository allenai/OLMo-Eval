"""Tests for BeakerStatusReporter."""

import threading
import unittest
from unittest import mock

from olmo_eval.common import beaker_status


class InlineThread:
    """Run a thread target synchronously to keep status reporter tests deterministic."""

    def __init__(self, *, target, name: str, daemon: bool) -> None:
        self.target = target
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self.target()


class BeakerStatusReporterTest(unittest.TestCase):
    def test_disabled_when_beaker_config_missing(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.object(
                beaker_status.Beaker,
                "from_env",
                side_effect=beaker_status.BeakerConfigurationError("no config"),
            ),
        ):
            reporter = beaker_status.BeakerStatusReporter()
        self.assertIsNone(reporter._client)
        reporter.update("hello")

    def test_throttles_updates_within_interval(self) -> None:
        env = {
            "BEAKER_WORKLOAD_ID": "wl_123",
            "GIT_COMMIT": "abc123",
            "GIT_BRANCH": "main",
        }
        fake_client = mock.MagicMock()
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch.object(
                beaker_status.Beaker, "from_env", return_value=fake_client
            ) as from_env,
            mock.patch.object(beaker_status, "Thread", InlineThread),
        ):
            reporter = beaker_status.BeakerStatusReporter(min_interval=60.0)

            self.assertIsNotNone(reporter._client)
            fake_client.workload.get.assert_not_called()

            with mock.patch("time.monotonic", side_effect=[0.0, 1.0, 61.0]):
                reporter.update("first")
                reporter.update("second")
                reporter.update("third")

            from_env.assert_called_once_with(check_for_upgrades=False)

        self.assertEqual(fake_client.workload.update.call_count, 2)
        suffix = "git_commit: abc123 git_branch: main"
        workload = beaker_status.BeakerWorkload(
            experiment=beaker_status.BeakerExperiment(id="wl_123")
        )
        fake_client.workload.update.assert_any_call(workload, description=f"first {suffix}")
        fake_client.workload.update.assert_any_call(workload, description=f"third {suffix}")

    def test_git_suffix_uses_unknown_when_env_missing(self) -> None:
        env = {"BEAKER_WORKLOAD_ID": "wl_123"}
        fake_client = mock.MagicMock()
        with (
            mock.patch.dict("os.environ", env, clear=True),
            mock.patch.object(beaker_status.Beaker, "from_env", return_value=fake_client),
            mock.patch.object(beaker_status, "Thread", InlineThread),
        ):
            reporter = beaker_status.BeakerStatusReporter(min_interval=0.0)
            reporter.update("hello")

        workload = beaker_status.BeakerWorkload(
            experiment=beaker_status.BeakerExperiment(id="wl_123")
        )
        fake_client.workload.update.assert_called_once_with(
            workload, description="hello git_commit: unknown git_branch: unknown"
        )

    def test_git_suffix_prefers_gantry_checkout_ref(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIT_REF": "current-checkout",
                "GIT_COMMIT": "stale-image-commit",
                "GIT_BRANCH": "main",
            },
            clear=True,
        ):
            self.assertEqual(
                beaker_status._git_suffix(),
                "git_commit: current-checkout git_branch: main",
            )

    def test_force_bypasses_throttle(self) -> None:
        fake_client = mock.MagicMock()
        with (
            mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True),
            mock.patch.object(beaker_status.Beaker, "from_env", return_value=fake_client),
            mock.patch.object(beaker_status, "Thread", InlineThread),
        ):
            reporter = beaker_status.BeakerStatusReporter(min_interval=60.0)

        with mock.patch("time.monotonic", side_effect=[0.0, 1.0]):
            reporter.update("a")
            reporter.update("b", force=True)

        self.assertEqual(fake_client.workload.update.call_count, 2)

    def test_update_failure_is_nonfatal_and_disables_reporting(self) -> None:
        fake_client = mock.MagicMock()
        fake_client.workload.update.side_effect = RuntimeError("API unavailable")
        with (
            mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True),
            mock.patch.object(beaker_status.Beaker, "from_env", return_value=fake_client),
            mock.patch.object(beaker_status, "Thread", InlineThread),
        ):
            reporter = beaker_status.BeakerStatusReporter()
            reporter.update("starting")

        self.assertIsNone(reporter._client)

    def test_update_starts_a_daemon_thread(self) -> None:
        fake_client = mock.MagicMock()
        fake_thread = mock.MagicMock()
        with (
            mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True),
            mock.patch.object(beaker_status.Beaker, "from_env", return_value=fake_client),
            mock.patch.object(beaker_status, "Thread", return_value=fake_thread) as thread_class,
        ):
            reporter = beaker_status.BeakerStatusReporter()
            reporter.update("starting")

        thread_class.assert_called_once()
        self.assertEqual(thread_class.call_args.kwargs["name"], "beaker-status-update")
        self.assertTrue(thread_class.call_args.kwargs["daemon"])
        fake_thread.start.assert_called_once_with()

    def _slow_reporter(self, release: threading.Event):
        sent: list[str] = []

        def update(_workload, description: str) -> None:
            sent.append(description.split(" git_commit")[0])
            if len(sent) == 1:
                release.wait(5)

        fake_client = mock.MagicMock()
        fake_client.workload.update.side_effect = update
        with (
            mock.patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_xyz"}, clear=True),
            mock.patch.object(beaker_status.Beaker, "from_env", return_value=fake_client),
        ):
            reporter = beaker_status.BeakerStatusReporter(min_interval=0.0)
        return reporter, sent

    def test_forced_update_during_slow_request_is_sent_after_it(self) -> None:
        release = threading.Event()
        reporter, sent = self._slow_reporter(release)

        reporter.update("first")
        reporter.update("skipped")
        reporter.update("final", force=True)
        threading.Timer(0.2, release.set).start()
        reporter.flush()

        self.assertEqual(sent, ["first", "final"])

    def test_only_the_newest_forced_update_is_kept(self) -> None:
        release = threading.Event()
        reporter, sent = self._slow_reporter(release)

        reporter.update("first")
        reporter.update("older", force=True)
        reporter.update("newest", force=True)
        release.set()
        reporter.flush()

        self.assertEqual(sent, ["first", "newest"])

    def test_flush_without_updates_returns(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            beaker_status.BeakerStatusReporter().flush()


if __name__ == "__main__":
    unittest.main()
