"""Tests for the time budget on sandbox image resolution."""

import subprocess
import unittest
from unittest import mock

from olmo_eval.harness.sandbox import image


class TestSwerexImageTimeout(unittest.TestCase):
    def setUp(self) -> None:
        image._resolved_images.clear()
        config = mock.MagicMock(swerex_registry="")
        patcher = mock.patch.object(image, "get_infra_config", return_value=config)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_stuck_build_raises_timeout_error(self) -> None:
        def fake_run(cmd, **kwargs):
            if cmd[1] == "build":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return subprocess.CompletedProcess(cmd, returncode=1)

        with (
            mock.patch.object(image.subprocess, "run", side_effect=fake_run) as run,
            self.assertRaisesRegex(TimeoutError, "exceeded 5s during: podman build"),
        ):
            image.get_swerex_image("base:latest", "podman", timeout=5)

        build = next(c for c in run.call_args_list if c.args[0][1] == "build")
        self.assertLessEqual(build.kwargs["timeout"], 5)
        self.assertEqual(image._resolved_images, {})

    def test_stuck_base_image_pull_raises_timeout_error(self) -> None:
        def fake_run(cmd, **kwargs):
            if cmd[1] == "pull":
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return subprocess.CompletedProcess(cmd, returncode=1)

        with (
            mock.patch.object(image.subprocess, "run", side_effect=fake_run),
            self.assertRaisesRegex(TimeoutError, "during: podman pull"),
        ):
            image.get_swerex_image("base:latest", "podman", timeout=5)

    def test_spent_budget_stops_before_the_next_step(self) -> None:
        clock = iter([0.0, 0.0, 10.0])
        done = subprocess.CompletedProcess([], returncode=1)
        with (
            mock.patch.object(image.time, "monotonic", side_effect=lambda: next(clock)),
            mock.patch.object(image.subprocess, "run", return_value=done) as run,
            self.assertRaises(TimeoutError),
        ):
            image.get_swerex_image("base:latest", "podman", timeout=5)
        self.assertEqual(run.call_count, 1)

    def test_no_timeout_by_default(self) -> None:
        done = subprocess.CompletedProcess([], returncode=0)
        with mock.patch.object(image.subprocess, "run", return_value=done) as run:
            image.get_swerex_image("base:latest", "podman")
        self.assertIsNone(run.call_args.kwargs["timeout"])
