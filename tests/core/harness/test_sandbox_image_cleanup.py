"""Tests for removing derived sandbox images."""

import subprocess
import unittest
from unittest import mock

from olmo_eval.harness.sandbox import image


class TestRemoveSwerexImage(unittest.TestCase):
    def test_removes_derived_and_base_images_and_forgets_resolution(self) -> None:
        tag_hash = image._swerex_tag_hash("base:latest")
        image._resolved_images[tag_hash] = "registry/swerex:latest"
        done = subprocess.CompletedProcess(args=[], returncode=0)
        with mock.patch.object(image.subprocess, "run", return_value=done) as run:
            image.remove_swerex_image("base:latest", "podman")

        removed = sorted(call.args[0][-1] for call in run.call_args_list)
        self.assertEqual(
            removed, sorted(["base:latest", f"swerex-{tag_hash}:latest", "registry/swerex:latest"])
        )
        self.assertTrue(
            all(call.args[0][:3] == ["podman", "rmi", "--force"] for call in run.call_args_list)
        )
        self.assertNotIn(tag_hash, image._resolved_images)

    def test_failures_do_not_raise(self) -> None:
        failed = subprocess.CompletedProcess(args=[], returncode=1, stderr=b"image not known")
        with mock.patch.object(image.subprocess, "run", side_effect=[OSError("no podman"), failed]):
            image.remove_swerex_image("base:latest", "podman")
