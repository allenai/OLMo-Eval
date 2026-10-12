"""Tests for the derived swe-rex image build."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox import image as image_mod


class TestDockerfile:
    def test_default_build_puts_the_venv_on_path(self) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", ("RUN echo extra",))

        assert dockerfile.startswith("FROM base:1\n")
        assert "apt-get install" in dockerfile
        assert 'ENV PATH="/root/venv/bin:$PATH"' in dockerfile
        assert "RUN echo extra" in dockerfile

    def test_pristine_build_leaves_the_base_environment_alone(self) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", ("RUN echo extra",), pristine=True)

        assert dockerfile.startswith("FROM base:1\n")
        assert "apt-get" not in dockerfile
        assert "ENV PATH" not in dockerfile
        assert "ENV VIRTUAL_ENV" not in dockerfile
        assert (
            "pip install --python /opt/swerex/venv/bin/python --no-cache-dir swe-rex" in dockerfile
        )
        assert (
            "ln -sf /opt/swerex/venv/bin/swerex-remote /usr/local/bin/swerex-remote" in dockerfile
        )
        assert "rm /usr/local/bin/swerex-uv" in dockerfile
        assert "/usr/local/bin/uv" not in dockerfile
        assert "RUN echo extra" in dockerfile
        assert dockerfile.count("USER ") == 1

    @pytest.mark.parametrize("user", [None, "root", "0"])
    def test_pristine_build_stays_root_for_root_images(self, user) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", pristine=True, user=user)
        assert dockerfile.rstrip().endswith("rm /usr/local/bin/swerex-uv")

    def test_pristine_build_hands_back_to_the_base_user(self) -> None:
        dockerfile = image_mod.build_swerex_dockerfile("base:1", pristine=True, user="model")
        assert dockerfile.rstrip().endswith("USER model")
        assert "chmod -R a+rX /opt/swerex" in dockerfile


class TestImageResolution:
    def setup_method(self) -> None:
        image_mod._resolved_images.clear()

    def teardown_method(self) -> None:
        image_mod._resolved_images.clear()

    def test_pristine_and_default_images_have_different_tags(self) -> None:
        with mock.patch.object(image_mod, "_resolve_swerex_image", return_value="x") as resolve:
            image_mod.get_swerex_image("base:1", "docker")
            image_mod.get_swerex_image("base:1", "docker", pristine=True)

        tags = [call.args[4] for call in resolve.call_args_list]
        assert len(set(tags)) == 2
        assert resolve.call_args_list[1].args[5] is True

    def test_resolution_is_cached_per_tag(self) -> None:
        with mock.patch.object(image_mod, "_resolve_swerex_image", return_value="x") as resolve:
            image_mod.get_swerex_image("base:1", "docker", pristine=True)
            image_mod.get_swerex_image("base:1", "docker", pristine=True)

        assert resolve.call_count == 1

    def test_a_build_uses_the_pristine_dockerfile_on_the_base_platform(self) -> None:
        completed = mock.Mock(returncode=0, stderr=b"")
        missing = mock.Mock(returncode=1, stderr=b"")
        inspected = mock.Mock(returncode=0, stdout=b"linux/amd64\n")
        user = mock.Mock(returncode=0, stdout=b"model\n")
        with (
            mock.patch.object(image_mod, "get_infra_config") as infra,
            mock.patch.object(
                image_mod.subprocess, "run", side_effect=[missing, inspected, user, completed]
            ) as run,
        ):
            infra.return_value.swerex_registry = None
            image = image_mod._resolve_swerex_image("base:1", "docker", (), False, "abc", True)

        assert image == "swerex-abc:latest"
        build = run.call_args_list[3]
        assert build.args[0] == [
            "docker",
            "build",
            "-t",
            "swerex-abc:latest",
            "--platform=linux/amd64",
            "-",
        ]
        assert b"ENV PATH" not in build.kwargs["input"]
        assert b"swerex-remote" in build.kwargs["input"]
        assert build.kwargs["input"].rstrip().endswith(b"USER model")

    def test_a_missing_base_image_is_pulled_to_read_its_platform(self) -> None:
        not_local = mock.Mock(returncode=1)
        pulled = mock.Mock(returncode=0)
        inspected = mock.Mock(returncode=0, stdout=b"linux/arm64")
        with mock.patch.object(
            image_mod.subprocess, "run", side_effect=[not_local, pulled, inspected]
        ) as run:
            assert image_mod._image_platform("podman", "base:1") == "linux/arm64"

        assert run.call_args_list[1].args[0] == ["podman", "pull", "base:1"]

    def test_an_unreadable_platform_leaves_the_build_default(self) -> None:
        failed = mock.Mock(returncode=1, stderr=b"nope")
        with mock.patch.object(image_mod.subprocess, "run", return_value=failed):
            assert image_mod._image_platform("docker", "base:1") is None


def test_pristine_image_round_trips_through_dict() -> None:
    config = SandboxConfig(image="base:1", mode=SandboxMode.DOCKER, pristine_image=True)

    restored = SandboxConfig.from_dict(config.to_dict())

    assert restored.pristine_image is True
    assert SandboxConfig.from_dict({"image": "b", "mode": "docker"}).pristine_image is False


class TestTaskImageBuild:
    def test_the_context_hash_tracks_file_contents(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM a\n")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "x.bin").write_bytes(b"1")
        first = image_mod.build_context_hash(tmp_path)
        assert first == image_mod.build_context_hash(tmp_path)
        (tmp_path / "data" / "x.bin").write_bytes(b"2")
        assert image_mod.build_context_hash(tmp_path) != first

    def test_a_cached_image_is_reused(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM a\n")
        with mock.patch.object(
            image_mod.subprocess, "run", return_value=mock.Mock(returncode=0)
        ) as run:
            image = image_mod.build_task_image(tmp_path, "docker", "tb-task-x", 60.0)

        assert image.startswith("tb-task-x:")
        assert run.call_count == 1
        assert run.call_args.args[0][:3] == ["docker", "image", "inspect"]

    def test_a_missing_image_is_built_with_the_context(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM a\n")
        missing = mock.Mock(returncode=1)
        built = mock.Mock(returncode=0, stderr=b"")
        with mock.patch.object(image_mod.subprocess, "run", side_effect=[missing, built]) as run:
            image = image_mod.build_task_image(tmp_path, "podman", "tb-task-x", 60.0)

        build = run.call_args_list[1]
        assert build.args[0] == ["podman", "build", "-t", image, str(tmp_path)]
        assert build.kwargs["timeout"] == 60.0

    def test_a_failed_build_raises_with_its_output(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM a\n")
        missing = mock.Mock(returncode=1)
        failed = mock.Mock(returncode=1, stderr=b"apt broke")
        with (
            mock.patch.object(image_mod.subprocess, "run", side_effect=[missing, failed]),
            pytest.raises(RuntimeError, match="apt broke"),
        ):
            image_mod.build_task_image(tmp_path, "docker", "tb-task-x")

    def test_a_timed_out_build_raises(self, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text("FROM a\n")
        missing = mock.Mock(returncode=1)
        with (
            mock.patch.object(
                image_mod.subprocess,
                "run",
                side_effect=[missing, image_mod.subprocess.TimeoutExpired("build", 5)],
            ),
            pytest.raises(RuntimeError, match="exceeded 5"),
        ):
            image_mod.build_task_image(tmp_path, "docker", "tb-task-x", 5.0)

    def test_a_context_without_a_dockerfile_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No Dockerfile"):
            image_mod.build_task_image(tmp_path, "docker", "tb-task-x")
