"""Tests for Beaker secret handling: BEAKER_TOKEN and HF_TOKEN injection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from beaker.exceptions import BeakerSecretNotFound
from huggingface_hub.errors import RepositoryNotFoundError

from olmo_eval.launch.beaker.launcher import BeakerJobConfig
from olmo_eval.launch.beaker.secrets import (
    HfAccessCheck,
    check_hf_access,
    plan_hf_token,
)

SECRETS = "olmo_eval.launch.beaker.secrets"


def _beaker_client(existing: set[str]) -> MagicMock:
    """A mock Beaker client whose secret store holds ``existing``."""
    client = MagicMock(user_name="alice")

    def get(name, workspace=None):
        if name not in existing:
            raise BeakerSecretNotFound(name)
        return MagicMock(name=name)

    client.secret.get.side_effect = get
    return client


# ---------------------------------------------------------------------------
# BEAKER_TOKEN
# ---------------------------------------------------------------------------


class TestBeakerTokenInjection:
    def _launch(self, client: MagicMock, dry_run: bool = False):
        from olmo_eval.launch import BeakerLauncher

        launcher = BeakerLauncher(workspace="ai2/oe-data")
        launcher._beaker = client
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            weka_buckets=[],
        )
        with patch("gantry.api.launch_experiment") as mock_launch:
            launcher.launch(config, dry_run=dry_run)
        return mock_launch.call_args.kwargs.get("env_secrets") or []

    def test_missing_secret_skips_injection_with_warning(self, capsys):
        client = _beaker_client(existing=set())
        env_secrets = self._launch(client)
        assert not any(name == "BEAKER_TOKEN" for name, _ in env_secrets)
        client.secret.write.assert_not_called()
        out = " ".join(capsys.readouterr().out.split())
        assert "alice_BEAKER_TOKEN not found" in out
        assert "status updates in the Beaker UI are disabled" in out
        assert "beaker secret write" in out

    def test_existing_secret_is_injected(self):
        client = _beaker_client(existing={"alice_BEAKER_TOKEN"})
        env_secrets = self._launch(client)
        assert ("BEAKER_TOKEN", "alice_BEAKER_TOKEN") in env_secrets
        client.secret.write.assert_not_called()

    def test_dry_run_does_not_query_beaker_secrets(self, capsys):
        client = _beaker_client(existing={"alice_BEAKER_TOKEN"})
        env_secrets = self._launch(client, dry_run=True)
        client.secret.get.assert_not_called()
        assert not any(name == "BEAKER_TOKEN" for name, _ in env_secrets)
        assert "Not checked in a dry run" in " ".join(capsys.readouterr().out.split())

    def test_explicit_beaker_token_is_kept(self):
        from olmo_eval.launch import BeakerLauncher
        from olmo_eval.launch.beaker.launcher import BeakerEnvSecret

        launcher = BeakerLauncher(workspace="ai2/oe-data")
        launcher._beaker = _beaker_client(existing=set())
        config = BeakerJobConfig(
            name="test",
            command=["echo"],
            cluster="h100",
            workspace="ai2/oe-data",
            budget="ai2/oe-other",
            weka_buckets=[],
            env_secrets=[BeakerEnvSecret("BEAKER_TOKEN", "my_token")],
        )
        with patch("gantry.api.launch_experiment") as mock_launch:
            launcher.launch(config)
        env_secrets = mock_launch.call_args.kwargs["env_secrets"]
        assert [s for s in env_secrets if s[0] == "BEAKER_TOKEN"] == [("BEAKER_TOKEN", "my_token")]
        launcher._beaker.secret.get.assert_not_called()


def test_status_reporter_tolerates_missing_beaker_token():
    from olmo_eval.common import beaker_status

    with (
        patch.dict("os.environ", {"BEAKER_WORKLOAD_ID": "wl_123"}, clear=True),
        patch.object(
            beaker_status.Beaker,
            "from_env",
            side_effect=beaker_status.BeakerConfigurationError("no token"),
        ),
    ):
        reporter = beaker_status.BeakerStatusReporter()
        reporter.update("hello", force=True)
        reporter.progress_callback("items")(1, 2)
        reporter.flush()
    assert reporter._client is None


# ---------------------------------------------------------------------------
# HF_TOKEN: gating check
# ---------------------------------------------------------------------------


def _hf_api(repos: dict[str, object]) -> MagicMock:
    """A mock HfApi. Values: an info namespace, or an exception to raise.

    A repo listed under "private:<id>" is found only with a token.
    """
    api = MagicMock()

    def repo_info(repo_id, *, repo_type=None, timeout=None, token=None):
        if f"private:{repo_id}" in repos and token:
            return SimpleNamespace(gated=False, private=True)
        value = repos.get(repo_id)
        if value is None:
            raise RepositoryNotFoundError("not found", response=MagicMock())
        if isinstance(value, Exception):
            raise value
        return value

    api.repo_info.side_effect = repo_info
    return api


PUBLIC = SimpleNamespace(gated=False, private=False)


class TestCheckHfAccess:
    def test_public_models_need_no_token(self):
        api = _hf_api(
            {
                "Qwen/Qwen2-0.5B": PUBLIC,
                "Qwen/Qwen2.5-0.5B": PUBLIC,
                "allenai/OLMo-2-0425-1B": PUBLIC,
            }
        )
        with patch("huggingface_hub.HfApi", return_value=api):
            check = check_hf_access(
                [
                    ("Qwen/Qwen2-0.5B", "model"),
                    ("Qwen/Qwen2.5-0.5B", "model"),
                    ("allenai/OLMo-2-0425-1B", "model"),
                ]
            )
        assert check == HfAccessCheck()
        for call in api.repo_info.call_args_list:
            assert call.kwargs["token"] is False

    def test_gated_model_and_dataset_need_token(self):
        api = _hf_api(
            {
                "meta-llama/Llama-3.1-8B": SimpleNamespace(gated="manual", private=False),
                "Idavidrein/gpqa": SimpleNamespace(gated="auto", private=False),
            }
        )
        with patch("huggingface_hub.HfApi", return_value=api):
            check = check_hf_access(
                [("meta-llama/Llama-3.1-8B", "model"), ("hf://Idavidrein/gpqa", "dataset")]
            )
        assert check.needs_token == [
            "dataset Idavidrein/gpqa (gated)",
            "model meta-llama/Llama-3.1-8B (gated)",
        ]

    def test_private_repo_detected_with_local_token(self):
        api = _hf_api({"private:allenai/secret-model": True})
        with patch("huggingface_hub.HfApi", return_value=api):
            check = check_hf_access([("allenai/secret-model", "model")], token="hf_x")
        assert check.needs_token == ["model allenai/secret-model (private)"]

    def test_missing_repo_without_token_is_ignored(self):
        api = _hf_api({})
        with patch("huggingface_hub.HfApi", return_value=api):
            check = check_hf_access([("org/does-not-exist", "model")])
        assert check == HfAccessCheck()

    def test_network_error_is_unchecked(self):
        api = _hf_api({"Qwen/Qwen2-0.5B": TimeoutError("slow")})
        with patch("huggingface_hub.HfApi", return_value=api):
            check = check_hf_access([("Qwen/Qwen2-0.5B", "model")])
        assert check.unchecked == ["model Qwen/Qwen2-0.5B"]
        assert check.needs_token == []

    def test_paths_and_api_models_are_not_looked_up(self):
        api = _hf_api({})
        with patch("huggingface_hub.HfApi", return_value=api):
            check_hf_access(
                [
                    ("s3://bucket/model", "model"),
                    ("gs://bucket/model", "model"),
                    ("/weka/oe-eval/model", "model"),
                    ("gpt-4o", "model"),
                ]
            )
        api.repo_info.assert_not_called()


# ---------------------------------------------------------------------------
# HF_TOKEN: plan
# ---------------------------------------------------------------------------


def _plan(**kwargs):
    defaults = {
        "username": "alice",
        "mode": None,
        "secret_present": False,
        "required": False,
        "access": HfAccessCheck(),
        "have_local_token": True,
    }
    return plan_hf_token(**{**defaults, **kwargs})


class TestPlanHfToken:
    def test_public_does_not_copy_or_inject(self):
        plan = _plan()
        assert not plan.inject
        assert not plan.copy_local
        assert plan.error is None

    def test_existing_secret_is_injected_without_copy(self):
        plan = _plan(secret_present=True)
        assert plan.inject
        assert not plan.copy_local
        assert plan.secret_name == "alice_HF_TOKEN"

    def test_flag_forces_copy(self):
        plan = _plan(mode=True)
        assert plan.inject
        assert plan.copy_local

    def test_task_requirement_copies(self):
        plan = _plan(required=True)
        assert plan.inject
        assert plan.copy_local

    def test_gated_copies(self):
        plan = _plan(access=HfAccessCheck(needs_token=["model meta-llama/x (gated)"]))
        assert plan.copy_local
        assert "meta-llama/x" in plan.message

    def test_needed_without_local_token_is_an_error(self):
        plan = _plan(required=True, have_local_token=False)
        assert not plan.inject
        assert plan.error is not None
        assert "beaker secret write alice_HF_TOKEN" in plan.error

    def test_no_flag_never_injects(self):
        plan = _plan(mode=False, secret_present=True, required=True)
        assert not plan.inject
        assert not plan.copy_local
        assert plan.warning

    def test_unchecked_repo_warns_without_copying(self):
        plan = _plan(access=HfAccessCheck(unchecked=["model Qwen/Qwen2-0.5B"]))
        assert not plan.copy_local
        assert plan.warning
        assert "--hf-token" in plan.message

    def test_dry_run_needed_injects_without_copying(self):
        plan = _plan(secret_present=None, mode=True)
        assert plan.inject
        assert not plan.copy_local


# ---------------------------------------------------------------------------
# HF_TOKEN: CLI wiring
# ---------------------------------------------------------------------------


class TestPrepareSecrets:
    def _run(
        self,
        *,
        existing: set[str],
        repos: dict[str, object],
        hf_token: bool | None = None,
        required: set[str] | None = None,
        dry_run: bool = False,
        model: str = "Qwen/Qwen2-0.5B",
    ):
        from olmo_eval.cli.beaker.launch import _prepare_secrets

        launcher = MagicMock(_workspace="ai2/oe-data")
        launcher.beaker = _beaker_client(existing)
        api = _hf_api(repos)
        with (
            patch("huggingface_hub.HfApi", return_value=api),
            patch(f"{SECRETS}.get_local_hf_token", return_value="hf_local"),
            patch(f"{SECRETS}.get_local_wandb_api_key", return_value=None),
            patch(f"{SECRETS}.ensure_common_secrets", return_value=[]),
            patch(f"{SECRETS}.ensure_task_secrets", return_value=[]),
        ):
            common, _task = _prepare_secrets(
                launcher,
                dry_run=dry_run,
                all_required_secrets=required or set(),
                hf_token=hf_token,
                hf_repos=[(model, "model")],
            )
        return common, launcher.beaker, api

    @pytest.mark.parametrize(
        "model", ["Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B", "allenai/OLMo-2-0425-1B"]
    )
    def test_public_model_does_not_copy_hf_token(self, model):
        common, client, _ = self._run(existing=set(), repos={model: PUBLIC}, model=model)
        client.secret.write.assert_not_called()
        assert not any(env == "HF_TOKEN" for env, _ in common)

    def test_existing_hf_secret_is_injected_without_copy(self):
        common, client, api = self._run(
            existing={"alice_HF_TOKEN"}, repos={"Qwen/Qwen2-0.5B": PUBLIC}
        )
        assert ("HF_TOKEN", "alice_HF_TOKEN") in common
        client.secret.write.assert_not_called()
        api.repo_info.assert_not_called()

    def test_hf_token_flag_forces_copy(self):
        common, client, _ = self._run(
            existing=set(), repos={"Qwen/Qwen2-0.5B": PUBLIC}, hf_token=True
        )
        assert ("HF_TOKEN", "alice_HF_TOKEN") in common
        client.secret.write.assert_called_once()
        assert client.secret.write.call_args.args[:2] == ("alice_HF_TOKEN", "hf_local")

    def test_gated_model_copies(self):
        common, client, _ = self._run(
            existing=set(),
            repos={"meta-llama/Llama-3.1-8B": SimpleNamespace(gated="manual", private=False)},
            model="meta-llama/Llama-3.1-8B",
        )
        assert ("HF_TOKEN", "alice_HF_TOKEN") in common
        client.secret.write.assert_called_once()

    def test_task_declared_hf_token_copies(self):
        common, client, _ = self._run(
            existing=set(), repos={"Qwen/Qwen2-0.5B": PUBLIC}, required={"HF_TOKEN"}
        )
        assert ("HF_TOKEN", "alice_HF_TOKEN") in common
        client.secret.write.assert_called_once()

    def test_no_hf_token_flag_skips_existing_secret(self):
        common, client, _ = self._run(
            existing={"alice_HF_TOKEN"}, repos={"Qwen/Qwen2-0.5B": PUBLIC}, hf_token=False
        )
        assert not any(env == "HF_TOKEN" for env, _ in common)
        client.secret.get.assert_not_called()

    def test_dry_run_does_not_touch_beaker_secrets(self):
        common, client, _ = self._run(
            existing=set(), repos={"Qwen/Qwen2-0.5B": PUBLIC}, hf_token=True, dry_run=True
        )
        client.secret.get.assert_not_called()
        client.secret.write.assert_not_called()
        assert ("HF_TOKEN", "alice_HF_TOKEN") in common


def test_hf_repos_collects_models_tokenizers_and_hf_datasets():
    from olmo_eval.cli.beaker.launch import _hf_repos
    from olmo_eval.data import DataSource

    task_cfg = SimpleNamespace(
        data_source=DataSource(path="Idavidrein/gpqa"),
        fewshot_source="s3://bucket/fewshot.jsonl",
    )
    repos = _hf_repos(["Qwen/Qwen2-0.5B"], [task_cfg])
    assert ("Qwen/Qwen2-0.5B", "model") in repos
    assert ("Idavidrein/gpqa", "dataset") in repos
    assert not any(r[0].startswith("s3://") for r in repos)


@pytest.mark.parametrize("exists", [True, False])
def test_wandb_key_is_injected_only_when_the_secret_exists(exists: bool) -> None:
    from olmo_eval.launch.beaker.secrets import ensure_common_secrets

    client = MagicMock()
    client.user_name = "alice"
    if not exists:
        client.secret.get.side_effect = BeakerSecretNotFound("missing")
    with (
        patch("beaker.Beaker.from_env", return_value=client),
        patch(f"{SECRETS}._get_beaker_username", return_value="alice"),
        patch(f"{SECRETS}.get_local_wandb_api_key", return_value="local-key") as local_key,
    ):
        secrets = ensure_common_secrets(workspace="ai2/ws")
    assert secrets == ([("WANDB_API_KEY", "alice_WANDB_API_KEY")] if exists else [])
    client.secret.write.assert_not_called()
    local_key.assert_not_called()
