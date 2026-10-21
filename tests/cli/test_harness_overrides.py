"""Tests that _apply_harness_overrides leaves the preset it was applied to intact."""

from __future__ import annotations

import pytest

from olmo_eval.cli.run.config import _apply_harness_overrides
from olmo_eval.harness import HarnessConfig, get_harness_preset
from olmo_eval.harness.config import ProviderConfig
from olmo_eval.harness.presets import HarnessPresets

PRESET_NAME = "override_isolation_probe"


@pytest.fixture
def preset(monkeypatch) -> HarnessConfig:
    """Register a preset with nested dicts in the fields to_dict can alias."""
    config = HarnessConfig(
        name=PRESET_NAME,
        provider=ProviderConfig(kwargs={"extra_body": {"a": 1}}),
        auxiliary_providers={"judge": ProviderConfig(kwargs={"extra_body": {"a": 1}})},
        scaffold="openai_agents",
        scaffold_kwargs={"model_settings": {"temperature": 0.0}},
    )
    monkeypatch.setattr(HarnessPresets, PRESET_NAME, config, raising=False)
    return config


@pytest.mark.parametrize(
    "override",
    [
        "provider.kwargs.extra_body.b=2",
        "auxiliary_providers.judge.kwargs.extra_body.b=2",
        "scaffold_kwargs.model_settings.top_p=0.9",
    ],
)
def test_override_does_not_mutate_cached_preset(preset, override):
    """A nested override changes the returned config and leaves the cached preset alone."""
    before = preset.to_dict()

    applied = _apply_harness_overrides(get_harness_preset(PRESET_NAME), [override])

    assert applied.to_dict() != before
    assert get_harness_preset(PRESET_NAME).to_dict() == before


def test_later_launch_does_not_inherit_earlier_override(preset):
    """Two launches from one cached preset each see only their own overrides."""
    _apply_harness_overrides(get_harness_preset(PRESET_NAME), ["provider.kwargs.extra_body.b=2"])
    second = _apply_harness_overrides(
        get_harness_preset(PRESET_NAME), ["provider.kwargs.extra_body.c=3"]
    )

    assert dict(second.provider.kwargs) == {"extra_body": {"a": 1, "c": 3}}
