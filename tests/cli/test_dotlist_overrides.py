"""Tests for _parse_override_value and _apply_dotlist_overrides."""

from __future__ import annotations

import copy

import pytest

from olmo_eval.cli.run.config import _apply_dotlist_overrides, _parse_override_value

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sandbox_preset() -> dict:
    """Simulates a codex_universal-style preset serialized to a dict."""
    return {
        "name": "codex_universal",
        "sandboxes": [
            {
                "instances": 16,
                "image": "volcengine/sandbox-fusion:base-20250609",
                "mode": "docker",
                "startup_timeout": 300.0,
                "inject_swerex": True,
                "dockerfile_extra": [
                    "RUN mkdir -p /runtime/java",
                    "RUN curl -L -o /runtime/java/javatuples-1.2.jar https://example.com/jar",
                ],
            }
        ],
    }


# ── _parse_override_value ───────────────────────────────────────────────────


class TestParseOverrideValue:
    def test_json_dict(self):
        assert _parse_override_value('{"mode":"modal"}') == {"mode": "modal"}

    def test_json_list(self):
        assert _parse_override_value('["a","b"]') == ["a", "b"]

    def test_omegaconf_list(self):
        assert _parse_override_value("[a,b,c]") == ["a", "b", "c"]

    def test_empty_list(self):
        assert _parse_override_value("[]") == []

    def test_bool_true(self):
        assert _parse_override_value("true") is True

    def test_bool_false(self):
        assert _parse_override_value("False") is False

    def test_int(self):
        assert _parse_override_value("42") == 42

    def test_float(self):
        assert _parse_override_value("0.6") == 0.6

    def test_string(self):
        assert _parse_override_value("modal") == "modal"


# ── Scalar and field-level overrides ────────────────────────────────────────


class TestDotlistScalarOverrides:
    """Setting individual fields via dotlist paths (field-by-field)."""

    def test_set_top_level_scalar(self):
        base = {"name": "old"}
        result = _apply_dotlist_overrides(base, ["name=new"])
        assert result["name"] == "new"

    def test_set_nested_field(self, sandbox_preset):
        result = _apply_dotlist_overrides(sandbox_preset, ["sandboxes.0.mode=modal"])
        sb = result["sandboxes"][0]
        assert sb["mode"] == "modal"
        # Other fields survive — this is a field-level override.
        assert sb["image"] == "volcengine/sandbox-fusion:base-20250609"
        assert sb["inject_swerex"] is True
        assert sb["startup_timeout"] == 300.0

    def test_set_multiple_fields(self, sandbox_preset):
        result = _apply_dotlist_overrides(
            sandbox_preset,
            ["sandboxes.0.mode=modal", "sandboxes.0.instances=8"],
        )
        sb = result["sandboxes"][0]
        assert sb["mode"] == "modal"
        assert sb["instances"] == 8
        # Unchanged
        assert sb["inject_swerex"] is True

    def test_set_bool_field(self, sandbox_preset):
        result = _apply_dotlist_overrides(sandbox_preset, ["sandboxes.0.inject_swerex=false"])
        assert result["sandboxes"][0]["inject_swerex"] is False

    def test_creates_intermediate_dicts(self):
        base: dict = {}
        result = _apply_dotlist_overrides(base, ["a.b.c=1"])
        assert result == {"a": {"b": {"c": 1}}}

    def test_creates_nested_provider_kwargs(self):
        base = {"provider": {"kwargs": {}}}
        result = _apply_dotlist_overrides(
            base,
            ["provider.kwargs.chat_template_kwargs.enable_thinking=false"],
        )

        assert result["provider"]["kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}


# ── JSON object deep-merge ──────────────────────────────────────────────────


class TestDotlistJsonMerge:
    """A JSON dict at a path deep-merges into the existing target."""

    def test_json_merges_into_list_item(self, sandbox_preset):
        """JSON object at a list index merges — preset fields survive."""
        override = 'sandboxes.0={"mode":"modal","instances":64,"registry_auth":{"provider":"gcp"}}'
        result = _apply_dotlist_overrides(sandbox_preset, [override])
        sb = result["sandboxes"][0]
        # Override fields applied
        assert sb["mode"] == "modal"
        assert sb["instances"] == 64
        assert sb["registry_auth"] == {"provider": "gcp"}
        # Preset fields survive
        assert sb["image"] == "volcengine/sandbox-fusion:base-20250609"
        assert sb["inject_swerex"] is True
        assert sb["startup_timeout"] == 300.0

    def test_json_merges_into_dict_key(self):
        base = {"metrics": {"kind": "bpb", "extra": True}}
        result = _apply_dotlist_overrides(base, ['metrics={"kind":"accuracy"}'])
        assert result["metrics"]["kind"] == "accuracy"
        # Unmentioned keys survive from base
        assert result["metrics"]["extra"] is True

    def test_json_overrides_overlapping_keys(self, sandbox_preset):
        """Override values win for keys present in both base and override."""
        override = 'sandboxes.0={"mode":"modal","inject_swerex":false,"startup_timeout":60}'
        result = _apply_dotlist_overrides(sandbox_preset, [override])
        sb = result["sandboxes"][0]
        assert sb["inject_swerex"] is False
        assert sb["startup_timeout"] == 60
        # Untouched keys survive
        assert sb["image"] == "volcengine/sandbox-fusion:base-20250609"
        assert sb["instances"] == 16

    def test_json_empty_dict_is_noop(self, sandbox_preset):
        """An empty JSON dict merges nothing — base is unchanged."""
        original = copy.deepcopy(sandbox_preset)
        result = _apply_dotlist_overrides(sandbox_preset, ["sandboxes.0={}"])
        assert result["sandboxes"][0] == original["sandboxes"][0]

    def test_json_merge_into_sandboxes_broadcasts_shared_fields_only(self):
        """Global sandbox overrides should treat instances as a shared pool, not per-config."""
        base = {
            "sandboxes": [
                {"image": "default:latest", "mode": "docker"},
                {"image": "named:latest", "mode": "docker", "instances": 8},
            ]
        }

        result = _apply_dotlist_overrides(
            base,
            [
                'sandboxes={"mode":"modal","instances":64,"min_instances":24,'
                '"registry_auth":{"provider":"gcp"}}'
            ],
        )

        assert result["sandbox_pool_instances"] == 64
        assert result["sandbox_pool_min_instances"] == 24
        assert result["sandboxes"][0]["mode"] == "modal"
        assert result["sandboxes"][0]["registry_auth"] == {"provider": "gcp"}
        assert "instances" not in result["sandboxes"][0]
        assert "min_instances" not in result["sandboxes"][0]
        assert result["sandboxes"][1]["mode"] == "modal"
        assert result["sandboxes"][1]["registry_auth"] == {"provider": "gcp"}
        assert result["sandboxes"][1]["instances"] == 8
        assert "min_instances" not in result["sandboxes"][1]


# ── Non-dict values at list indices ─────────────────────────────────────────


class TestDotlistListIndexNonDict:
    def test_scalar_replaces_dict_in_list(self):
        base = {"items": [{"a": 1}]}
        result = _apply_dotlist_overrides(base, ["items.0=replaced"])
        assert result["items"][0] == "replaced"

    def test_list_replaces_dict_in_list(self):
        base = {"items": [{"a": 1}]}
        result = _apply_dotlist_overrides(base, ['items.0=["x","y"]'])
        assert result["items"][0] == ["x", "y"]


# ── Error handling ──────────────────────────────────────────────────────────


class TestDotlistErrors:
    def test_numeric_index_on_dict_raises(self):
        base = {"sandboxes": {"a": 1}}
        with pytest.raises(ValueError, match="numeric index but target is dict"):
            _apply_dotlist_overrides(base, ["sandboxes.0.mode=modal"])

    def test_index_out_of_bounds_raises(self):
        base = {"sandboxes": [{"mode": "docker"}]}
        with pytest.raises(ValueError, match="index 5 out of bounds"):
            _apply_dotlist_overrides(base, ["sandboxes.5.mode=modal"])

    def test_final_index_out_of_bounds_raises(self):
        base = {"items": [1, 2]}
        with pytest.raises(ValueError, match="index 9 out of bounds"):
            _apply_dotlist_overrides(base, ["items.9=val"])

    def test_key_on_non_dict_raises(self):
        base = {"sandboxes": [{"mode": "docker"}]}
        with pytest.raises(ValueError, match="cannot set key 'sub' on str"):
            _apply_dotlist_overrides(base, ["sandboxes.0.mode.sub=val"])

    def test_final_key_on_non_dict_raises(self):
        base = {"items": [1, 2, 3]}
        with pytest.raises(ValueError, match="cannot set key"):
            _apply_dotlist_overrides(base, ["items.0.key=val"])

    def test_no_equals_sign_skipped(self):
        base = {"a": 1}
        result = _apply_dotlist_overrides(base, ["no_equals_here"])
        assert result == {"a": 1}


# ── Mutability and ordering ────────────────────────────────────────────────


class TestDotlistMutationBehavior:
    def test_returns_new_dict(self):
        base = {"a": 1}
        result = _apply_dotlist_overrides(base, ["a=2"])
        assert result is not base
        assert result["a"] == 2
        assert base["a"] == 1

    def test_does_not_mutate_nested_input(self, sandbox_preset):
        original = copy.deepcopy(sandbox_preset)
        _apply_dotlist_overrides(
            sandbox_preset,
            ['sandboxes.0={"mode":"modal"}', "sandboxes.0.inject_swerex=false"],
        )
        assert sandbox_preset == original

    def test_does_not_mutate_input_for_shared_sandbox_override(self, sandbox_preset):
        original = copy.deepcopy(sandbox_preset)
        result = _apply_dotlist_overrides(
            sandbox_preset, ['sandboxes={"mode":"modal","instances":4}']
        )
        assert sandbox_preset == original
        assert result["sandbox_pool_instances"] == 4

    def test_later_override_wins(self):
        base = {"a": 1}
        result = _apply_dotlist_overrides(base, ["a=2", "a=3"])
        assert result["a"] == 3

    def test_json_merge_then_field_override(self, sandbox_preset):
        """A JSON merge followed by a field override composes correctly."""
        result = _apply_dotlist_overrides(
            sandbox_preset,
            [
                'sandboxes.0={"mode":"modal"}',
                "sandboxes.0.instances=8",
            ],
        )
        sb = result["sandboxes"][0]
        assert sb["mode"] == "modal"
        assert sb["instances"] == 8
        # Preset fields survive the merge
        assert sb["image"] == "volcengine/sandbox-fusion:base-20250609"
