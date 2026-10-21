"""The OMEGA companion has a fixed, family-matched development population."""

from collections import Counter
from unittest import mock

import pytest

from olmo_eval.common.types import Instance
from olmo_eval.evals.suites import AggregationStrategy, get_suite
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.constants import omega_500_out as manifest
from olmo_eval.evals.tasks.constants.omega_500_out import (
    FAMILY_ALIASES,
    FAMILY_COUNTS,
    IDS_BY_CONFIG,
)
from olmo_eval.evals.tasks.omega_500 import OMEGA_500_REVISION
from olmo_eval.evals.tasks.omega_500_out import OMEGA_EXPLORATIVE_REVISION, SELECTED_IDS


def test_manifest_matches_all_500_family_counts():
    assert len(SELECTED_IDS) == 500
    assert sum(map(len, IDS_BY_CONFIG.values())) == 500
    assert Counter({FAMILY_ALIASES.get(k, k): v for k, v in FAMILY_COUNTS.items()}) == Counter(
        {k: len(v) for k, v in IDS_BY_CONFIG.items()}
    )
    assert all(f"{config}_test_out_" in i for config, ids in IDS_BY_CONFIG.items() for i in ids)


def test_same_scoring_and_inference_budget():
    anchor = get_task("omega_500:hillclimb")
    companion = get_task("omega_500_out")
    assert anchor.config.sampling_params == companion.config.sampling_params
    assert anchor.config.metrics == companion.config.metrics
    assert anchor.config.formatter == companion.config.formatter
    assert companion.config.get_primary_metric().name == "exact_match"
    assert companion.config.data_source.revision == OMEGA_EXPLORATIVE_REVISION
    assert all("test_out-" in f for f in companion.config.data_source.data_files)


def test_dev_suite_reports_out_minus_in():
    suite = get_suite("omega:dev")
    assert suite.aggregation == AggregationStrategy.GAP
    assert suite.expand() == ("omega_500:hillclimb", "omega_500_out")


def test_manifest_records_the_pinned_revisions():
    assert manifest.__doc__ is not None
    assert OMEGA_500_REVISION in manifest.__doc__
    assert OMEGA_EXPLORATIVE_REVISION in manifest.__doc__


def test_selection_preserves_native_id():
    task = get_task("omega_500_out")
    item_id = next(iter(SELECTED_IDS))
    doc = {"id": item_id, "messages": [{"role": "user", "content": "q"}], "ground_truth": "42"}
    instance = task.process_doc(doc, 123)
    assert instance.metadata["id"] == item_id
    assert task.process_doc({**doc, "id": "not-in-manifest"}) is None


@pytest.mark.parametrize("mode", ["complete", "missing", "duplicate"])
def test_population_checked_before_yielding(mode):
    task = get_task("omega_500_out")
    instances = [Instance(question="q", metadata={"id": i}) for i in sorted(SELECTED_IDS)]
    if mode == "missing":
        instances.pop()
    elif mode == "duplicate":
        instances[-1] = instances[0]
    with mock.patch.object(task, "_load_instances_cached", return_value=iter(instances)):
        if mode == "complete":
            assert len(list(task.instances)) == 500
        else:
            with pytest.raises(ValueError, match="frozen 500-item manifest"):
                next(task.instances)


def test_companion_uses_anchor_family_names():
    task = get_task("omega_500_out")
    families = set()
    for config, ids in IDS_BY_CONFIG.items():
        instance = task.process_doc(
            {
                "id": ids[0],
                "setting_key": config,
                "messages": [{"content": "q"}],
                "ground_truth": "42",
            }
        )
        families.add(instance.metadata["family"])
    assert families == set(FAMILY_COUNTS)


def test_population_error_reports_drift():
    task = get_task("omega_500_out")
    ids = sorted(SELECTED_IDS)
    observed = ids[2:] + [ids[2], "unexpected"]
    with (
        mock.patch.object(
            task,
            "_load_instances_cached",
            return_value=iter(Instance(question="q", metadata={"id": i}) for i in observed),
        ),
        pytest.raises(ValueError) as error,
    ):
        next(task.instances)
    assert ids[0] in str(error.value)
    assert "unexpected" in str(error.value)
    assert "duplicates=1" in str(error.value)
