"""Frozen, family-matched exploratory OOD companion to OMEGA-500."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

from olmo_eval.common.types import Instance
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.constants.omega_500_out import FAMILY_ALIASES, IDS_BY_CONFIG
from olmo_eval.evals.tasks.omega_500 import Omega500HillClimb

OMEGA_EXPLORATIVE_REVISION = "b04ae8d4757a2229e8ed65ac6a923e502d0bbb95"

CANONICAL_FAMILIES = {alias: family for family, alias in FAMILY_ALIASES.items()}

SELECTED_IDS = frozenset(item for items in IDS_BY_CONFIG.values() for item in items)


@register("omega_500_out")
class Omega500Out(Omega500HillClimb):
    """Evaluate 500 fixed test_out items with OMEGA-500's family proportions."""

    data_source = DataSource(
        path="allenai/omega-explorative",
        revision=OMEGA_EXPLORATIVE_REVISION,
        data_files=tuple(f"{config}/test_out-00000-of-00001.parquet" for config in IDS_BY_CONFIG),
    )

    @property
    def instances(self) -> Iterator[Instance]:
        instances = list(self._load_instances_cached())
        observed = [instance.metadata["id"] for instance in instances]
        if len(observed) != len(SELECTED_IDS) or set(observed) != SELECTED_IDS:
            counts = Counter(observed)
            missing = sorted(SELECTED_IDS - counts.keys())
            extra = sorted(counts.keys() - SELECTED_IDS)
            duplicates = sum(count - 1 for count in counts.values())
            raise ValueError(
                "OMEGA test_out data does not match the frozen 500-item manifest: "
                f"missing={missing[:5]!r}, extra={extra[:5]!r}, duplicates={duplicates}"
            )
        yield from instances

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        if doc["id"] not in SELECTED_IDS:
            return None
        instance = super().process_doc(doc, index)
        assert instance is not None
        family = instance.metadata["family"]
        instance.metadata["family"] = CANONICAL_FAMILIES.get(family, family)
        return instance
