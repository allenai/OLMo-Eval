"""Exercise manifest generation against local snapshots without network access."""

import ast
import importlib.util
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/internal/freeze_omega500_out.py"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def generator():
    spec = importlib.util.spec_from_file_location("freeze_omega500_out", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(generator, root, output):
    def snapshot(repo_id, **kwargs):
        if repo_id == "allenai/omega-500":
            assert kwargs == {
                "repo_type": "dataset",
                "revision": generator.OMEGA_500_REVISION,
                "allow_patterns": ["train-00000-of-00001.parquet"],
            }
            return str(root / "omega-500")
        assert repo_id == "allenai/omega-explorative"
        assert kwargs == {
            "repo_type": "dataset",
            "revision": generator.OMEGA_EXPLORATIVE_REVISION,
            "allow_patterns": ["*/test_out-00000-of-00001.parquet"],
        }
        return str(root / "omega-explorative")

    with mock.patch.object(generator, "snapshot_download", side_effect=snapshot) as download:
        generator.main(
            [
                str(output),
                "--omega-revision",
                generator.OMEGA_500_REVISION,
                "--explorative-revision",
                generator.OMEGA_EXPLORATIVE_REVISION,
            ]
        )
        assert download.call_count == 2


def test_manifest_selection_is_frozen_and_order_independent(tmp_path, generator):
    families = ["geometry_rotation", "geometry_rotation", "number_theory_prime_mod", "other"]
    anchor = tmp_path / "omega-500/train-00000-of-00001.parquet"
    candidates = {
        "geometry_polygon_rotation": [f"geom{i}" for i in range(5)],
        "numbertheory_lte_qr": [f"prime{i}" for i in range(5)],
        "other": [f"other{i}" for i in range(5)],
    }
    outputs = []
    for reverse in [False, True]:
        ordered = list(reversed(families)) if reverse else families
        write_rows(anchor, [{"family": family} for family in ordered])
        for config, ids in candidates.items():
            ordered_ids = list(reversed(ids)) if reverse else ids
            # Deliberately omit answers: selection must depend only on IDs and families.
            write_rows(
                tmp_path / "omega-explorative" / config / "test_out-00000-of-00001.parquet",
                [{"id": item_id} for item_id in ordered_ids],
            )
        output = tmp_path / f"manifest-{reverse}.py"
        generate(generator, tmp_path, output)
        outputs.append(output.read_text())
    assert outputs[0] == outputs[1]
    assert generator.OMEGA_500_REVISION in outputs[0]
    assert generator.OMEGA_EXPLORATIVE_REVISION in outputs[0]
    values = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(outputs[0]).body
        if isinstance(node, ast.Assign)
    }
    assert values["FAMILY_COUNTS"] == {
        "geometry_rotation": 2,
        "number_theory_prime_mod": 1,
        "other": 1,
    }
    assert values["IDS_BY_CONFIG"] == {
        "geometry_polygon_rotation": ("geom3", "geom4"),
        "numbertheory_lte_qr": ("prime4",),
        "other": ("other3",),
    }
    assert values["FAMILY_ALIASES"]["geometry_rotation"] == "geometry_polygon_rotation"
    assert values["FAMILY_ALIASES"]["number_theory_prime_mod"] == "numbertheory_lte_qr"


@pytest.mark.parametrize("ids", [["a"], ["a", "a", "b"]], ids=["insufficient", "duplicate"])
def test_invalid_candidate_pool_does_not_write_manifest(tmp_path, ids, generator):
    write_rows(
        tmp_path / "omega-500/train-00000-of-00001.parquet",
        [{"family": "other"}, {"family": "other"}],
    )
    write_rows(
        tmp_path / "omega-explorative/other/test_out-00000-of-00001.parquet",
        [{"id": item_id} for item_id in ids],
    )
    output = tmp_path / "manifest.py"
    with pytest.raises(ValueError, match="test_out IDs"):
        generate(generator, tmp_path, output)
    assert not output.exists()


def test_mutable_revision_is_rejected_before_download(tmp_path, generator):
    with mock.patch.object(generator, "snapshot_download") as download:
        with pytest.raises(ValueError, match="immutable"):
            generator.main([str(tmp_path / "out.py"), "--omega-revision", "main"])
        download.assert_not_called()


def test_requested_revisions_are_downloaded_and_recorded(tmp_path, generator):
    generator.OMEGA_500_REVISION = "a" * 40
    generator.OMEGA_EXPLORATIVE_REVISION = "b" * 40
    write_rows(tmp_path / "omega-500/train-00000-of-00001.parquet", [{"family": "other"}])
    write_rows(
        tmp_path / "omega-explorative/other/test_out-00000-of-00001.parquet", [{"id": "one"}]
    )
    output = tmp_path / "out.py"
    generate(generator, tmp_path, output)
    assert "a" * 40 in output.read_text()
    assert "b" * 40 in output.read_text()
