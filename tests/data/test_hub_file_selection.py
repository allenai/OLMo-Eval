"""Tests for data file selection when a repository has a legacy loading script.

`datasets` refuses to run legacy loading scripts, so the HuggingFace backend
falls back to reading the repository's data files directly. That fallback has
to load the files the caller asked for.
"""

from __future__ import annotations

from typing import Any

import pytest

from olmo_eval.data import DataSource
from olmo_eval.data.backends.huggingface import HuggingFaceBackend


class _RecordingLoader:
    """Stands in for `datasets.load_dataset`, recording how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"path": path, **kwargs})
        return [{"doc": "loaded"}]


@pytest.fixture
def recording_loader(monkeypatch: pytest.MonkeyPatch) -> _RecordingLoader:
    import datasets

    loader = _RecordingLoader()
    monkeypatch.setattr(datasets, "load_dataset", loader)
    return loader


def _all_repo_files(
    monkeypatch: pytest.MonkeyPatch, files: list[str], listed: dict[str, Any] | None = None
) -> None:
    """Make the Hub report `files` as the repository's contents.

    `listed`, when given, records the arguments the listing was asked for.
    """
    import huggingface_hub

    class _StubApi:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def list_repo_files(
            self, path: str, repo_type: str = "dataset", revision: str | None = None
        ) -> list[str]:
            if listed is not None:
                listed.update(path=path, repo_type=repo_type, revision=revision)
            return files

    monkeypatch.setattr(huggingface_hub, "HfApi", _StubApi)


def test_declared_data_files_are_the_only_ones_loaded(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The repository holds six data files; the caller asked for three of them.
    _all_repo_files(
        monkeypatch,
        ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
    )
    source = DataSource(
        path="org/scripted-repo",
        data_files=("test.jsonl", "test2.jsonl", "test3.jsonl"),
        split="train",
    )

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        data_files=["test.jsonl", "test2.jsonl", "test3.jsonl"],
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": [
            "hf://datasets/org/scripted-repo/test.jsonl",
            "hf://datasets/org/scripted-repo/test2.jsonl",
            "hf://datasets/org/scripted-repo/test3.jsonl",
        ]
    }


def test_a_single_declared_data_file_is_not_widened(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a subset, substring matching accepts every file in the repository.
    _all_repo_files(monkeypatch, ["test.jsonl", "test2.jsonl", "test3.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files="test.jsonl", split="train")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        data_files="test.jsonl",
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": ["hf://datasets/org/scripted-repo/test.jsonl"]
    }


def test_subset_matching_still_applies_without_declared_files(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["math/test.jsonl", "biology/test.jsonl", "README.md"])
    source = DataSource(path="org/scripted-repo", subset="math", split="test")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
    )

    assert recording_loader.calls[0]["data_files"] == {
        "test": ["hf://datasets/org/scripted-repo/math/test.jsonl"]
    }


def test_missing_subset_files_raise(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["math/test.jsonl"])
    source = DataSource(path="org/scripted-repo", subset="chemistry", split="test")

    with pytest.raises(FileNotFoundError):
        HuggingFaceBackend._load_from_hub_files(
            "org/scripted-repo",
            source,
            streaming=False,
            token=None,
        )


def test_revision_reaches_both_the_listing_and_the_files(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A listing taken from a different revision selects against other
    # contents, and files read from the default branch are the wrong data.
    listed: dict[str, Any] = {}
    _all_repo_files(monkeypatch, ["math/test.jsonl"], listed)
    source = DataSource(path="org/scripted-repo", subset="math", split="test", revision="abc123")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        revision="abc123",
    )

    assert listed["revision"] == "abc123"
    assert recording_loader.calls[0]["data_files"] == {
        "test": ["hf://datasets/org/scripted-repo@abc123/math/test.jsonl"]
    }


def test_declared_files_are_read_at_the_requested_revision(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["test.jsonl"])
    source = DataSource(
        path="org/scripted-repo", data_files=("test.jsonl",), split="train", revision="v1.0"
    )

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo",
        source,
        streaming=False,
        token=None,
        data_files=["test.jsonl"],
        revision="v1.0",
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": ["hf://datasets/org/scripted-repo@v1.0/test.jsonl"]
    }


def test_no_revision_leaves_urls_unpinned(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["test.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files="test.jsonl", split="train")

    HuggingFaceBackend._load_from_hub_files(
        "org/scripted-repo", source, streaming=False, token=None, data_files="test.jsonl"
    )

    assert recording_loader.calls[0]["data_files"] == {
        "train": ["hf://datasets/org/scripted-repo/test.jsonl"]
    }


def test_empty_declared_data_files_is_rejected(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_repo_files(monkeypatch, ["test.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files=(), split="train")

    with pytest.raises(ValueError, match="nothing to load"):
        HuggingFaceBackend._load_from_hub_files(
            "org/scripted-repo", source, streaming=False, token=None, data_files=[]
        )


def test_unreadable_declared_file_type_is_rejected(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A typo naming a documentation file would otherwise surface as an
    # obscure parsing error from the loader.
    _all_repo_files(monkeypatch, ["README.md"])
    source = DataSource(path="org/scripted-repo", data_files="README.md", split="train")

    with pytest.raises(ValueError, match="Cannot read README.md"):
        HuggingFaceBackend._load_from_hub_files(
            "org/scripted-repo", source, streaming=False, token=None, data_files="README.md"
        )


def test_mixed_declared_file_types_are_rejected(
    recording_loader: _RecordingLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One loader module reads the whole split, so the files must agree.
    _all_repo_files(monkeypatch, ["a.jsonl", "b.parquet"])
    source = DataSource(
        path="org/scripted-repo", data_files=("a.jsonl", "b.parquet"), split="train"
    )

    with pytest.raises(ValueError, match="mix file types"):
        HuggingFaceBackend._load_from_hub_files(
            "org/scripted-repo",
            source,
            streaming=False,
            token=None,
            data_files=["a.jsonl", "b.parquet"],
        )


class _ScriptRejectingLoader(_RecordingLoader):
    """Stands in for `datasets.load_dataset` against a repository with a script.

    The first call names the repository and fails the way `datasets` v4+ fails
    on a legacy loading script. Later calls name a loader module and succeed,
    so a test can follow a load all the way through the fallback.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def __call__(self, path: str, **kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"path": path, **kwargs})
        if len(self.calls) == 1:
            raise self.error
        return [{"doc": "loaded"}]


def _reject_scripts(monkeypatch: pytest.MonkeyPatch, error: Exception) -> _ScriptRejectingLoader:
    """Make loading the repository itself raise `error`."""
    import datasets

    loader = _ScriptRejectingLoader(error)
    monkeypatch.setattr(datasets, "load_dataset", loader)
    return loader


def test_load_carries_declared_files_and_revision_into_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The public entry point has to hand the fallback what the source asked
    # for; dropping either one reads the wrong files, or the right files at
    # the wrong revision.
    loader = _reject_scripts(monkeypatch, RuntimeError("Dataset scripts are no longer supported"))
    _all_repo_files(monkeypatch, ["test.jsonl", "test2.jsonl", "test3.jsonl"])
    source = DataSource(
        path="org/scripted-repo",
        data_files=("test.jsonl", "test2.jsonl"),
        split="train",
        revision="abc123",
    )

    assert list(HuggingFaceBackend().load(source)) == [{"doc": "loaded"}]

    attempted, fell_back = loader.calls
    assert attempted["path"] == "org/scripted-repo"
    # `datasets` rejects a tuple, so the source's files arrive as a list.
    assert attempted["data_files"] == ["test.jsonl", "test2.jsonl"]
    assert attempted["revision"] == "abc123"
    assert fell_back["data_files"] == {
        "train": [
            "hf://datasets/org/scripted-repo@abc123/test.jsonl",
            "hf://datasets/org/scripted-repo@abc123/test2.jsonl",
        ]
    }


def test_load_reaches_subset_matching_without_declared_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _reject_scripts(monkeypatch, RuntimeError("Dataset scripts are no longer supported"))
    _all_repo_files(monkeypatch, ["math/test.jsonl", "biology/test.jsonl", "README.md"])
    source = DataSource(path="org/scripted-repo", subset="math", split="test")

    assert list(HuggingFaceBackend().load(source)) == [{"doc": "loaded"}]

    attempted, fell_back = loader.calls
    assert "data_files" not in attempted
    assert "revision" not in attempted
    assert fell_back["data_files"] == {"test": ["hf://datasets/org/scripted-repo/math/test.jsonl"]}


def test_load_falls_back_when_the_library_reports_a_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The library's own fallback logic swallows the script error and raises
    # this instead, so the cache miss has to trigger the fallback too.
    loader = _reject_scripts(monkeypatch, ValueError("Couldn't find cache for org/scripted-repo"))
    _all_repo_files(monkeypatch, ["test.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files="test.jsonl", split="train")

    assert list(HuggingFaceBackend().load(source)) == [{"doc": "loaded"}]

    assert loader.calls[1]["data_files"] == {
        "train": ["hf://datasets/org/scripted-repo/test.jsonl"]
    }


def test_load_does_not_swallow_unrelated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _reject_scripts(monkeypatch, ValueError("Split 'train' not found"))
    _all_repo_files(monkeypatch, ["test.jsonl"])
    source = DataSource(path="org/scripted-repo", data_files="test.jsonl", split="train")

    with pytest.raises(ValueError, match="not found"):
        list(HuggingFaceBackend().load(source))
