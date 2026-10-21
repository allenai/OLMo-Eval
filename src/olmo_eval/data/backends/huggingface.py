"""HuggingFace Hub dataset backend."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource

#: File types the fallback loader can read, mapped to the loader module.
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    "jsonl": "json",
    "json": "json",
    "parquet": "parquet",
    "csv": "csv",
}


def _loader_module(file_names: list[str], path: str) -> str:
    """Return the loader module the given data files need.

    One module reads the whole split, so the files have to agree on a type.
    Naming an unreadable or mixed set fails here rather than as an obscure
    parsing error from the loader.
    """
    unreadable = sorted(
        name for name in file_names if name.rsplit(".", 1)[-1] not in _SUPPORTED_EXTENSIONS
    )
    if unreadable:
        raise ValueError(
            f"Cannot read {', '.join(unreadable)} from {path}. "
            f"Supported file types: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )

    modules = {_SUPPORTED_EXTENSIONS[name.rsplit(".", 1)[-1]] for name in file_names}
    if len(modules) > 1:
        raise ValueError(
            f"Data files for {path} mix file types ({', '.join(sorted(file_names))}), "
            f"which cannot be read as one split."
        )
    return modules.pop()


class HuggingFaceBackend:
    """Load datasets from HuggingFace Hub.

    Supports all HuggingFace datasets accessible via the `datasets` library.
    The path can be in org/repo format or prefixed with hf://.

    Examples:
        >>> backend = HuggingFaceBackend()
        >>> source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from HuggingFace Hub.

        Args:
            source: The data source with HuggingFace dataset path.
            streaming: Whether to stream the dataset.

        Yields:
            Raw document dictionaries from the dataset.
        """
        import os

        from datasets import load_dataset

        # Remove hf:// prefix if present
        path = source.path.removeprefix("hf://")

        # Use HF_TOKEN for authentication if available
        token = os.getenv("HF_TOKEN")

        kwargs: dict[str, Any] = {}
        if source.data_files is not None:
            kwargs["data_files"] = (
                list(source.data_files)
                if isinstance(source.data_files, tuple)
                else source.data_files
            )
        if source.revision is not None:
            kwargs["revision"] = source.revision

        try:
            dataset = load_dataset(
                path,
                name=source.subset,
                split=source.split,
                streaming=streaming,
                token=token,
                **kwargs,
            )
        except (RuntimeError, ValueError) as exc:
            err = str(exc)
            is_script_error = "Dataset scripts are no longer supported" in err
            is_cache_error = "Couldn't find cache" in err
            if not is_script_error and not is_cache_error:
                raise
            # datasets v4+ rejects repos that contain a legacy loading script.
            # The RuntimeError is silently caught by the library's fallback
            # logic and replaced with a confusing cache-miss ValueError.
            # Work around both by loading the data files directly from the Hub.
            dataset = self._load_from_hub_files(path, source, streaming, token, **kwargs)

        yield from dataset

    @staticmethod
    def _load_from_hub_files(
        path: str,
        source: DataSource,
        streaming: bool,
        token: str | None,
        **kwargs: Any,
    ) -> Any:
        """Load a dataset directly from Hub data files, bypassing the module factory."""
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = kwargs.get("revision")

        declared = kwargs.get("data_files")
        if declared is not None:
            # An explicit file list names the split exactly; subset matching
            # would silently widen it to every data file in the repository.
            candidates = [declared] if isinstance(declared, str) else list(declared)
            if not candidates:
                raise ValueError(
                    f"data_files for {path} is empty, so there is nothing to load. "
                    f"Name the file(s) to read, or leave data_files unset to select "
                    f"them by subset."
                )
        else:
            api = HfApi(token=token)
            # The listing has to come from the same revision the files are
            # read from, or selection is computed against other contents.
            repo_files = api.list_repo_files(path, repo_type="dataset", revision=revision)

            # Find data files matching the subset name
            subset = source.subset or ""
            candidates = [
                f
                for f in repo_files
                if subset in f and f.rsplit(".", 1)[-1] in _SUPPORTED_EXTENSIONS
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No data files matching subset '{subset}' in {path}. "
                    f"This dataset has a legacy loading script that is no longer supported."
                )

        module = _loader_module(candidates, path)
        at_revision = f"@{revision}" if revision else ""
        data_urls = [f"hf://datasets/{path}{at_revision}/{f}" for f in candidates]

        return load_dataset(
            module,
            data_files={source.split: data_urls},
            split=source.split,
            streaming=streaming,
            token=token,
        )
