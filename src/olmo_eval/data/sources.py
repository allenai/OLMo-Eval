"""Data source abstraction for dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from urllib.parse import parse_qs, urlparse

from olmo_eval.common.repr import hide_unset


class SourceType(Enum):
    """Supported data source types."""

    HF = "hf"
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"


@hide_unset()
@dataclass(frozen=True)
class DataSource:
    """Identifies a dataset location.

    Supports multiple source types with automatic detection:
    - HuggingFace: "hf://cais/mmlu" or "cais/mmlu" (org/repo format)
    - S3: "s3://bucket/path/to/file.jsonl"
    - GCS: "gs://bucket/path/to/file.parquet"
    - Local: "/path/to/file.jsonl" or "file:///path/to/file.jsonl"

    Examples:
        >>> DataSource(path="cais/mmlu", subset="abstract_algebra")
        >>> DataSource.from_uri("hf://cais/mmlu?subset=abstract_algebra&split=test")
        >>> DataSource.from_uri("s3://bucket/datasets/mmlu.jsonl")
        >>> DataSource.from_uri("/local/path/to/dataset.jsonl")
    """

    path: str
    subset: str | None = None
    split: str = "test"
    source_type: SourceType | None = None
    #: Data file(s) to load, relative to the repository root. A tuple selects
    #: several files that together form one split.
    data_files: str | tuple[str, ...] | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        """Auto-detect source type from path if not specified."""
        if self.source_type is None:
            object.__setattr__(self, "source_type", self._detect_type())

    def _detect_type(self) -> SourceType:
        """Detect source type from path format."""
        if self.path.startswith("hf://"):
            return SourceType.HF
        elif self.path.startswith("s3://"):
            return SourceType.S3
        elif self.path.startswith("gs://"):
            return SourceType.GCS
        elif (
            self.path.startswith("file://")
            or self.path.startswith("/")
            or self.path.startswith("./")
        ):
            return SourceType.LOCAL
        elif "/" in self.path and self.path.count("/") == 1 and not self.path.endswith("/"):
            # Looks like org/repo format (HuggingFace)
            return SourceType.HF
        elif "." in self.path.split("/")[-1]:
            # Has file extension, treat as local
            return SourceType.LOCAL
        else:
            # Default to HuggingFace for simple names
            return SourceType.HF

    @classmethod
    def from_uri(cls, uri: str, **kwargs) -> DataSource:
        """Parse a URI into a DataSource.

        Supported URI formats:
            hf://cais/mmlu?subset=abstract_algebra&split=test
            s3://bucket/datasets/mmlu.jsonl
            gs://bucket/datasets/mmlu.parquet
            /local/path/to/dataset.jsonl
            file:///local/path/to/dataset.jsonl

        Args:
            uri: The URI to parse.
            **kwargs: Override any parsed values.

        Returns:
            A DataSource instance.
        """
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()

        # Determine source type and path
        if scheme == "hf":
            source_type = SourceType.HF
            path = parsed.netloc + parsed.path if parsed.netloc else parsed.path.lstrip("/")
        elif scheme == "s3":
            source_type = SourceType.S3
            path = uri  # Keep full s3:// URI
        elif scheme == "gs":
            source_type = SourceType.GCS
            path = uri  # Keep full gs:// URI
        elif scheme == "file":
            source_type = SourceType.LOCAL
            path = parsed.path
        elif scheme == "":
            # No scheme - detect from path
            source_type = None
            path = uri.split("?")[0]  # Remove query string for detection
        else:
            raise ValueError(f"Unsupported URI scheme: {scheme}")

        # Parse query parameters
        query_params = parse_qs(parsed.query)
        subset = query_params.get("subset", [None])[0]
        split = query_params.get("split", ["test"])[0]

        # Apply overrides from kwargs
        return cls(
            path=kwargs.get("path", path),
            subset=kwargs.get("subset", subset),
            split=kwargs.get("split", split),
            source_type=kwargs.get("source_type", source_type),
            data_files=kwargs.get("data_files"),
            revision=kwargs.get("revision"),
        )

    def with_split(self, split: str) -> DataSource:
        """Return a new DataSource with a different split."""
        return DataSource(
            path=self.path,
            subset=self.subset,
            split=split,
            source_type=self.source_type,
            data_files=self.data_files,
            revision=self.revision,
        )

    def with_subset(self, subset: str | None) -> DataSource:
        """Return a new DataSource with a different subset."""
        return DataSource(
            path=self.path,
            subset=subset,
            split=self.split,
            source_type=self.source_type,
            data_files=self.data_files,
            revision=self.revision,
        )

    def to_dict(self) -> dict:
        """Serialize to a dictionary."""
        return {
            "path": self.path,
            "subset": self.subset,
            "split": self.split,
            "source_type": self.source_type.value if self.source_type else None,
            "data_files": list(self.data_files)
            if isinstance(self.data_files, tuple)
            else self.data_files,
            "revision": self.revision,
        }
