"""Tests for the DataSource abstraction."""

from olmo_eval.data import DataSource, SourceType


class TestDataSource:
    """Tests for DataSource dataclass."""

    def test_huggingface_detection_org_repo(self):
        source = DataSource(path="cais/mmlu")
        assert source.source_type == SourceType.HF

    def test_huggingface_detection_hf_prefix(self):
        source = DataSource(path="hf://cais/mmlu")
        assert source.source_type == SourceType.HF

    def test_huggingface_detection_simple_name(self):
        source = DataSource(path="gsm8k")
        assert source.source_type == SourceType.HF

    def test_s3_detection(self):
        source = DataSource(path="s3://bucket/path/data.jsonl")
        assert source.source_type == SourceType.S3

    def test_gcs_detection(self):
        source = DataSource(path="gs://bucket/path/data.jsonl")
        assert source.source_type == SourceType.GCS

    def test_local_detection_absolute_path(self):
        source = DataSource(path="/path/to/data.jsonl")
        assert source.source_type == SourceType.LOCAL

    def test_local_detection_relative_path(self):
        source = DataSource(path="./data/file.jsonl")
        assert source.source_type == SourceType.LOCAL

    def test_local_detection_file_prefix(self):
        source = DataSource(path="file:///path/to/data.jsonl")
        assert source.source_type == SourceType.LOCAL

    def test_explicit_source_type(self):
        source = DataSource(path="some/path", source_type=SourceType.S3)
        assert source.source_type == SourceType.S3

    def test_subset_and_split(self):
        source = DataSource(
            path="cais/mmlu",
            subset="abstract_algebra",
            split="validation",
        )
        assert source.path == "cais/mmlu"
        assert source.subset == "abstract_algebra"
        assert source.split == "validation"


class TestDataSourceFromUri:
    """Tests for DataSource.from_uri() parsing."""

    def test_huggingface_uri(self):
        source = DataSource.from_uri("hf://cais/mmlu?subset=abstract_algebra&split=test")
        assert source.path == "cais/mmlu"
        assert source.subset == "abstract_algebra"
        assert source.split == "test"
        assert source.source_type == SourceType.HF

    def test_huggingface_uri_no_query(self):
        source = DataSource.from_uri("hf://cais/mmlu")
        assert source.path == "cais/mmlu"
        assert source.subset is None
        assert source.split == "test"

    def test_s3_uri(self):
        source = DataSource.from_uri("s3://bucket/path/to/data.jsonl")
        assert source.path == "s3://bucket/path/to/data.jsonl"
        assert source.source_type == SourceType.S3

    def test_gcs_uri(self):
        source = DataSource.from_uri("gs://bucket/path/to/data.parquet")
        assert source.path == "gs://bucket/path/to/data.parquet"
        assert source.source_type == SourceType.GCS

    def test_file_uri(self):
        source = DataSource.from_uri("file:///path/to/data.jsonl")
        assert source.path == "/path/to/data.jsonl"
        assert source.source_type == SourceType.LOCAL

    def test_local_path_no_scheme(self):
        source = DataSource.from_uri("/path/to/data.jsonl")
        assert source.path == "/path/to/data.jsonl"
        assert source.source_type == SourceType.LOCAL

    def test_kwargs_override(self):
        source = DataSource.from_uri(
            "hf://cais/mmlu?subset=math&split=train",
            split="dev",
        )
        assert source.split == "dev"
        assert source.subset == "math"


class TestDataSourceMethods:
    """Tests for DataSource helper methods."""

    def test_with_split(self):
        source = DataSource(path="cais/mmlu", split="test")
        new_source = source.with_split("train")
        assert new_source.split == "train"
        assert source.split == "test"
        assert new_source.path == source.path

    def test_with_subset(self):
        source = DataSource(path="cais/mmlu", subset="math")
        new_source = source.with_subset("biology")
        assert new_source.subset == "biology"
        assert source.subset == "math"
        assert new_source.path == source.path

    def test_with_subset_none(self):
        source = DataSource(path="cais/mmlu", subset="math")
        new_source = source.with_subset(None)
        assert new_source.subset is None


class TestDataSourceDataFiles:
    """Tests for selecting specific data files."""

    def test_multiple_data_files_survive_split_and_subset_changes(self):
        source = DataSource(path="org/repo", data_files=("a.jsonl", "b.jsonl"))
        assert source.with_split("train").data_files == ("a.jsonl", "b.jsonl")
        assert source.with_subset("other").data_files == ("a.jsonl", "b.jsonl")

    def test_multiple_data_files_serialize_as_a_list(self):
        source = DataSource(path="org/repo", data_files=("a.jsonl", "b.jsonl"))
        assert source.to_dict()["data_files"] == ["a.jsonl", "b.jsonl"]

    def test_single_data_file_serializes_unchanged(self):
        source = DataSource(path="org/repo", data_files="a.jsonl")
        assert source.to_dict()["data_files"] == "a.jsonl"

    def test_source_with_multiple_data_files_is_hashable(self):
        source = DataSource(path="org/repo", data_files=("a.jsonl", "b.jsonl"))
        assert hash(source) == hash(DataSource(path="org/repo", data_files=("a.jsonl", "b.jsonl")))
