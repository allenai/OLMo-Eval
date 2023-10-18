import gzip
import json
import os
from copy import deepcopy
from typing import Any, Dict, Sequence

from cached_path import cached_path
from catwalk.task import InstanceFormat, Task

# Task for evaluating a generic set of files (or URLs) on perplexity metrics


class PerplexityJsonLTask(Task):
    def __init__(self, files=None):  # files (or URLs) to be used
        Task.__init__(self)
        self.files = files
        self._cached_paths = None
        self._cache_dir = None  # Can override cache dir
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc)
        self.file_extensions = [".jsonl.gz"]

    def clone(self, files):
        new_task = deepcopy(self)
        new_task.files = files
        return new_task

    def has_split(self, split: str) -> bool:
        return True  # Assume the files are for the requested split

    def cached_paths(self):
        if self._cached_paths is None:
            self._cached_paths = []
            for file in self.files:
                self._cached_paths.append((file, cached_path(file, cache_dir=self._cache_dir)))
        return self._cached_paths

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        instances = []
        all_files = []
        # expand directories if need be
        for orig_file, cache_file in self.cached_paths():
            if os.path.isdir(cache_file):
                for root, dirs, files in os.walk(cache_file):
                    for file in files:
                        for extension in self.file_extensions:
                            if file.endswith(extension):
                                all_files.append((file, os.path.join(root, file)))
            else:
                all_files.append((orig_file, cache_file))

        for orig_file, cache_file in all_files:
            if orig_file.endswith(".gz"):
                with gzip.open(cache_file, "r") as file:
                    for line in file:
                        instance = json.loads(line.decode("utf-8").strip())
                        instance["orig_file_name"] = orig_file
                        instances.append(instance)
            else:
                with open(cache_file, "r") as file:
                    for line in file:
                        instance = json.loads(line.strip())
                        instance["orig_file_name"] = orig_file
                        instances.append(instance)
        return instances

    def instance_as_eleuther_doc(self, instance):
        return instance.get("text", instance.get("doc"))

    @property
    def default_split(self) -> str:
        return "validation"
