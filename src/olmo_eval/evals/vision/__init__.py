"""Vision (multimodal) evaluation: tasks, scoring, benchmarks, and data access.

Importing this package registers every vision benchmark, mirroring how
``olmo_eval.evals`` imports ``tasks``.
"""

from olmo_eval.evals.vision import benchmarks as _benchmarks  # noqa: F401

__all__: list[str] = []
