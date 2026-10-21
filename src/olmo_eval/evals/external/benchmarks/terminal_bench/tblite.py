"""OpenThoughts-TBLite external evaluation.

A difficulty-calibrated set of 100 terminal tasks in the Terminal-Bench task
format, built to track Terminal-Bench 2 while giving smaller models a usable
signal. Its tasks ship a Dockerfile rather than a prebuilt image.

Repository: https://github.com/open-thoughts/OpenThoughts-TBLite
"""

from __future__ import annotations

from .eval import TerminalBenchDataset, TerminalBenchExternalEval

TBLITE_DATASET = TerminalBenchDataset(
    name="openthoughts_tblite",
    description=(
        "Evaluates LLM agents on OpenThoughts-TBLite, 100 difficulty-calibrated "
        "terminal tasks in the Terminal-Bench format"
    ),
    repo_url="https://github.com/open-thoughts/OpenThoughts-TBLite.git",
    repo_ref="f075e463472c7790b85793b392dff1fff20cc0e3",
    version="2.0",
    cache_dir="/tmp/openthoughts-tblite-cache",
    image_prefix="tblite-task",
)


class OpenThoughtsTBLiteExternalEval(TerminalBenchExternalEval):
    """OpenThoughts-TBLite evaluation, sharing the Terminal-Bench harness."""

    dataset = TBLITE_DATASET
