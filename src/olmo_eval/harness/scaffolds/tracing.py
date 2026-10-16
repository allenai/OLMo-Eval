"""Trace output handling for agent execution."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from agents import set_trace_processors
from agents.tracing.processor_interface import TracingExporter
from agents.tracing.processors import BatchTraceProcessor

logger = logging.getLogger(__name__)


class FileSpanExporter(TracingExporter):
    """Exports traces to per-trace JSONL files.

    Each unique trace_id gets its own file. Since each agent run has a unique
    trace_id, concurrent agents write to separate files with no contention.
    Files are named: {output_dir}/traces/{trace_name}_{hash6}.jsonl
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = os.path.join(output_dir, "traces")
        # Cache trace_id -> filename to ensure consistent naming across batches
        self._trace_files: dict[str, str] = {}

    def _get_trace_info(self, item: Any) -> tuple[str, str | None]:
        """Extract trace_id and trace_name from an item.

        The trace_name is expected to be in format "{config_name}:{instance_id}"
        which provides all the info needed for a descriptive filename.
        """
        trace_id = "unknown"
        trace_name = None

        if hasattr(item, "trace_id"):
            trace_id = str(item.trace_id)
        if hasattr(item, "name"):
            trace_name = str(item.name)

        if hasattr(item, "export"):
            data = item.export()
            if isinstance(data, dict):
                if "trace_id" in data:
                    trace_id = str(data["trace_id"])
                if "name" in data:
                    trace_name = str(data["name"])

        return trace_id, trace_name

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in a filename."""
        # Replace spaces and special chars with underscores
        import re

        sanitized = re.sub(r"[^\w\-]", "_", name)
        # Collapse multiple underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized.strip("_").lower()

    def _get_filename(self, trace_id: str, trace_name: str | None) -> str:
        """Get or create a filename for a trace.

        Expected trace_name format: "{config_name}:{instance_id}"
        Output filename: "{config_name}_{instance_id}_{hash6}.jsonl"
        """
        if trace_id in self._trace_files:
            return self._trace_files[trace_id]

        # Use last 6 chars of trace_id as short hash
        short_hash = trace_id[-6:] if len(trace_id) >= 6 else trace_id

        if trace_name:
            name_part = self._sanitize_name(trace_name)
            filename = f"{name_part}_{short_hash}.jsonl"
        else:
            filename = f"trace_{short_hash}.jsonl"

        self._trace_files[trace_id] = filename
        return filename

    def export(self, items: list[Any]) -> None:
        """Export spans/traces to per-trace files."""
        # exist_ok=True handles any races safely
        os.makedirs(self._output_dir, exist_ok=True)

        # Group items by trace_id and collect trace names
        by_trace: dict[str, list[str]] = {}
        trace_names: dict[str, str | None] = {}

        for item in items:
            trace_id, trace_name = self._get_trace_info(item)

            if hasattr(item, "export"):
                data = item.export()
            elif hasattr(item, "model_dump"):
                data = item.model_dump()
            else:
                data = {"type": type(item).__name__, "str": str(item)}

            if trace_id not in by_trace:
                by_trace[trace_id] = []
                trace_names[trace_id] = trace_name
            elif trace_name and not trace_names[trace_id]:
                # Update name if we find it later
                trace_names[trace_id] = trace_name

            by_trace[trace_id].append(json.dumps(data))

        # Write each trace's items to its own file
        for trace_id, lines in by_trace.items():
            filename = self._get_filename(trace_id, trace_names.get(trace_id))
            file_path = os.path.join(self._output_dir, filename)
            with open(file_path, "a") as f:
                f.write("\n".join(lines) + "\n")

    def shutdown(self) -> None:
        """No-op since we don't keep file handles open."""

    def force_flush(self) -> None:
        """No-op since we write and close immediately."""


@lru_cache(maxsize=1)
def configure_trace_output(output_dir: str) -> None:
    """Configure trace output to write per-agent JSONL files.

    This sets up a BatchTraceProcessor with a FileSpanExporter that writes
    each trace to its own file: {output_dir}/traces/trace_{trace_id}.jsonl

    Called once per worker process at startup before concurrent work begins.
    Uses lru_cache to ensure idempotent configuration.

    Args:
        output_dir: Base directory for trace output.
    """
    exporter = FileSpanExporter(output_dir)
    processor = BatchTraceProcessor(exporter)
    set_trace_processors([processor])
    logger.info(f"Agent traces will be written to {output_dir}/traces/")
