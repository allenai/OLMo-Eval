"""Shared constants and exceptions for evaluation runners."""


class ValidationError(Exception):
    """Raised when validation of runner inputs fails."""

    pass


class HardFailureRateExceeded(Exception):
    """Raised when a task's hard-failure rate exceeds the configured threshold.

    Raised after results have been written, so a run that trips the gate still
    leaves its metrics.json and predictions behind for diagnosis.
    """

    pass
