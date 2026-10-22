"""Sandbox infrastructure exception types."""

from olmo_eval.common.scorers.execution import SandboxInfrastructureError


class SandboxTransportError(SandboxInfrastructureError):
    """A logical sandbox request exhausted all transport retries."""


__all__ = ["SandboxInfrastructureError", "SandboxTransportError"]
