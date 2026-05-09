"""V3.3 base protocols for diagnostic and decision modules.

Two roles:
  - DiagnosticSink: read-only observers (NoTradeLogger, TCMonitor, ExecutionQuality)
  - DecisionPolicy: action-affecting policies (EdgeTier, Pyramid, Rotation, ...)

Both are runtime_checkable so isinstance() works for duck-typed implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DiagnosticSink(Protocol):
    """Read-only observer protocol. Records events without affecting decisions."""

    name: str

    def record(self, event: dict) -> None:
        """Record a single event. Called per candidate / per action / per session."""
        ...

    def flush(self) -> None:
        """Persist buffered events. Called at session boundary or shutdown."""
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Action-affecting policy protocol. Each policy has a feature flag."""

    name: str

    @property
    def enabled(self) -> bool:
        """Whether this policy is currently active (driven by FeatureFlags)."""
        ...
