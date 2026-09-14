"""Cryptographic provenance and execution manifest for Boltzmann solvers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def compute_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 hex digest for arbitrary input."""
    if isinstance(data, dict):
        canonical_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
        encoded = canonical_str.encode("utf-8")
    elif isinstance(data, str):
        encoded = data.encode("utf-8")
    elif isinstance(data, bytes):
        encoded = data
    else:
        encoded = str(data).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True, frozen=True)
class SolverManifest:
    """Cryptographic audit trail capturing solver inputs, environment, and runtime metrics.

    Parameters
    ----------
    solver_name:
        Identifier of the Boltzmann engine (e.g., ``'axiclass'``, ``'class_ede'``, ``'mock'``).
    solver_version:
        Git commit hash or canonical release version string.
    config_hash:
        SHA-256 checksum of the exact solver configuration string or dictionary.
    input_parameters:
        Dictionary of serialized input parameters supplied to the solver.
    raw_units:
        Mapping of raw variable names to original solver units prior to ingestion.
    execution_time_seconds:
        Measured wall-clock runtime in seconds.
    stdout_summary:
        Truncated execution log from the solver's standard output stream.
    stderr_summary:
        Truncated execution log from the solver's standard error stream.
    git_commit:
        Current repository commit hash under which the calculation was performed.
    timestamp_utc:
        ISO 8601 UTC timestamp of execution completion.
    """

    solver_name: str
    solver_version: str
    config_hash: str
    input_parameters: dict[str, Any]
    raw_units: dict[str, str]
    execution_time_seconds: float
    stdout_summary: str = ""
    stderr_summary: str = ""
    git_commit: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to a standard JSON-compatible dictionary."""
        return asdict(self)
