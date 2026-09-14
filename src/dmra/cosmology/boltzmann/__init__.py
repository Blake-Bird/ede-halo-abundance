"""Boltzmann solver protocol, domain models, and unit conversion gatekeeper."""

from __future__ import annotations

from dmra.cosmology.boltzmann.manifest import SolverManifest, compute_sha256
from dmra.cosmology.boltzmann.protocol import (
    CANONICAL_REDSHIFTS,
    BoltzmannSolver,
    LinearTheoryRequest,
)
from dmra.cosmology.boltzmann.result import (
    BackgroundHistory,
    LinearTheoryResult,
    MatterPowerGrid,
)
from dmra.cosmology.boltzmann.units import SPEED_OF_LIGHT_KM_S, UnitGatekeeper

__all__ = [
    "CANONICAL_REDSHIFTS",
    "SPEED_OF_LIGHT_KM_S",
    "BackgroundHistory",
    "BoltzmannSolver",
    "LinearTheoryRequest",
    "LinearTheoryResult",
    "MatterPowerGrid",
    "SolverManifest",
    "UnitGatekeeper",
    "compute_sha256",
]
