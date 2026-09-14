"""Protocol and request abstractions for linear Boltzmann solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from dmra.cosmology.boltzmann.result import LinearTheoryResult
from dmra.cosmology.ede.model import EDECosmology

CANONICAL_REDSHIFTS: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
)


@dataclass(slots=True, frozen=True)
class LinearTheoryRequest:
    """Specification of cosmological perturbation calculations requested from a Boltzmann engine.

    Parameters
    ----------
    cosmology:
        The target EDE or LCDM physical domain model.
    redshifts:
        Evaluation redshifts for power spectra and transfer functions.
    k_min_h_mpc:
        Minimum wavenumber in :math:`h/\\mathrm{Mpc}`. Default is :math:`10^{-4}`.
    k_max_h_mpc:
        Maximum wavenumber in :math:`h/\\mathrm{Mpc}`. Default is :math:`10^2`.
    num_k_points:
        Number of logarithmic sampling points in wavenumber. Default is 500.
    transfer_types:
        Tuple of requested transfer components (e.g., ``('total_matter',)``, ``('cb',)``).
    gauge:
        Gauge formulation (``'synchronous'`` or ``'newtonian'``).
    """

    cosmology: EDECosmology
    redshifts: tuple[float, ...] = CANONICAL_REDSHIFTS
    k_min_h_mpc: float = 1.0e-4
    k_max_h_mpc: float = 1.0e2
    num_k_points: int = 500
    transfer_types: tuple[str, ...] = ("total_matter",)
    gauge: Literal["synchronous", "newtonian"] = "synchronous"

    def __post_init__(self) -> None:
        if not np.isfinite(self.k_min_h_mpc) or self.k_min_h_mpc <= 0.0:
            raise ValueError(f"k_min_h_mpc must be positive and finite, got {self.k_min_h_mpc}")
        if not np.isfinite(self.k_max_h_mpc) or self.k_max_h_mpc <= self.k_min_h_mpc:
            raise ValueError(
                f"k_max_h_mpc must be strictly greater than k_min_h_mpc, got {self.k_max_h_mpc}"
            )
        if not isinstance(self.num_k_points, int) or self.num_k_points < 8:
            raise ValueError(f"num_k_points must be an integer >= 8, got {self.num_k_points}")
        if len(self.redshifts) == 0:
            raise ValueError("redshifts tuple cannot be empty")
        if len(set(self.redshifts)) != len(self.redshifts):
            raise ValueError("redshifts must be unique")
        if self.transfer_types != ("total_matter",):
            raise ValueError("only total_matter is currently supported")
        for z in self.redshifts:
            if not np.isfinite(z) or z < 0.0:
                raise ValueError(f"redshift values must be finite and non-negative, got {z}")
        if self.gauge not in ("synchronous", "newtonian"):
            raise ValueError(f"gauge must be 'synchronous' or 'newtonian', got '{self.gauge}'")


@runtime_checkable
class BoltzmannSolver(Protocol):
    """Abstract interface satisfied by all Boltzmann solver adapters."""

    @property
    def name(self) -> str:
        """Name of the Boltzmann engine."""
        ...

    @property
    def version(self) -> str:
        """Version identifier or git commit hash."""
        ...

    def run(self, request: LinearTheoryRequest) -> LinearTheoryResult:
        """Execute linear theory calculations for the requested cosmology."""
        ...
