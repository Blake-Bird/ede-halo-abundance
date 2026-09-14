"""Analytic and empirical halo multiplicity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.special import gamma  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.constants import DEFAULT_DELTA_C
from dmra.statistics.peak_height import peak_height


def _require_positive(values: FloatArray, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


@runtime_checkable
class MultiplicityModel(Protocol):
    """Interface consumed by the generic HMF transformation."""

    @property
    def name(self) -> str: ...

    def evaluate(self, sigma: FloatArray, *, delta_c: float = DEFAULT_DELTA_C) -> FloatArray:
        """Return the dimensionless multiplicity associated with ``sigma``."""
        ...


@dataclass(frozen=True, slots=True)
class PressSchechter:
    """Press-Schechter multiplicity in the convention ``nu = delta_c/sigma``."""

    @property
    def name(self) -> str:
        return "Press-Schechter (1974)"

    def evaluate(self, sigma: FloatArray, *, delta_c: float = DEFAULT_DELTA_C) -> FloatArray:
        sigma_array = _require_positive(sigma, "sigma")
        nu = peak_height(sigma_array, delta_c=delta_c)
        return np.asarray(np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ShethTormen:
    """Sheth-Tormen multiplicity with normalization derived from ``p``."""

    a: float = 0.707
    p: float = 0.3
    A: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.a) or self.a <= 0.0:
            raise ValueError("a must be finite and positive")
        if not np.isfinite(self.p) or not 0.0 <= self.p < 0.5:
            raise ValueError("p must satisfy 0 <= p < 1/2 for finite normalization")
        if self.A is not None and (not np.isfinite(self.A) or self.A <= 0.0):
            raise ValueError("A must be finite and positive when supplied")

    @property
    def name(self) -> str:
        return "Sheth-Tormen (1999)"

    @property
    def normalization(self) -> float:
        """Normalization such that ``integral f(nu) dln(nu) = 1``."""
        if self.A is not None:
            return self.A
        correction = 2.0 ** (-self.p) * gamma(0.5 - self.p) / np.sqrt(np.pi)
        return float(1.0 / (1.0 + correction))

    def evaluate(self, sigma: FloatArray, *, delta_c: float = DEFAULT_DELTA_C) -> FloatArray:
        sigma_array = _require_positive(sigma, "sigma")
        nu = peak_height(sigma_array, delta_c=delta_c)
        a_nu2 = self.a * nu**2
        return np.asarray(
            self.normalization
            * np.sqrt(2.0 * self.a / np.pi)
            * (1.0 + a_nu2 ** (-self.p))
            * nu
            * np.exp(-0.5 * a_nu2),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class Tinker2008Delta200MeanZ0:
    """Tinker et al. (2008) Delta=200-mean, z=0 reference fit.

    This foundation intentionally implements one explicit calibration point. General
    overdensity interpolation and redshift evolution belong in the later
    baseline module where halo-definition bookkeeping is available.
    """

    A: float = 0.186
    a: float = 1.47
    b: float = 2.57
    c: float = 1.19

    def __post_init__(self) -> None:
        if any(not np.isfinite(v) or v <= 0 for v in (self.A, self.a, self.b, self.c)):
            raise ValueError("Tinker coefficients must be finite and positive")

    @property
    def name(self) -> str:
        return "Tinker et al. (2008), Delta=200m, z=0"

    def evaluate(self, sigma: FloatArray, *, delta_c: float = DEFAULT_DELTA_C) -> FloatArray:
        del delta_c  # The published Tinker fit is expressed directly in sigma.
        sigma_array = _require_positive(sigma, "sigma")
        return np.asarray(
            self.A * ((sigma_array / self.b) ** (-self.a) + 1.0) * np.exp(-self.c / sigma_array**2),
            dtype=np.float64,
        )
