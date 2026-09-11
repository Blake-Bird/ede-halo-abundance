"""Validated tabulated linear matter power spectra."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray


@dataclass(frozen=True, slots=True)
class TabulatedPowerSpectrum:
    """Positive linear matter power spectrum sampled at increasing wavenumber.

    Parameters
    ----------
    k:
        Wavenumber in ``h / Mpc``.
    pk:
        Linear matter power in ``(Mpc / h)^3``.

    Notes
    -----
    Interpolation is performed in ``(ln k, ln P)`` with PCHIP. PCHIP preserves
    monotonic segments without the overshoot that unconstrained cubic splines
    can introduce. Extrapolation is disabled by default because tail behavior
    materially affects variance integrals.
    """

    k: FloatArray
    pk: FloatArray
    _interpolator: PchipInterpolator = field(init=False, repr=False, compare=False)
    _interpolator_extrap: PchipInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        k = np.asarray(self.k, dtype=np.float64)
        pk = np.asarray(self.pk, dtype=np.float64)

        if k.ndim != 1 or pk.ndim != 1:
            raise ValueError("k and pk must be one-dimensional")
        if k.size != pk.size:
            raise ValueError("k and pk must have the same length")
        if k.size < 8:
            raise ValueError("a power spectrum requires at least eight samples")
        if np.any(~np.isfinite(k)) or np.any(~np.isfinite(pk)):
            raise ValueError("k and pk must be finite")
        if np.any(k <= 0.0) or np.any(pk <= 0.0):
            raise ValueError("k and pk must be strictly positive")
        if np.any(np.diff(k) <= 0.0):
            raise ValueError("k must be strictly increasing")

        log_k = np.log(k)
        log_pk = np.log(pk)

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "pk", pk)
        object.__setattr__(
            self, "_interpolator", PchipInterpolator(log_k, log_pk, extrapolate=False)
        )
        object.__setattr__(
            self, "_interpolator_extrap", PchipInterpolator(log_k, log_pk, extrapolate=True)
        )

    @property
    def k_min(self) -> float:
        return float(self.k[0])

    @property
    def k_max(self) -> float:
        return float(self.k[-1])

    @property
    def dimensionless_power(self) -> FloatArray:
        r"""Return :math:`\Delta^2(k) = k^3 P(k)/(2\pi^2)`."""
        return self.k**3 * self.pk / (2.0 * np.pi**2)

    def evaluate(self, k_query: ArrayLike, *, extrapolate: bool = False) -> FloatArray:
        """Interpolate ``P(k)`` in log-log coordinates.

        Extrapolation is a caller-visible choice. The default rejects requests
        outside the supplied support instead of silently inventing tail power.
        """
        query = np.asarray(k_query, dtype=np.float64)
        if np.any(~np.isfinite(query)) or np.any(query <= 0.0):
            raise ValueError("requested k must be finite and positive")
        if not extrapolate and (np.any(query < self.k_min) or np.any(query > self.k_max)):
            raise ValueError("requested k lies outside the tabulated support")

        interpolator = self._interpolator_extrap if extrapolate else self._interpolator
        return np.asarray(np.exp(interpolator(np.log(query))), dtype=np.float64)

    def rescale_amplitude(self, factor: float) -> TabulatedPowerSpectrum:
        """Return a spectrum with ``P(k)`` multiplied by a positive factor."""
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("amplitude factor must be finite and positive")
        return TabulatedPowerSpectrum(self.k.copy(), self.pk * factor)
