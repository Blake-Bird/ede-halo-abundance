"""Top-hat-smoothed linear variance with support diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson

from dmra._typing import FloatArray
from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.window import spherical_tophat, spherical_tophat_derivative


@dataclass(frozen=True, slots=True)
class VarianceResult:
    """Variance values plus diagnostics for finite k-domain truncation."""

    radius: FloatArray
    sigma: FloatArray
    low_k_edge_ratio: FloatArray
    high_k_edge_ratio: FloatArray
    dln_sigma_inv_dln_r: FloatArray | None = None
    dln_sigma_inv_dln_m: FloatArray | None = None

    @property
    def max_edge_ratio(self) -> FloatArray:
        """Larger of the low- and high-k edge contributions for each radius."""
        return np.maximum(self.low_k_edge_ratio, self.high_k_edge_ratio)


class VarianceCalculator:
    r"""Compute :math:`\sigma(R)` from a tabulated linear power spectrum.

    Integration is carried out in ``ln(k)``:

    .. math::

       \sigma^2(R) = \int d\ln k\;\Delta^2(k) W^2(kR).

    The calculator also computes the exact analytical logarithmic derivative:

    .. math::

       R\frac{d\sigma^2}{dR} = 2\int d\ln k\;\Delta^2(k) W(kR)\left[kR\,W'(kR)\right],

    yielding the analytical Jacobian :math:`d\ln\sigma^{-1}/d\ln M = -(R/6\sigma^2) d\sigma^2/dR`
    without requiring numerical spline differentiation of discrete mass points.
    """

    def __init__(self, power: TabulatedPowerSpectrum, *, chunk_size: int = 256) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._power = power
        self._chunk_size = chunk_size
        self._lnk = np.log(power.k)
        self._delta2 = power.dimensionless_power

    @property
    def power(self) -> TabulatedPowerSpectrum:
        return self._power

    def sigma_r(self, radius: ArrayLike, *, compute_derivative: bool = True) -> VarianceResult:
        """Evaluate ``sigma(R)`` for one or more positive radii."""
        radii = np.atleast_1d(np.asarray(radius, dtype=np.float64))
        if radii.ndim != 1 or radii.size == 0:
            raise ValueError("radius must be nonempty and one-dimensional")
        if np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("radius must be finite and positive")

        sigma2 = np.empty_like(radii)
        low_edge = np.empty_like(radii)
        high_edge = np.empty_like(radii)
        r_dsigma2_dr = np.empty_like(radii) if compute_derivative else None

        for start in range(0, radii.size, self._chunk_size):
            stop = min(start + self._chunk_size, radii.size)
            r = radii[start:stop]
            x = r[:, None] * self._power.k[None, :]
            window = spherical_tophat(x)
            integrand = self._delta2[None, :] * window**2

            sigma2[start:stop] = simpson(integrand, x=self._lnk, axis=1)
            peak = np.max(integrand, axis=1)
            low_edge[start:stop] = integrand[:, 0] / peak
            high_edge[start:stop] = integrand[:, -1] / peak

            if compute_derivative and r_dsigma2_dr is not None:
                window_deriv = spherical_tophat_derivative(x)
                deriv_integrand = self._delta2[None, :] * (2.0 * window * x * window_deriv)
                r_dsigma2_dr[start:stop] = simpson(deriv_integrand, x=self._lnk, axis=1)

        if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            raise FloatingPointError(
                "variance integration produced a non-positive or non-finite result"
            )

        dln_sigma_inv_dln_r: FloatArray | None = None
        dln_sigma_inv_dln_m: FloatArray | None = None
        if compute_derivative and r_dsigma2_dr is not None:
            # d(ln sigma^-1) / d(ln R) = - R / (2 * sigma^2) * d(sigma^2)/dR
            dln_sigma_inv_dln_r = -0.5 * r_dsigma2_dr / sigma2
            # Since M ~ R^3, d(ln M) = 3 * d(ln R), so d(ln sigma^-1) / d(ln M) = (1/3) * d(ln sigma^-1) / d(ln R)
            dln_sigma_inv_dln_m = dln_sigma_inv_dln_r / 3.0

        return VarianceResult(
            radius=radii,
            sigma=np.sqrt(sigma2),
            low_k_edge_ratio=low_edge,
            high_k_edge_ratio=high_edge,
            dln_sigma_inv_dln_r=dln_sigma_inv_dln_r,
            dln_sigma_inv_dln_m=dln_sigma_inv_dln_m,
        )

    def sigma_m(
        self, mass: ArrayLike, cosmology: FlatLambdaCDM, *, compute_derivative: bool = True
    ) -> VarianceResult:
        """Evaluate ``sigma(M)`` using the Lagrangian top-hat mass-radius map."""
        return self.sigma_r(cosmology.mass_to_radius(mass), compute_derivative=compute_derivative)
