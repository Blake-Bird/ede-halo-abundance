"""Halo-scale cosmological descriptors: mass variance sigma(M), peak height nu, and effective spectral slope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.constants import DEFAULT_DELTA_C
from dmra.cosmology.boltzmann.result import MatterPowerGrid
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.variance import VarianceCalculator


@dataclass(slots=True, frozen=True)
class HaloDescriptorsResult:
    """Halo-scale cosmological properties evaluated across mass and redshift.

    Parameters
    ----------
    mass:
        Lagrangian halo mass array in :math:`M_\\odot / h`.
    radius:
        Top-hat smoothing radius in :math:`\\mathrm{Mpc} / h`.
    redshift:
        Redshift of evaluation.
    sigma:
        Top-hat mass variance :math:`\\sigma(M, z)`.
    dln_sigma_inv_dln_m:
        Logarithmic Jacobian :math:`|d\\ln\\sigma^{-1} / d\\ln M|`.
    nu:
        Peak height :math:`\\nu(M, z) = \\delta_c(z) / \\sigma(M, z)`.
    n_eff:
        Effective local spectral index :math:`n_{\\rm eff}(M) = d\\ln P / d\\ln k` at :math:`k_R = 1/R(M)`.
    """

    mass: FloatArray
    radius: FloatArray
    redshift: float
    sigma: FloatArray
    dln_sigma_inv_dln_m: FloatArray
    nu: FloatArray
    n_eff: FloatArray


class HaloScaleDescriptors:
    """Computes halo-scale variance, peak height, and spectral slope from a MatterPowerGrid."""

    def __init__(
        self,
        power_grid: MatterPowerGrid,
        cosmology: EDECosmology,
        delta_c: float = DEFAULT_DELTA_C,
    ) -> None:
        if not np.isfinite(delta_c) or delta_c <= 0:
            raise ValueError("delta_c must be finite and positive")
        self.power_grid = power_grid
        self.cosmology = cosmology
        self.delta_c = delta_c
        self.background_lcdm = cosmology.to_flat_lambdacdm()

    def evaluate(
        self,
        masses: ArrayLike,
        redshift: float = 0.0,
    ) -> HaloDescriptorsResult:
        """Evaluate all halo-scale descriptors across input mass array at specified redshift."""
        mass_arr = np.atleast_1d(np.asarray(masses, dtype=np.float64))
        if (
            mass_arr.ndim != 1
            or mass_arr.size == 0
            or np.any(~np.isfinite(mass_arr))
            or np.any(mass_arr <= 0.0)
        ):
            raise ValueError("masses must be a nonempty one-dimensional finite positive array")

        pk_slice = self.power_grid.at_redshift(redshift)
        var_calc = VarianceCalculator(pk_slice)

        var_res = var_calc.sigma_m(mass_arr, self.background_lcdm, compute_derivative=True)
        assert var_res.dln_sigma_inv_dln_m is not None

        # Peak height nu = delta_c / sigma
        nu = self.delta_c / var_res.sigma

        # Local spectral index n_eff at k_R = 1 / R
        r_arr = var_res.radius
        k_r = 1.0 / r_arr

        ln_k = np.log(pk_slice.k)
        ln_pk = np.log(pk_slice.pk)
        p_spline = PchipInterpolator(ln_k, ln_pk)

        if np.any(k_r < pk_slice.k[0]) or np.any(k_r > pk_slice.k[-1]):
            raise ValueError("1/R lies outside power support; cannot estimate local spectral slope")
        n_eff = np.asarray(p_spline.derivative()(np.log(k_r)), dtype=np.float64)

        return HaloDescriptorsResult(
            mass=mass_arr,
            radius=r_arr,
            redshift=float(redshift),
            sigma=var_res.sigma,
            dln_sigma_inv_dln_m=var_res.dln_sigma_inv_dln_m,
            nu=nu,
            n_eff=n_eff,
        )
