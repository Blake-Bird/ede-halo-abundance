"""Scale-dependent linear growth diagnostics for Early Dark Energy cosmologies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.cosmology.boltzmann.result import MatterPowerGrid


@dataclass(slots=True, frozen=True)
class GrowthDiagnostics:
    """Scale-dependent linear growth metrics evaluated on a MatterPowerGrid.

    Parameters
    ----------
    k_h_mpc:
        Wavenumber array in :math:`h/\\mathrm{Mpc}`.
    redshifts:
        Redshift evaluation array.
    d_eff_grid:
        2D array of effective growth factors :math:`D_{\\rm eff}(k, z) = \\sqrt{P(k, z)/P(k, 0)}`,
        shape ``(num_z, num_k)``.
    epsilon_d_grid:
        2D array of scale-dependence diagnostics
        :math:`\\epsilon_D(k, z) = D_{\\rm eff}(k, z) / D_{\\rm eff}(k_{\\rm ref}, z) - 1`,
        shape ``(num_z, num_k)``.
    k_ref_h_mpc:
        Reference wavenumber used for scale-dependence normalization (default :math:`0.05\\,h/\\mathrm{Mpc}`).
    """

    k_h_mpc: FloatArray
    redshifts: FloatArray
    d_eff_grid: FloatArray
    epsilon_d_grid: FloatArray
    k_ref_h_mpc: float = 0.05

    @classmethod
    def from_power_grid(
        cls,
        grid: MatterPowerGrid,
        k_ref_h_mpc: float = 0.05,
    ) -> GrowthDiagnostics:
        """Compute D_eff(k, z) and scale-dependence epsilon_D(k, z) from a MatterPowerGrid."""
        k = grid.k_h_mpc
        z_arr = grid.redshifts
        num_z = z_arr.size
        num_k = k.size

        # Find index of z=0
        present = np.flatnonzero(z_arr == 0)
        if present.size != 1:
            raise ValueError("growth normalization requires a z=0 power slice")
        if not np.isfinite(k_ref_h_mpc) or not k[0] <= k_ref_h_mpc <= k[-1]:
            raise ValueError("reference wavenumber lies outside power support")
        idx_z0 = int(present[0])
        pk_z0 = grid.pk_grid[idx_z0]

        d_eff = np.zeros((num_z, num_k), dtype=np.float64)
        for i in range(num_z):
            d_eff[i, :] = np.sqrt(grid.pk_grid[i] / pk_z0)

        # Locate reference wavenumber
        d_ref = np.exp(PchipInterpolator(np.log(k), np.log(d_eff), axis=1)(np.log(k_ref_h_mpc)))[
            :, None
        ]

        epsilon_d = d_eff / d_ref - 1.0

        return cls(
            k_h_mpc=k,
            redshifts=z_arr,
            d_eff_grid=d_eff,
            epsilon_d_grid=epsilon_d,
            k_ref_h_mpc=k_ref_h_mpc,
        )

    def linear_growth_rate(self, k_eval: float = 0.05) -> tuple[FloatArray, FloatArray]:
        """Compute f(a) = d ln D / d ln a at wavenumber k_eval using PCHIP splines."""
        if not np.isfinite(k_eval) or not self.k_h_mpc[0] <= k_eval <= self.k_h_mpc[-1]:
            raise ValueError("growth-rate wavenumber lies outside power support")
        if self.redshifts.size < 3:
            raise ValueError("growth-rate estimation requires at least three redshifts")
        d_k = np.exp(
            PchipInterpolator(np.log(self.k_h_mpc), np.log(self.d_eff_grid), axis=1)(np.log(k_eval))
        )

        a_arr = 1.0 / (1.0 + self.redshifts)
        sort_idx = np.argsort(a_arr)
        a_sorted = a_arr[sort_idx]
        d_sorted = d_k[sort_idx]

        ln_a = np.log(a_sorted)
        ln_d = np.log(d_sorted)

        spline = PchipInterpolator(ln_a, ln_d)
        f_a = np.asarray(spline.derivative()(ln_a), dtype=np.float64)
        return a_sorted, f_a
