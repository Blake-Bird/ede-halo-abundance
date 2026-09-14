"""Integration tests verifying Boltzmann linear theory domain models interoperate with HMF pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.boltzmann.result import MatterPowerGrid
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.variance import VarianceCalculator
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import ShethTormen


def test_linear_theory_to_variance_and_hmf_integration() -> None:
    """Verify that MatterPowerGrid slices plug directly into VarianceCalculator and HMF."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.08,
        log10_z_c=3.55,
        theta_i=2.83,
    )

    # Verify background compatibility
    bg_lcdm = cosmo.to_flat_lambdacdm()
    assert isinstance(bg_lcdm, FlatLambdaCDM)
    assert pytest.approx(bg_lcdm.h) == 0.6736

    # Construct synthetic MatterPowerGrid
    k = np.geomspace(1e-4, 50.0, 300)
    redshifts = np.array([0.0, 0.5, 1.0, 2.0])

    # Power spectrum: P_0(k) ~ k / (1 + (k/0.05)^2)^1.2
    pk_0 = 15000.0 * k / (1.0 + (k / 0.05) ** 2) ** 1.2
    pk_grid = np.array([pk_0 / (1.0 + z) ** 1.8 for z in redshifts])

    grid = MatterPowerGrid(k_h_mpc=k, redshifts=redshifts, pk_grid=pk_grid)

    # Extract power spectrum at z=0.5
    pk_slice_z05 = grid.at_redshift(0.5)

    # Feed slice into VarianceCalculator
    var_calc = VarianceCalculator(pk_slice_z05)
    var_res_r8 = var_calc.sigma_r(8.0)
    assert np.isfinite(var_res_r8.sigma[0])
    assert var_res_r8.sigma[0] > 0.0

    # Evaluate sigma(M) across mass range 10^10 to 10^15 M_sun / h
    masses = np.geomspace(1e10, 1e15, 20)
    var_res_m = var_calc.sigma_m(masses, bg_lcdm, compute_derivative=True)

    # Evaluate HMF
    hmf_calc = HaloMassFunctionCalculator(rho_m0=bg_lcdm.rho_m0)
    hmf_res = hmf_calc.evaluate(
        mass=masses,
        sigma=var_res_m.sigma,
        model=ShethTormen(),
        jacobian=var_res_m.dln_sigma_inv_dln_m,
    )

    assert np.all(np.isfinite(hmf_res.dn_dln_mass))
    assert np.all(hmf_res.dn_dln_mass > 0.0)
    assert np.all(np.diff(hmf_res.dn_dln_mass) < 0.0)
