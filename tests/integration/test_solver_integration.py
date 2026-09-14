"""Integration tests verifying Boltzmann solver execution, cross-solver consistency, and end-to-end pipelines."""

from __future__ import annotations

import numpy as np
import pytest

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.class_ede import CLASSEDESolver
from dmra.cosmology.ede.descriptors import HaloScaleDescriptors
from dmra.cosmology.ede.model import EDECosmology

pytestmark = pytest.mark.external_solver


def test_axiclass_smoke() -> None:
    """Integration Test 1: Runs minimal AxiCLASS execution."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736, f_ede=0.08)
    req = LinearTheoryRequest(cosmology=cosmo, redshifts=(0.0, 1.0))
    solver = AxiCLASSSolver()
    res = solver.run(req)

    assert res.manifest.solver_name == "AxiCLASS"
    assert res.background.z.size > 10
    assert res.power_grid.num_redshifts == 2


def test_class_ede_smoke() -> None:
    """Integration Test 2: Runs minimal CLASS_EDE execution."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736, f_ede=0.08)
    req = LinearTheoryRequest(cosmology=cosmo, redshifts=(0.0, 1.0))
    solver = CLASSEDESolver()
    res = solver.run(req)

    assert res.manifest.solver_name == "CLASS_EDE"
    assert res.background.z.size > 10
    assert res.power_grid.num_redshifts == 2


def test_lcdm_cross_solver() -> None:
    """Integration Test 3: Cross-solver agreement between AxiCLASS and CLASS_EDE on LCDM."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736, f_ede=0.0)
    req = LinearTheoryRequest(cosmology=cosmo, redshifts=(0.0, 1.0))

    res_axi = AxiCLASSSolver().run(req)
    res_ede = CLASSEDESolver().run(req)

    # Background comparison
    z_eval = np.linspace(0.0, 5.0, 20)
    h_axi = res_axi.background.hubble(z_eval)
    h_ede = res_ede.background.hubble(z_eval)
    np.testing.assert_allclose(h_axi, h_ede, rtol=1e-4)

    # Power spectrum comparison
    pk_axi = res_axi.power_grid.at_redshift(0.0).pk
    pk_ede = res_ede.power_grid.at_redshift(0.0).pk
    np.testing.assert_allclose(pk_axi, pk_ede, rtol=1e-4)


def test_ede_cross_solver() -> None:
    """Integration Test 4: Cross-solver agreement between AxiCLASS and CLASS_EDE on EDE."""
    cosmo = EDECosmology(
        omega_b=0.02260,
        omega_cdm=0.1300,
        h=0.7210,
        f_ede=0.10,
        log10_z_c=3.55,
        theta_i=2.83,
    )
    req = LinearTheoryRequest(cosmology=cosmo, redshifts=(0.0, 1.0))

    res_axi = AxiCLASSSolver().run(req)
    res_ede = CLASSEDESolver().run(req)

    # Background comparison
    z_eval = np.linspace(0.0, 5.0, 20)
    h_axi = res_axi.background.hubble(z_eval)
    h_ede = res_ede.background.hubble(z_eval)
    np.testing.assert_allclose(h_axi, h_ede, rtol=1e-3)


def test_linear_to_variance_pipeline() -> None:
    """Integration Test 5: End-to-end pipeline execution: theta_EDE -> P(k, z) -> sigma(M, z) -> nu."""
    cosmo = EDECosmology(
        omega_b=0.02260,
        omega_cdm=0.1300,
        h=0.7210,
        f_ede=0.10,
        log10_z_c=3.55,
        theta_i=2.83,
    )
    req = LinearTheoryRequest(cosmology=cosmo)
    solver = AxiCLASSSolver()
    res = solver.run(req)

    desc = HaloScaleDescriptors(res.power_grid, cosmo)
    masses = np.geomspace(1e10, 1e15, 10)
    res_desc = desc.evaluate(masses, redshift=1.0)

    assert res_desc.mass.size == 10
    assert np.all(res_desc.sigma > 0.0)
    assert np.all(res_desc.nu > 0.0)
    assert np.all(res_desc.dln_sigma_inv_dln_m > 0.0)
    # Variance must decrease with mass
    assert np.all(np.diff(res_desc.sigma) < 0.0)
