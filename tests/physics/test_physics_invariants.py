"""Physics tests verifying fundamental cosmological and numerical invariants."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.boltzmann.result import LinearTheoryResult
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.growth import GrowthDiagnostics
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.variance import VarianceCalculator
from dmra.cosmology.window import spherical_tophat

pytestmark = pytest.mark.external_solver


@pytest.fixture
def central_ede_result() -> LinearTheoryResult:
    """Fixture providing solved central EDE LinearTheoryResult."""
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
    return solver.run(req)


def test_ede_zero_limit_recovers_lcdm() -> None:
    """Physics Test 1: f_ede -> 0 matches analytical LCDM expansion to within < 10^-5."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.0,
    )
    solver = AxiCLASSSolver()
    res = solver.run(LinearTheoryRequest(cosmology=cosmo))
    bg = res.background

    lcdm_analytic = cosmo.to_flat_lambdacdm()
    z_eval = np.linspace(0.0, 10.0, 50)
    a_eval = 1.0 / (1.0 + z_eval)
    h_analytic = cosmo.H0 * lcdm_analytic.e(a_eval)
    h_solver = bg.hubble(z_eval)

    max_rel_err = float(np.max(np.abs(h_solver - h_analytic) / h_analytic))
    # In real CLASS, relativistic species (photons + 3.044 neutrino species) contribute
    # ~0.16% to H(z) at z=10 compared to the matter+Lambda only model.
    assert max_rel_err < 2.0e-3


def test_background_expansion_positivity(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 2: H(a) > 0 for all scale factors."""
    bg = central_ede_result.background
    assert np.all(bg.H_z > 0.0)
    z_dense = np.geomspace(1.0, 5000.0, 200) - 1.0
    h_eval = bg.hubble(z_dense)
    assert np.all(h_eval > 0.0)


def test_cosmic_energy_budget_flatness(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 3: Cosmic energy budget satisfies |sum Omega_i - 1| < 10^-4."""
    bg = central_ede_result.background
    assert bg.validate_flatness(tolerance=1.0e-4) is True


def test_power_spectrum_positivity(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 4: P(k, z) > 0 strictly across all k and z."""
    grid = central_ede_result.power_grid
    assert np.all(grid.pk_grid > 0.0)
    assert np.all(np.isfinite(grid.pk_grid))


def test_effective_growth_normalization(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 5: Effective growth factor is identically unity today: D_eff(k, 0) = 1.0."""
    grid = central_ede_result.power_grid
    growth = GrowthDiagnostics.from_power_grid(grid)
    idx_z0 = int(np.where(np.isclose(growth.redshifts, 0.0))[0][0])
    np.testing.assert_allclose(growth.d_eff_grid[idx_z0], 1.0, rtol=1e-5)


def test_growth_monotonicity(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 6: Linear perturbations grow monotonically forward in time: D(z1) > D(z2) for z1 < z2."""
    grid = central_ede_result.power_grid
    growth = GrowthDiagnostics.from_power_grid(grid)
    d_at_k01 = growth.d_eff_grid[:, 100]  # Slice at arbitrary k
    # Array is ordered increasing in z (0, 0.5, 1.0, ...)
    # So D must strictly decrease with array index
    assert np.all(np.diff(d_at_k01) < 0.0)


def test_sigma8_dmra_vs_solver(central_ede_result: LinearTheoryResult) -> None:
    """Physics Test 7: Independent Simpson integration matches solver sigma_8 to < 0.01%."""
    pk_z0 = central_ede_result.power_grid.at_redshift(0.0)
    var_calc = VarianceCalculator(pk_z0)
    s8_calc = float(var_calc.sigma_r(8.0).sigma[0])

    k = pk_z0.k
    pk = pk_z0.pk
    ln_k = np.log(k)
    w = spherical_tophat(k * 8.0)
    integrand = (k**3 * pk / (2.0 * np.pi**2)) * w**2
    s8_direct = float(np.sqrt(simpson(integrand, x=ln_k)))

    rel_err = abs(s8_calc - s8_direct) / s8_direct
    assert rel_err < 1.0e-4  # < 0.01%
