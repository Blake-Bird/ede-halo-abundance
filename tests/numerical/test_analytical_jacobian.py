import numpy as np

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import ShethTormen


def _smooth_power() -> TabulatedPowerSpectrum:
    k = np.geomspace(1.0e-6, 1.0e4, 8193)
    pk = 3.0e3 * k / (1.0 + (k / 25.0) ** 8)
    return TabulatedPowerSpectrum(k, pk)


def test_analytical_jacobian_matches_pchip_spline() -> None:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    power = _smooth_power()
    calculator = VarianceCalculator(power)

    masses = np.geomspace(1.0e9, 1.0e15, 64)
    variance_res = calculator.sigma_m(masses, cosmology, compute_derivative=True)

    assert variance_res.dln_sigma_inv_dln_m is not None
    analytical_jac = variance_res.dln_sigma_inv_dln_m

    pchip_jac = HaloMassFunctionCalculator.logarithmic_jacobian(masses, variance_res.sigma)

    # Analytical and PCHIP should closely agree across the smooth interior
    # Median discrepancy is typically < 1e-4; PCHIP discretization error bounds max discrepancy to < 2e-3
    rel_diff = np.abs(analytical_jac[5:-5] - pchip_jac[5:-5]) / analytical_jac[5:-5]
    assert np.median(rel_diff) < 1.0e-4
    assert np.max(rel_diff) < 2.0e-3


def test_hmf_evaluation_with_analytical_jacobian() -> None:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    power = _smooth_power()
    calculator = VarianceCalculator(power)

    masses = np.geomspace(1.0e9, 1.0e15, 64)
    variance_res = calculator.sigma_m(masses, cosmology, compute_derivative=True)
    assert variance_res.dln_sigma_inv_dln_m is not None

    hmf_calc = HaloMassFunctionCalculator(rho_m0=cosmology.rho_m0)
    model = ShethTormen()

    result_analytical = hmf_calc.evaluate(
        masses, variance_res.sigma, model, jacobian=variance_res.dln_sigma_inv_dln_m
    )
    result_pchip = hmf_calc.evaluate(masses, variance_res.sigma, model)

    assert np.all(result_analytical.dn_dln_mass > 0.0)
    assert np.all(np.isfinite(result_analytical.dn_dln_mass))

    # Should match to within PCHIP interpolation tolerance
    np.testing.assert_allclose(
        result_analytical.dn_dln_mass[5:-5],
        result_pchip.dn_dln_mass[5:-5],
        rtol=2.0e-3,
    )


def test_background_with_radiation() -> None:
    cosmology_rad = FlatLambdaCDM(omega_m0=0.315, h=0.674, omega_r0=8.5e-5)
    cosmology_norad = FlatLambdaCDM(omega_m0=0.315, h=0.674, omega_r0=0.0)

    # At z=0 (a=1), radiation is negligible
    np.testing.assert_allclose(cosmology_rad.e(1.0), cosmology_norad.e(1.0), rtol=1.0e-4)

    # At early times a=1e-3, radiation contributes significantly to E(a)
    e_rad = cosmology_rad.e(1.0e-3)
    e_norad = cosmology_norad.e(1.0e-3)
    assert e_rad > e_norad
