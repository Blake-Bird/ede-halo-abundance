import numpy as np

from dmra.cosmology.background import FlatLambdaCDM
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import PressSchechter, ShethTormen, Tinker2008Delta200MeanZ0


def _power_law_sigma(mass: np.ndarray, *, alpha: float = 0.2) -> np.ndarray:
    return 0.9 * (mass / 1.0e12) ** (-alpha)


def test_logarithmic_jacobian_recovers_power_law_exactly() -> None:
    mass = np.geomspace(1.0e8, 1.0e16, 512)
    sigma = _power_law_sigma(mass, alpha=0.23)
    jac = HaloMassFunctionCalculator.logarithmic_jacobian(mass, sigma)
    np.testing.assert_allclose(jac, 0.23, rtol=3.0e-12, atol=2.0e-12)


def test_physical_power_law_index_recovers_exact_jacobian() -> None:
    """For P(k) ~ k^n, sigma(M) ~ M^{-(n+3)/6}, so dln(sigma^-1)/dlnM = (n+3)/6 exactly."""
    mass = np.geomspace(1.0e8, 1.0e16, 512)
    for n in (-2.0, -1.8, -1.5, -1.2, -1.0):
        expected_jacobian = (n + 3.0) / 6.0
        sigma = 1.0 * (mass / 1.0e12) ** (-expected_jacobian)
        computed_jacobian = HaloMassFunctionCalculator.logarithmic_jacobian(mass, sigma)
        np.testing.assert_allclose(computed_jacobian, expected_jacobian, rtol=1.0e-11, atol=1.0e-11)


def test_rare_peak_asymptotic_sensitivity() -> None:
    """Verify analytical identity d ln(f_PS) / d ln(sigma) = nu^2 - 1 ~ nu^2 for rare peaks."""
    model = PressSchechter()
    delta_c = 1.686
    eps = 1.0e-5

    for nu_val in (4.0, 5.0, 6.0):
        sigma_val = delta_c / nu_val
        # Vary sigma by (1 +/- eps)
        sigma_low = sigma_val * (1.0 - eps)
        sigma_high = sigma_val * (1.0 + eps)

        f_low = model.evaluate(np.array([sigma_low]), delta_c=delta_c)[0]
        f_high = model.evaluate(np.array([sigma_high]), delta_c=delta_c)[0]

        dln_f_dln_sigma = (np.log(f_high) - np.log(f_low)) / (2.0 * eps)

        expected_derivative = nu_val**2 - 1.0
        np.testing.assert_allclose(dln_f_dln_sigma, expected_derivative, rtol=1.0e-4)
        # Verify that nu^2 is an accurate asymptotic surrogate within 7% for nu >= 4
        np.testing.assert_allclose(dln_f_dln_sigma, nu_val**2, rtol=0.07)


def test_hmf_models_are_positive() -> None:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    mass = np.geomspace(1.0e8, 1.0e16, 512)
    sigma = _power_law_sigma(mass)
    calculator = HaloMassFunctionCalculator(rho_m0=cosmology.rho_m0)

    for model in (PressSchechter(), ShethTormen(), Tinker2008Delta200MeanZ0()):
        result = calculator.evaluate(mass, sigma, model)
        assert np.all(result.dn_dln_mass > 0.0)
        assert np.all(np.isfinite(result.dn_dln_mass))
        np.testing.assert_allclose(result.dn_dmass * mass, result.dn_dln_mass)
