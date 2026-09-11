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
