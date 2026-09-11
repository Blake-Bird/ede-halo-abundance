import numpy as np
from scipy.integrate import simpson

from dmra.statistics.multiplicity import PressSchechter, ShethTormen


def _sigma_from_nu(nu: np.ndarray, delta_c: float = 1.68647) -> np.ndarray:
    return delta_c / nu


def test_press_schechter_mass_fraction_is_unity() -> None:
    nu = np.geomspace(1.0e-8, 20.0, 200_001)
    f = PressSchechter().evaluate(_sigma_from_nu(nu))
    integral = simpson(f, x=np.log(nu))
    assert abs(integral - 1.0) < 2.0e-8


def test_sheth_tormen_mass_fraction_is_unity() -> None:
    nu = np.geomspace(1.0e-18, 20.0, 300_001)
    f = ShethTormen().evaluate(_sigma_from_nu(nu))
    integral = simpson(f, x=np.log(nu))
    assert abs(integral - 1.0) < 2.0e-6


def test_sheth_tormen_contains_press_schechter_limit() -> None:
    nu = np.geomspace(1.0e-3, 10.0, 4096)
    sigma = _sigma_from_nu(nu)
    ps = PressSchechter().evaluate(sigma)
    st_limit = ShethTormen(a=1.0, p=0.0).evaluate(sigma)
    np.testing.assert_allclose(st_limit, ps, rtol=5.0e-15, atol=1.0e-15)


def test_standard_sheth_tormen_normalization_constant() -> None:
    assert abs(ShethTormen().normalization - 0.32218363495952934) < 1.0e-14
