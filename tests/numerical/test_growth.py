import numpy as np

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.growth import solve_linear_growth


def test_eds_growing_mode_is_exact_reference() -> None:
    cosmology = FlatLambdaCDM(omega_m0=1.0, h=0.7)
    a = np.geomspace(1.0e-3, 1.0, 256)
    solution = solve_linear_growth(cosmology, a)
    np.testing.assert_allclose(solution.D, a, rtol=2.0e-9, atol=2.0e-12)
    np.testing.assert_allclose(solution.f, np.ones_like(a), rtol=2.0e-9, atol=2.0e-9)


def test_lcdm_growth_is_normalized_at_present() -> None:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    a = np.geomspace(1.0e-3, 1.0, 256)
    solution = solve_linear_growth(cosmology, a)
    assert abs(solution.D[-1] - 1.0) < 1.0e-12
    assert np.all(np.diff(solution.D) > 0.0)
