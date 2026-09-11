import numpy as np

from dmra.cosmology.window import spherical_tophat


def test_tophat_origin_limit() -> None:
    x = np.array([0.0, 1.0e-12, 1.0e-8, 1.0e-5])
    expected = 1.0 - x**2 / 10.0 + x**4 / 280.0 - x**6 / 15120.0
    np.testing.assert_allclose(spherical_tophat(x), expected, rtol=0.0, atol=2.0e-16)


def test_tophat_is_even() -> None:
    x = np.geomspace(1.0e-5, 30.0, 1000)
    np.testing.assert_allclose(
        spherical_tophat(x), spherical_tophat(-x), rtol=2.0e-14, atol=2.0e-14
    )
