import numpy as np

from dmra.cosmology.window import spherical_tophat, spherical_tophat_derivative


def test_tophat_derivative_origin_limit() -> None:
    x = np.array([0.0, 1.0e-12, 1.0e-8, 1.0e-5, 0.01])
    x2 = x * x
    expected = -x * (0.2 - x2 * ((1.0 / 70.0) - x2 * ((1.0 / 2520.0) - x2 / 166320.0)))
    np.testing.assert_allclose(spherical_tophat_derivative(x), expected, rtol=1.0e-14, atol=1.0e-18)


def test_tophat_derivative_is_odd() -> None:
    x = np.geomspace(1.0e-5, 30.0, 1000)
    np.testing.assert_allclose(
        spherical_tophat_derivative(x),
        -spherical_tophat_derivative(-x),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_tophat_derivative_matches_finite_differences() -> None:
    x = np.geomspace(0.01, 20.0, 200)
    # Optimal step size for 4th-order central difference in float64 is h ~ eps^(1/5) ~ 1e-3
    h = 1.0e-3
    numerical_deriv = (
        -spherical_tophat(x + 2 * h)
        + 8.0 * spherical_tophat(x + h)
        - 8.0 * spherical_tophat(x - h)
        + spherical_tophat(x - 2 * h)
    ) / (12.0 * h)
    analytical_deriv = spherical_tophat_derivative(x)
    np.testing.assert_allclose(analytical_deriv, numerical_deriv, rtol=5.0e-8, atol=1.0e-9)
