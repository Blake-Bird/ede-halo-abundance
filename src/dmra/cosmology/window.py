"""Smoothing windows used to connect linear fluctuations to halo mass."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from dmra._typing import FloatArray


def spherical_tophat(x: ArrayLike) -> FloatArray:
    r"""Fourier transform of a unit-normalized real-space spherical top hat.

    .. math::

       W(x) = 3\,\frac{\sin x - x\cos x}{x^3}.

    Direct evaluation loses precision near the origin because ``sin(x)`` and
    ``x*cos(x)`` nearly cancel. The local Horner-form series is used for
    ``|x| < 0.08`` to guarantee full 64-bit IEEE-754 precision everywhere.
    """
    values = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("window argument must be finite")

    result = np.empty_like(values)
    small = np.abs(values) < 0.08

    if np.any(small):
        xs = values[small]
        x2 = xs * xs
        # Horner-form Taylor expansion of 3*(sin(x) - x*cos(x))/x^3
        result[small] = 1.0 - x2 * (
            0.1 - x2 * ((1.0 / 280.0) - x2 * ((1.0 / 15120.0) - x2 / 1330560.0))
        )

    if np.any(~small):
        xl = values[~small]
        result[~small] = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / xl**3

    return result


def spherical_tophat_derivative(x: ArrayLike) -> FloatArray:
    r"""First derivative :math:`dW(x)/dx` of the spherical top hat window.

    .. math::

       W'(x) = \frac{3}{x^4}\left(x^2\sin x + 3x\cos x - 3\sin x\right)
             = \frac{3}{x}\left(\frac{\sin x}{x} - W(x)\right).

    Near :math:`x=0`, the analytical Taylor series is used for ``|x| < 0.08``:

    .. math::

       W'(x) = -\frac{x}{5} + \frac{x^3}{70} - \frac{x^5}{2520} + \frac{x^7}{166320} + \mathcal{O}(x^9).
    """
    values = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("window argument must be finite")

    result = np.empty_like(values)
    small = np.abs(values) < 0.08

    if np.any(small):
        xs = values[small]
        x2 = xs * xs
        # Horner-form Taylor expansion of dW/dx
        result[small] = -xs * (0.2 - x2 * ((1.0 / 70.0) - x2 * ((1.0 / 2520.0) - x2 / 166320.0)))

    if np.any(~small):
        xl = values[~small]
        w_l = 3.0 * (np.sin(xl) - xl * np.cos(xl)) / xl**3
        result[~small] = (3.0 / xl) * (np.sin(xl) / xl - w_l)

    return result
