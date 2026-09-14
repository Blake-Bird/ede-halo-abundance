"""High-accuracy linear growth for the flat-LambdaCDM reference model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from dmra._typing import FloatArray
from dmra.cosmology.background import FlatLambdaCDM


@dataclass(frozen=True, slots=True)
class GrowthSolution:
    """Normalized growing mode and logarithmic growth rate."""

    a: FloatArray
    D: FloatArray
    f: FloatArray


def solve_linear_growth(
    cosmology: FlatLambdaCDM,
    a: ArrayLike,
    *,
    a_initial: float = 1.0e-3,
    rtol: float = 1.0e-11,
    atol: float = 1.0e-13,
) -> GrowthSolution:
    r"""Solve the linear growing-mode ODE and return ``D`` and ``f``.

    The equation is

    .. math::

       D'' + \left(\frac{3}{a} + \frac{d\ln E}{da}\right)D'
       - \frac{3}{2}\frac{\Omega_{m,0}}{a^5 E^2(a)}D = 0.

    Initial conditions use the matter-era growing mode ``D ~ a``. Both ``D``
    and ``dD/da`` are read directly from the ODE solution; the growth rate is
    then ``f = a D'/D``. This avoids differentiating an interpolated ``D(a)``.
    """
    a_query = np.atleast_1d(np.asarray(a, dtype=np.float64))
    if a_query.ndim != 1 or a_query.size == 0:
        raise ValueError("a must be nonempty and one-dimensional")
    if cosmology.omega_r0 > 0:
        raise ValueError(
            "matter-era initial conditions require omega_r0=0; use Boltzmann growth for radiation"
        )
    if np.any(~np.isfinite(a_query)) or np.any(a_query <= 0.0):
        raise ValueError("a must be finite and positive")
    if np.any(np.diff(a_query) <= 0.0):
        raise ValueError("a must be strictly increasing")
    if not 0.0 < a_initial <= float(a_query[0]):
        raise ValueError("a_initial must be positive and no larger than the first requested a")
    if float(a_query[-1]) > 1.0:
        raise ValueError("the reference growth solver is restricted to a <= 1")

    def rhs(a_value: float, state: np.ndarray) -> np.ndarray:
        D, dD_da = state
        e2 = float(cosmology.e2(a_value))
        friction = 3.0 / a_value + float(cosmology.dln_e_da(a_value))
        source = 1.5 * cosmology.omega_m0 / (a_value**5 * e2)
        return np.array([dD_da, -friction * dD_da + source * D], dtype=np.float64)

    solution = solve_ivp(
        rhs,
        (a_initial, 1.0),
        y0=np.array([a_initial, 1.0], dtype=np.float64),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"growth integration failed: {solution.message}")

    raw = solution.sol(a_query)
    D_raw = np.asarray(raw[0], dtype=np.float64)
    dD_raw = np.asarray(raw[1], dtype=np.float64)

    D_at_one = float(solution.sol(1.0)[0])
    D = D_raw / D_at_one
    dD_da = dD_raw / D_at_one
    f = a_query * dD_da / D

    if np.any(D <= 0.0) or np.any(~np.isfinite(D)) or np.any(~np.isfinite(f)):
        raise FloatingPointError("growth solution contains non-physical values")

    return GrowthSolution(a=a_query, D=D, f=f)
