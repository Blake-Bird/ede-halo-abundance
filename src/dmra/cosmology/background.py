"""Minimal flat-LambdaCDM background used by the HMF foundation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from dmra._typing import FloatArray
from dmra.constants import RHO_CRIT_0_H_UNITS


@dataclass(frozen=True, slots=True)
class FlatLambdaCDM:
    """Flat matter + cosmological-constant background.

    The class deliberately contains the background quantities needed to
    validate the HMF machinery. Radiation is supported via ``omega_r0``
    (defaulting to 0 for pure matter+Lambda).

    Unit convention
    ---------------
    Mass: ``M_sun / h``
    Comoving length: ``Mpc / h``
    Density: ``(M_sun / h) / (Mpc / h)^3``
    """

    omega_m0: float
    h: float
    omega_r0: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.omega_m0 <= 1.0:
            raise ValueError("omega_m0 must satisfy 0 < omega_m0 <= 1")
        if not 0.0 <= self.omega_r0 < 1.0:
            raise ValueError("omega_r0 must satisfy 0 <= omega_r0 < 1")
        if self.omega_m0 + self.omega_r0 > 1.0:
            raise ValueError("omega_m0 + omega_r0 must satisfy omega_m0 + omega_r0 <= 1")
        if not 0.0 < self.h < 2.0:
            raise ValueError("h must satisfy 0 < h < 2")

    @property
    def omega_lambda0(self) -> float:
        """Present-day cosmological-constant density parameter."""
        return 1.0 - self.omega_m0 - self.omega_r0

    @property
    def rho_crit0(self) -> float:
        """Critical density today in the project's h-scaled density units."""
        return RHO_CRIT_0_H_UNITS

    @property
    def rho_m0(self) -> float:
        """Mean comoving matter density today in h-scaled density units."""
        return self.omega_m0 * self.rho_crit0

    def e2(self, a: ArrayLike) -> FloatArray:
        """Return ``E(a)^2 = [H(a)/H0]^2`` for flat matter + radiation + Lambda."""
        scale_factor = np.asarray(a, dtype=np.float64)
        if np.any(~np.isfinite(scale_factor)) or np.any(scale_factor <= 0.0):
            raise ValueError("scale factor must be finite and positive")
        inv_a = 1.0 / scale_factor
        inv_a2 = inv_a * inv_a
        inv_a3 = inv_a2 * inv_a
        res = self.omega_m0 * inv_a3 + self.omega_lambda0
        if self.omega_r0 > 0.0:
            res = res + self.omega_r0 * (inv_a3 * inv_a)
        return res

    def e(self, a: ArrayLike) -> FloatArray:
        """Return ``E(a) = H(a)/H0``."""
        return np.sqrt(self.e2(a))

    def dln_e_da(self, a: ArrayLike) -> FloatArray:
        """Analytic derivative ``d ln E / da`` used by the growth ODE."""
        scale_factor = np.asarray(a, dtype=np.float64)
        e2 = self.e2(scale_factor)
        inv_a = 1.0 / scale_factor
        inv_a2 = inv_a * inv_a
        inv_a4 = inv_a2 * inv_a2
        deriv = -1.5 * self.omega_m0 * inv_a4
        if self.omega_r0 > 0.0:
            deriv = deriv - 2.0 * self.omega_r0 * (inv_a4 * inv_a)
        return deriv / e2

    def mass_to_radius(self, mass: ArrayLike) -> FloatArray:
        """Map Lagrangian top-hat mass to comoving radius.

        Parameters
        ----------
        mass:
            Mass in ``M_sun / h``.

        Returns
        -------
        numpy.ndarray
            Radius in ``Mpc / h``.
        """
        mass_array = np.asarray(mass, dtype=np.float64)
        if np.any(~np.isfinite(mass_array)) or np.any(mass_array <= 0.0):
            raise ValueError("mass must be finite and positive")
        return (3.0 * mass_array / (4.0 * np.pi * self.rho_m0)) ** (1.0 / 3.0)

    def radius_to_mass(self, radius: ArrayLike) -> FloatArray:
        """Map comoving top-hat radius to Lagrangian mass."""
        radius_array = np.asarray(radius, dtype=np.float64)
        if np.any(~np.isfinite(radius_array)) or np.any(radius_array <= 0.0):
            raise ValueError("radius must be finite and positive")
        return (4.0 * np.pi / 3.0) * self.rho_m0 * radius_array**3
