"""Generic transformation from multiplicity to differential halo abundance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.constants import DEFAULT_DELTA_C
from dmra.statistics.multiplicity import MultiplicityModel
from dmra.statistics.peak_height import peak_height


@dataclass(frozen=True, slots=True)
class HaloMassFunctionResult:
    """Differential halo abundance and the intermediate quantities behind it."""

    model_name: str
    mass: FloatArray
    sigma: FloatArray
    nu: FloatArray
    multiplicity: FloatArray
    dln_sigma_inv_dln_mass: FloatArray
    dn_dln_mass: FloatArray
    dn_dmass: FloatArray


class HaloMassFunctionCalculator:
    r"""Apply a multiplicity model to a validated ``sigma(M)`` relation.

    The transformation is

    .. math::

       \frac{dn}{d\ln M} = \frac{\bar\rho_m}{M}
       f(\sigma)\frac{d\ln\sigma^{-1}}{d\ln M}.

    A shape-preserving PCHIP derivative is used in ``(ln M, ln sigma)``. The
    derivative step is kept in one location because subtle differences in this
    Jacobian can otherwise masquerade as differences between HMF models.
    """

    def __init__(self, *, rho_m0: float, delta_c: float = DEFAULT_DELTA_C) -> None:
        if not np.isfinite(rho_m0) or rho_m0 <= 0.0:
            raise ValueError("rho_m0 must be finite and positive")
        if not np.isfinite(delta_c) or delta_c <= 0.0:
            raise ValueError("delta_c must be finite and positive")
        self._rho_m0 = float(rho_m0)
        self._delta_c = float(delta_c)

    @property
    def rho_m0(self) -> float:
        return self._rho_m0

    @property
    def delta_c(self) -> float:
        return self._delta_c

    @staticmethod
    def _validate_mass_sigma(mass: ArrayLike, sigma: ArrayLike) -> tuple[FloatArray, FloatArray]:
        mass_array = np.asarray(mass, dtype=np.float64)
        sigma_array = np.asarray(sigma, dtype=np.float64)

        if mass_array.ndim != 1 or sigma_array.ndim != 1:
            raise ValueError("mass and sigma must be one-dimensional")
        if mass_array.shape != sigma_array.shape:
            raise ValueError("mass and sigma must have identical shapes")
        if mass_array.size < 6:
            raise ValueError("at least six mass samples are required for a stable slope estimate")
        if np.any(~np.isfinite(mass_array)) or np.any(~np.isfinite(sigma_array)):
            raise ValueError("mass and sigma must be finite")
        if np.any(mass_array <= 0.0) or np.any(sigma_array <= 0.0):
            raise ValueError("mass and sigma must be positive")
        if np.any(np.diff(mass_array) <= 0.0):
            raise ValueError("mass must be strictly increasing")
        if np.any(np.diff(sigma_array) >= 0.0):
            raise ValueError("sigma must strictly decrease across the requested mass domain")

        return mass_array, sigma_array

    @classmethod
    def _compute_logarithmic_jacobian(
        cls, mass_array: FloatArray, sigma_array: FloatArray
    ) -> FloatArray:
        log_mass = np.log(mass_array)
        log_sigma = np.log(sigma_array)
        interpolator = PchipInterpolator(log_mass, log_sigma, extrapolate=False)
        slope = -np.asarray(interpolator.derivative()(log_mass), dtype=np.float64)
        if np.any(~np.isfinite(slope)) or np.any(slope <= 0.0):
            raise FloatingPointError("HMF Jacobian is non-positive or non-finite")
        return slope

    @classmethod
    def logarithmic_jacobian(cls, mass: ArrayLike, sigma: ArrayLike) -> FloatArray:
        r"""Return :math:`d\ln\sigma^{-1}/d\ln M`."""
        mass_array, sigma_array = cls._validate_mass_sigma(mass, sigma)
        return cls._compute_logarithmic_jacobian(mass_array, sigma_array)

    def evaluate(
        self,
        mass: ArrayLike,
        sigma: ArrayLike,
        model: MultiplicityModel,
        *,
        jacobian: ArrayLike | None = None,
    ) -> HaloMassFunctionResult:
        r"""Evaluate one HMF model on a common ``mass, sigma`` grid.

        Parameters
        ----------
        mass:
            Mass values in ``M_sun / h``.
        sigma:
            Variance values corresponding to ``mass``.
        model:
            Multiplicity model to evaluate.
        jacobian:
            Optional precomputed analytical Jacobian :math:`d\ln\sigma^{-1}/d\ln M`.
            If not provided, it is computed via shape-preserving PCHIP spline.
        """
        mass_array, sigma_array = self._validate_mass_sigma(mass, sigma)
        if jacobian is not None:
            jac_array = np.asarray(jacobian, dtype=np.float64)
            if jac_array.shape != mass_array.shape:
                raise ValueError("supplied jacobian must match the shape of mass")
            if np.any(~np.isfinite(jac_array)) or np.any(jac_array <= 0.0):
                raise FloatingPointError("supplied HMF Jacobian is non-positive or non-finite")
            effective_jacobian = jac_array
        else:
            effective_jacobian = self._compute_logarithmic_jacobian(mass_array, sigma_array)

        multiplicity = np.asarray(
            model.evaluate(sigma_array, delta_c=self._delta_c), dtype=np.float64
        )
        if multiplicity.shape != mass_array.shape:
            raise ValueError("multiplicity model returned an unexpected shape")
        if np.any(~np.isfinite(multiplicity)) or np.any(multiplicity < 0.0):
            raise FloatingPointError("multiplicity is negative or non-finite")

        dn_dln_mass = self._rho_m0 / mass_array * multiplicity * effective_jacobian
        dn_dmass = dn_dln_mass / mass_array
        nu = peak_height(sigma_array, delta_c=self._delta_c)

        if np.any(~np.isfinite(dn_dln_mass)) or np.any(dn_dln_mass < 0.0):
            raise FloatingPointError("halo mass function is negative or non-finite")

        return HaloMassFunctionResult(
            model_name=model.name,
            mass=mass_array,
            sigma=sigma_array,
            nu=nu,
            multiplicity=multiplicity,
            dln_sigma_inv_dln_mass=effective_jacobian,
            dn_dln_mass=dn_dln_mass,
            dn_dmass=dn_dmass,
        )
