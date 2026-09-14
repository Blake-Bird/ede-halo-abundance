"""Unit gatekeeper establishing invariant transformations at the solver boundary."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from dmra._typing import FloatArray

# Speed of light in km/s (exact SI definition via c = 299,792,458 m/s)
SPEED_OF_LIGHT_KM_S: float = 299792.458


class UnitGatekeeper:
    """Rigorous conversion gatekeeper between external Boltzmann units and DMRA standards.

    Conventions
    -----------
    External (e.g. CLASS / AxiCLASS):
        - Wavenumber: :math:`1 / \\mathrm{Mpc}`
        - Power Spectrum: :math:`\\mathrm{Mpc}^3`
        - Hubble expansion: :math:`1 / \\mathrm{Mpc}` (conformal / comoving) or :math:`\\mathrm{km/s/Mpc}`
    Internal (DMRA Standard):
        - Wavenumber: :math:`h / \\mathrm{Mpc}`
        - Power Spectrum: :math:`(\\mathrm{Mpc} / h)^3`
        - Hubble expansion: :math:`\\mathrm{km/s/Mpc}`
    """

    @staticmethod
    def class_to_project_power(
        k_mpc: ArrayLike,
        pk_mpc3: ArrayLike,
        h: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Convert CLASS power spectrum tables to DMRA standard h-scaled units.

        Parameters
        ----------
        k_mpc:
            Wavenumber in :math:`\\mathrm{Mpc}^{-1}`.
        pk_mpc3:
            Matter power spectrum in :math:`\\mathrm{Mpc}^3`.
        h:
            Dimensionless Hubble parameter :math:`H_0 / (100\\,\\mathrm{km/s/Mpc})`.

        Returns
        -------
        k_h_mpc:
            Wavenumber in :math:`h\\,\\mathrm{Mpc}^{-1}`.
        pk_h_mpc3:
            Power spectrum in :math:`(\\mathrm{Mpc}/h)^3`.
        """
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(f"dimensionless Hubble parameter h must be positive, got {h}")

        k_arr = np.asarray(k_mpc, dtype=np.float64)
        pk_arr = np.asarray(pk_mpc3, dtype=np.float64)

        if (
            np.any(~np.isfinite(k_arr))
            or np.any(~np.isfinite(pk_arr))
            or np.any(k_arr <= 0.0)
            or np.any(pk_arr <= 0.0)
        ):
            raise ValueError("k and P(k) must be finite and strictly positive")

        k_h_mpc = k_arr / h
        pk_h_mpc3 = pk_arr * (h**3)
        return k_h_mpc, pk_h_mpc3

    @staticmethod
    def project_to_class_power(
        k_h_mpc: ArrayLike,
        pk_h_mpc3: ArrayLike,
        h: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Convert DMRA h-scaled power spectrum tables back to CLASS units."""
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(f"dimensionless Hubble parameter h must be positive, got {h}")

        k_arr = np.asarray(k_h_mpc, dtype=np.float64)
        pk_arr = np.asarray(pk_h_mpc3, dtype=np.float64)

        if (
            np.any(~np.isfinite(k_arr))
            or np.any(~np.isfinite(pk_arr))
            or np.any(k_arr <= 0.0)
            or np.any(pk_arr <= 0.0)
        ):
            raise ValueError("k and P(k) must be finite and strictly positive")

        k_mpc = k_arr * h
        pk_mpc3 = pk_arr / (h**3)
        return k_mpc, pk_mpc3

    @staticmethod
    def hubble_inv_mpc_to_kms_mpc(h_inv_mpc: ArrayLike) -> FloatArray:
        """Convert Hubble rate from :math:`\\mathrm{Mpc}^{-1}` to :math:`\\mathrm{km/s/Mpc}`."""
        arr = np.asarray(h_inv_mpc, dtype=np.float64)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError("Hubble rate must be finite and strictly positive")
        return arr * SPEED_OF_LIGHT_KM_S

    @staticmethod
    def record_units_manifest(engine: str) -> dict[str, str]:
        """Return standardized provenance dictionary of raw source units."""
        if engine.lower() in ("class", "axiclass", "class_ede"):
            return {
                "source_engine": engine,
                "raw_wavenumber": "1/Mpc",
                "raw_power": "Mpc^3",
                "raw_hubble": "1/Mpc",
                "target_wavenumber": "h/Mpc",
                "target_power": "(Mpc/h)^3",
                "target_hubble": "km/s/Mpc",
            }
        return {
            "source_engine": engine,
            "raw_wavenumber": "h/Mpc",
            "raw_power": "(Mpc/h)^3",
            "raw_hubble": "km/s/Mpc",
            "target_wavenumber": "h/Mpc",
            "target_power": "(Mpc/h)^3",
            "target_hubble": "km/s/Mpc",
        }
