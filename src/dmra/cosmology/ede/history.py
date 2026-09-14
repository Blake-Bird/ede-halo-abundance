"""Realized Early Dark Energy history extraction and shooting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from dmra._typing import FloatArray
from dmra.cosmology.boltzmann.result import BackgroundHistory


@dataclass(slots=True, frozen=True)
class EDEHistory:
    """Detailed realization diagnostics extracted from an EDE BackgroundHistory.

    Parameters
    ----------
    f_ede_realized:
        Peak fractional scalar field density :math:`\\max_z \\Omega_\\phi(z)`.
    z_c_realized:
        Redshift at which :math:`\\Omega_\\phi(z)` attains its global maximum.
    fwhm_ln_1pz:
        Full-width at half-maximum of the EDE transition in :math:`\\ln(1+z)`.
    f_ede_shooting_error:
        Relative discrepancy :math:`|f_{\\rm EDE}^{\\rm realized} - f_{\\rm EDE}^{\\rm requested}| / f_{\\rm EDE}^{\\rm requested}`.
    z_c_shooting_error:
        Relative discrepancy :math:`|z_c^{\\rm realized} - z_c^{\\rm requested}| / z_c^{\\rm requested}`.
    """

    f_ede_realized: float
    z_c_realized: float
    fwhm_ln_1pz: float
    f_ede_shooting_error: float
    z_c_shooting_error: float

    @classmethod
    def from_background(
        cls,
        bg: BackgroundHistory,
        requested_f_ede: float,
        requested_z_c: float,
    ) -> EDEHistory:
        """Extract EDEHistory from a solved BackgroundHistory."""
        f_realized, z_realized = bg.omega_phi_peak()
        fwhm = bg.fwhm_ln_1pz()

        if requested_f_ede > 1e-6:
            err_f = abs(f_realized - requested_f_ede) / requested_f_ede
            err_zc = abs(z_realized - requested_z_c) / requested_z_c
        else:
            err_f = 0.0
            err_zc = 0.0

        return cls(
            f_ede_realized=f_realized,
            z_c_realized=z_realized,
            fwhm_ln_1pz=fwhm,
            f_ede_shooting_error=err_f,
            z_c_shooting_error=err_zc,
        )

    @staticmethod
    def compute_delta_h_over_h(
        ede_bg: BackgroundHistory,
        lcdm_bg: BackgroundHistory,
        z_eval: FloatArray,
    ) -> FloatArray:
        """Compute relative Hubble expansion perturbation :math:`\\Delta H(z)/H_{\\Lambda{\\rm CDM}}(z)`."""
        h_ede = ede_bg.hubble(z_eval)
        h_lcdm = lcdm_bg.hubble(z_eval)
        return (h_ede - h_lcdm) / h_lcdm
