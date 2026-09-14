"""Physical domain models for Early Dark Energy (EDE) cosmologies."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dmra.cosmology.background import FlatLambdaCDM


@dataclass(slots=True, frozen=True)
class EDECosmology:
    """Canonical physical parameter specification for an EDE cosmology.

    Parameters
    ----------
    omega_b:
        Physical baryon density parameter, :math:`\\omega_b \\equiv \\Omega_b h^2`.
    omega_cdm:
        Physical cold dark matter density parameter, :math:`\\omega_{\\rm cdm} \\equiv \\Omega_{\\rm cdm} h^2`.
    h:
        Dimensionless Hubble parameter, :math:`h \\equiv H_0 / (100\\,\\mathrm{km/s/Mpc})`.
    A_s:
        Primordial scalar perturbation amplitude at pivot scale :math:`k_0 = 0.05\\,\\mathrm{Mpc}^{-1}`.
    n_s:
        Primordial scalar spectral index.
    tau_reio:
        Thomson optical depth to reionization.
    f_ede:
        Peak fractional energy density of the scalar field,
        :math:`f_{\\rm EDE} \\equiv \\max_z [\\rho_\\phi(z) / \\rho_{\\rm tot}(z)]`.
    log10_z_c:
        Decimal logarithm of the transition redshift :math:`z_c` where :math:`\\rho_\\phi/\\rho_{\\rm tot}` peaks.
    theta_i:
        Initial field displacement angle on the periodic potential :math:`\\theta_i \\equiv \\phi_i / f \\in [0, \\pi]`.
    potential_index:
        Power index :math:`n` in the axion-like potential :math:`V(\\phi) = m^2 f^2 [1 - \\cos(\\phi/f)]^n`.
        Default is 3, characteristic of canonical string axion models.
    omega_r0:
        Optional radiation density parameter :math:`\\omega_r \\equiv \\Omega_r h^2`. Default is 0.0.
    """

    omega_b: float
    omega_cdm: float
    h: float
    A_s: float = 2.1e-9
    n_s: float = 0.965
    tau_reio: float = 0.054
    f_ede: float = 0.0
    log10_z_c: float = 3.5
    theta_i: float = 2.8
    potential_index: int = 3
    omega_r0: float = 0.0

    def __post_init__(self) -> None:
        """Enforce strict theoretical and physical validity boundaries."""
        # Check finiteness across all parameters
        float_params = [
            ("omega_b", self.omega_b),
            ("omega_cdm", self.omega_cdm),
            ("h", self.h),
            ("A_s", self.A_s),
            ("n_s", self.n_s),
            ("tau_reio", self.tau_reio),
            ("f_ede", self.f_ede),
            ("log10_z_c", self.log10_z_c),
            ("theta_i", self.theta_i),
            ("omega_r0", self.omega_r0),
        ]
        for name, val in float_params:
            if not math.isfinite(val):
                raise ValueError(f"parameter {name} must be finite, got {val}")

        # Physical density bounds
        if not (0.005 <= self.omega_b <= 0.050):
            raise ValueError(f"omega_b must lie in [0.005, 0.050], got {self.omega_b}")
        if not (0.01 <= self.omega_cdm <= 0.60):
            raise ValueError(f"omega_cdm must lie in [0.01, 0.60], got {self.omega_cdm}")
        if not (0.40 <= self.h <= 1.20):
            raise ValueError(f"h must lie in [0.40, 1.20], got {self.h}")
        if not (0.0 <= self.omega_r0 <= 0.01):
            raise ValueError(f"omega_r0 must lie in [0.0, 0.01], got {self.omega_r0}")
        if self.Omega_m0 + self.Omega_r0 > 1.0:
            raise ValueError("omega_b, omega_cdm, h, and omega_r0 imply Omega_m0 + Omega_r0 > 1")

        # Primordial & reionization parameters
        if not (1.0e-10 <= self.A_s <= 1.0e-8):
            raise ValueError(f"A_s must lie in [1e-10, 1e-8], got {self.A_s}")
        if not (0.70 <= self.n_s <= 1.30):
            raise ValueError(f"n_s must lie in [0.70, 1.30], got {self.n_s}")
        if not (0.01 <= self.tau_reio <= 0.25):
            raise ValueError(f"tau_reio must lie in [0.01, 0.25], got {self.tau_reio}")

        # EDE specific parameters
        if not (0.0 <= self.f_ede <= 0.50):
            raise ValueError(f"f_ede must lie in [0.0, 0.50], got {self.f_ede}")
        if not (2.0 <= self.log10_z_c <= 5.0):
            raise ValueError(f"log10_z_c must lie in [2.0, 5.0], got {self.log10_z_c}")
        if not (0.0 <= self.theta_i <= math.pi):
            raise ValueError(f"theta_i must lie in [0.0, pi], got {self.theta_i}")
        if not isinstance(self.potential_index, int) or self.potential_index < 1:
            raise ValueError(
                f"potential_index must be a positive integer >= 1, got {self.potential_index}"
            )

    @property
    def omega_m(self) -> float:
        """Total physical matter density parameter: :math:`\\omega_m = \\omega_b + \\omega_{\\rm cdm}`."""
        return self.omega_b + self.omega_cdm

    @property
    def Omega_b0(self) -> float:
        """Present-day fractional baryon density: :math:`\\Omega_{b0} = \\omega_b / h^2`."""
        return self.omega_b / (self.h * self.h)

    @property
    def Omega_cdm0(self) -> float:
        """Present-day fractional cold dark matter density: :math:`\\Omega_{{\\rm cdm}0} = \\omega_{\\rm cdm} / h^2`."""
        return self.omega_cdm / (self.h * self.h)

    @property
    def Omega_m0(self) -> float:
        """Present-day fractional matter density: :math:`\\Omega_{m0} = \\omega_m / h^2`."""
        return self.omega_m / (self.h * self.h)

    @property
    def Omega_r0(self) -> float:
        """Present-day fractional radiation density: :math:`\\Omega_{r0} = \\omega_r / h^2`."""
        return self.omega_r0 / (self.h * self.h)

    @property
    def Omega_lambda0(self) -> float:
        """Present-day fractional cosmological constant density in flat space."""
        return 1.0 - self.Omega_m0 - self.Omega_r0

    @property
    def ln10_As(self) -> float:
        """Standard log-amplitude :math:`\\ln(10^{10} A_s)` used in Planck / CLASS."""
        return math.log(1.0e10 * self.A_s)

    @property
    def H0(self) -> float:
        """Hubble constant in :math:`\\mathrm{km/s/Mpc}`."""
        return 100.0 * self.h

    @property
    def z_c(self) -> float:
        """Critical EDE transition redshift: :math:`z_c = 10^{\\log_{10} z_c}`."""
        return float(10.0**self.log10_z_c)

    @property
    def is_ede(self) -> bool:
        """Return True if EDE is dynamically active (:math:`f_{\\rm EDE} > 10^{-6}`)."""
        return self.f_ede > 1.0e-6

    def to_flat_lambdacdm(self) -> FlatLambdaCDM:
        """Extract equivalent background FlatLambdaCDM instance for baseline checks."""
        return FlatLambdaCDM(
            omega_m0=self.Omega_m0,
            h=self.h,
            omega_r0=self.Omega_r0,
        )
