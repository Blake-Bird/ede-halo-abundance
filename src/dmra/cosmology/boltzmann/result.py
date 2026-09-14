"""Output domain models representing linear theory results, power spectrum grids, and background histories."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.cosmology.boltzmann.manifest import SolverManifest
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.power import TabulatedPowerSpectrum


@dataclass(slots=True, frozen=True)
class BackgroundHistory:
    """Evolution of background cosmological expansion and energy density fractions.

    Parameters
    ----------
    z:
        Array of redshifts, strictly monotonic.
    a:
        Array of scale factors :math:`a = 1/(1+z)`.
    H_z:
        Hubble expansion rate :math:`H(z)` in :math:`\\mathrm{km/s/Mpc}`.
    omega_m:
        Fractional matter density :math:`\\Omega_m(z) \\equiv \\rho_m(z)/\\rho_{\\rm crit}(z)`.
    omega_r:
        Fractional radiation density :math:`\\Omega_r(z) \\equiv \\rho_r(z)/\\rho_{\\rm crit}(z)`.
    omega_lambda:
        Fractional cosmological constant density :math:`\\Omega_\\Lambda(z)`.
    omega_phi:
        Fractional Early Dark Energy scalar field density :math:`\\Omega_\\phi(z)`.
    """

    z: FloatArray
    a: FloatArray
    H_z: FloatArray
    omega_m: FloatArray
    omega_r: FloatArray
    omega_lambda: FloatArray
    omega_phi: FloatArray
    _hubble_interp: PchipInterpolator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        z_arr = np.asarray(self.z, dtype=np.float64)
        a_arr = np.asarray(self.a, dtype=np.float64)
        h_arr = np.asarray(self.H_z, dtype=np.float64)
        om_arr = np.asarray(self.omega_m, dtype=np.float64)
        or_arr = np.asarray(self.omega_r, dtype=np.float64)
        ol_arr = np.asarray(self.omega_lambda, dtype=np.float64)
        op_arr = np.asarray(self.omega_phi, dtype=np.float64)

        for name, arr in [
            ("z", z_arr),
            ("a", a_arr),
            ("H_z", h_arr),
            ("omega_m", om_arr),
            ("omega_r", or_arr),
            ("omega_lambda", ol_arr),
            ("omega_phi", op_arr),
        ]:
            if arr.ndim != 1:
                raise ValueError(f"array {name} must be 1-dimensional")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"array {name} contains non-finite values")

        n_pts = z_arr.size
        if n_pts < 2:
            raise ValueError("BackgroundHistory requires at least 2 points")
        if not (
            a_arr.size
            == h_arr.size
            == om_arr.size
            == or_arr.size
            == ol_arr.size
            == op_arr.size
            == n_pts
        ):
            raise ValueError("all background history arrays must have identical length")

        if np.any(h_arr <= 0.0):
            raise ValueError("Hubble expansion rate H(z) must be strictly positive")
        if np.any(z_arr < 0) or not np.allclose(a_arr, 1 / (1 + z_arr), rtol=1e-10, atol=0):
            raise ValueError("background requires z >= 0 and a = 1/(1+z)")

        # Ensure arrays are sorted by increasing redshift for PCHIP
        if np.all(np.diff(z_arr) > 0.0):
            # Already ascending
            sort_idx = None
        elif np.all(np.diff(z_arr) < 0.0):
            # Descending, reverse
            sort_idx = np.argsort(z_arr)
            z_arr = z_arr[sort_idx]
            a_arr = a_arr[sort_idx]
            h_arr = h_arr[sort_idx]
            om_arr = om_arr[sort_idx]
            or_arr = or_arr[sort_idx]
            ol_arr = ol_arr[sort_idx]
            op_arr = op_arr[sort_idx]
        else:
            raise ValueError("redshift array z must be strictly monotonic")

        object.__setattr__(self, "z", z_arr)
        object.__setattr__(self, "a", a_arr)
        object.__setattr__(self, "H_z", h_arr)
        object.__setattr__(self, "omega_m", om_arr)
        object.__setattr__(self, "omega_r", or_arr)
        object.__setattr__(self, "omega_lambda", ol_arr)
        object.__setattr__(self, "omega_phi", op_arr)

        # Build monotonic PCHIP interpolator in ln(1+z) -> ln H
        ln_1pz = np.log1p(z_arr)
        ln_h = np.log(h_arr)
        object.__setattr__(
            self, "_hubble_interp", PchipInterpolator(ln_1pz, ln_h, extrapolate=False)
        )

    def validate_flatness(self, tolerance: float = 1e-4) -> bool:
        """Assert cosmic flatness :math:`|\\sum_i \\Omega_i(z) - 1.0| < \\epsilon`."""
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        total_density = self.omega_m + self.omega_r + self.omega_lambda + self.omega_phi
        residual = np.abs(total_density - 1.0)
        max_deviation = float(np.max(residual))
        if max_deviation >= tolerance:
            raise ValueError(
                f"cosmic flatness violated: maximum deviation is {max_deviation:.3e} "
                f"(tolerance: {tolerance:.3e})"
            )
        return True

    def hubble(self, z_query: ArrayLike) -> FloatArray:
        """Interpolate Hubble rate :math:`H(z)` in :math:`\\mathrm{km/s/Mpc}`."""
        q = np.asarray(z_query, dtype=np.float64)
        if np.any(~np.isfinite(q)) or np.any(q < self.z[0]) or np.any(q > self.z[-1]):
            raise ValueError("requested redshift lies outside background table support")
        ln_q = np.log1p(q)
        return np.asarray(np.exp(self._hubble_interp(ln_q)), dtype=np.float64)

    def omega_phi_peak(self) -> tuple[float, float]:
        """Return realized :math:`(f_{\\rm EDE}^{\\rm realized}, z_c^{\\rm realized})`."""
        idx_max = int(np.argmax(self.omega_phi))
        return float(self.omega_phi[idx_max]), float(self.z[idx_max])

    def fwhm_ln_1pz(self) -> float:
        """Compute full-width at half-maximum (FWHM) of the EDE transition in :math:`\\ln(1+z)`."""
        f_max, _ = self.omega_phi_peak()
        if f_max < 1e-5:
            return 0.0

        half_max = 0.5 * f_max
        above_half = self.omega_phi >= half_max
        if not np.any(above_half):
            return 0.0

        ln_1pz = np.log1p(self.z)
        ln_above = ln_1pz[above_half]
        return float(np.max(ln_above) - np.min(ln_above))


@dataclass(slots=True, frozen=True)
class MatterPowerGrid:
    """Two-dimensional tabulated linear matter power spectrum grid :math:`P(k, z)`.

    Parameters
    ----------
    k_h_mpc:
        One-dimensional array of wavenumbers in :math:`h/\\mathrm{Mpc}`.
    redshifts:
        One-dimensional array of evaluation redshifts :math:`z`.
    pk_grid:
        Two-dimensional array of linear matter power in :math:`(\\mathrm{Mpc}/h)^3`,
        with shape ``(len(redshifts), len(k_h_mpc))``.
    """

    k_h_mpc: FloatArray
    redshifts: FloatArray
    pk_grid: FloatArray

    def __post_init__(self) -> None:
        k_arr = np.asarray(self.k_h_mpc, dtype=np.float64)
        z_arr = np.asarray(self.redshifts, dtype=np.float64)
        pk_arr = np.asarray(self.pk_grid, dtype=np.float64)

        if k_arr.ndim != 1 or z_arr.ndim != 1:
            raise ValueError("k_h_mpc and redshifts must be 1-dimensional")
        if z_arr.size == 0:
            raise ValueError("redshifts must not be empty")
        if pk_arr.ndim != 2:
            raise ValueError("pk_grid must be 2-dimensional with shape (num_z, num_k)")
        if pk_arr.shape != (z_arr.size, k_arr.size):
            raise ValueError(
                f"pk_grid shape {pk_arr.shape} does not match expected ({z_arr.size}, {k_arr.size})"
            )
        if k_arr.size < 8:
            raise ValueError("wavenumber grid requires at least 8 samples")
        if (
            np.any(~np.isfinite(k_arr))
            or np.any(~np.isfinite(z_arr))
            or np.any(~np.isfinite(pk_arr))
        ):
            raise ValueError("power grid values must be finite")
        if np.any(k_arr <= 0.0) or np.any(pk_arr <= 0.0):
            raise ValueError("k and P(k, z) must be strictly positive")
        if np.any(np.diff(k_arr) <= 0.0):
            raise ValueError("k must be strictly increasing")
        if np.any(z_arr < 0.0):
            raise ValueError("redshifts must be non-negative")
        if np.unique(z_arr).size != z_arr.size:
            raise ValueError("redshifts must be unique")

        object.__setattr__(self, "k_h_mpc", k_arr)
        object.__setattr__(self, "redshifts", z_arr)
        object.__setattr__(self, "pk_grid", pk_arr)

    @property
    def num_redshifts(self) -> int:
        return self.redshifts.size

    @property
    def num_wavenumbers(self) -> int:
        return self.k_h_mpc.size

    def at_redshift(self, z: float, *, extrapolate: bool = False) -> TabulatedPowerSpectrum:
        """Extract or interpolate a TabulatedPowerSpectrum slice at target redshift.

        If target redshift aligns with a tabulated grid node (within :math:`10^{-6}`),
        returns that slice directly. Otherwise, interpolates logarithmically in
        :math:`\\ln(1+z)` between the two bounding redshift planes.

        Parameters
        ----------
        z:
            Target redshift :math:`z \\ge 0`.
        extrapolate:
            Whether to permit extrapolation beyond tabulated redshift bounds.
            Default is False to prevent unphysical tail power.

        Returns
        -------
        TabulatedPowerSpectrum:
            Validated power spectrum instance fully compatible with DMRA variance calculators.
        """
        if not np.isfinite(z) or z < 0.0:
            raise ValueError(f"redshift must be finite and non-negative, got {z}")

        z_nodes = self.redshifts
        idx_exact = np.where(np.isclose(z_nodes, z, atol=1e-6, rtol=0))[0]
        if idx_exact.size > 0:
            idx = int(idx_exact[0])
            return TabulatedPowerSpectrum(self.k_h_mpc.copy(), self.pk_grid[idx].copy())

        # Check domain boundaries
        z_min = float(np.min(z_nodes))
        z_max = float(np.max(z_nodes))
        if not extrapolate and (z < z_min or z > z_max):
            raise ValueError(
                f"requested redshift z={z:.3f} lies outside tabulated bounds "
                f"[{z_min:.3f}, {z_max:.3f}]"
            )

        # Logarithmic interpolation in (1+z)
        # Find enclosing or boundary extrapolation nodes
        sorted_indices = np.argsort(z_nodes)
        z_sorted = z_nodes[sorted_indices]
        pk_sorted = self.pk_grid[sorted_indices]

        n_nodes = z_sorted.size
        if n_nodes < 2:
            raise ValueError("interpolation requires at least two redshifts")
        if z < z_sorted[0]:
            idx_lower = 0
            idx_upper = 1
        elif z > z_sorted[-1]:
            idx_lower = n_nodes - 2
            idx_upper = n_nodes - 1
        else:
            idx_upper = int(np.searchsorted(z_sorted, z))
            idx_lower = idx_upper - 1

        z0 = z_sorted[idx_lower]
        z1 = z_sorted[idx_upper]
        pk0 = pk_sorted[idx_lower]
        pk1 = pk_sorted[idx_upper]

        ln_1pz0 = np.log1p(z0)
        ln_1pz1 = np.log1p(z1)
        ln_1pz = np.log1p(z)

        weight = (ln_1pz - ln_1pz0) / (ln_1pz1 - ln_1pz0)

        # Interpolate ln P(k)
        log_pk_interp = (1.0 - weight) * np.log(pk0) + weight * np.log(pk1)
        pk_interp = np.exp(log_pk_interp)

        return TabulatedPowerSpectrum(self.k_h_mpc.copy(), pk_interp)

    def growth_factor(self, z: float, k_ref: float = 0.05) -> float:
        """Compute effective linear growth factor :math:`D_{\\rm eff}(k_{\\rm ref}, z) = \\sqrt{P(k_{\\rm ref}, z)/P(k_{\\rm ref}, 0)}`."""
        pk_z = self.at_redshift(z).evaluate(k_ref)
        pk_0 = self.at_redshift(0.0).evaluate(k_ref)
        return float(np.sqrt(pk_z / pk_0))


@dataclass(slots=True, frozen=True)
class LinearTheoryResult:
    """Complete bundle containing cosmology, background, perturbation grids, and provenance manifest."""

    cosmology: EDECosmology
    background: BackgroundHistory
    power_grid: MatterPowerGrid
    manifest: SolverManifest
