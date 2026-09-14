"""Analytic development utility for adapter tests and exploratory plots.

This module is intentionally *not* a replacement for AxiCLASS or CLASS_EDE.
It uses a phenomenological EDE profile, an approximate transfer function, and
an illustrative scale-dependent growth prescription.  It may only be reached
through an adapter whose ``allow_mock`` flag was explicitly enabled; results
must never be represented as Boltzmann-solver output or scientific validation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from dmra._typing import FloatArray
from dmra.cosmology.boltzmann.manifest import SolverManifest, compute_sha256
from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.boltzmann.result import BackgroundHistory, LinearTheoryResult, MatterPowerGrid
from dmra.cosmology.ede.model import EDECosmology


def compute_ede_energy_density_fraction(
    z: FloatArray,
    f_ede: float,
    z_c: float,
    potential_index: int = 3,
    _theta_i: float = 2.8,
) -> FloatArray:
    """Return a smooth phenomenological EDE fraction for development use only."""
    if f_ede <= 1e-8:
        return np.zeros_like(z, dtype=np.float64)

    w_eff = (potential_index - 1.0) / (potential_index + 1.0)
    # Scale width of transition in ln(1+z)
    width = 2.0 / (3.0 * (1.0 + w_eff))

    ln_1pz = np.log1p(z)
    ln_1pzc = np.log1p(z_c)
    dx = (ln_1pz - ln_1pzc) / width

    # Bell-shaped profile peaking at z_c with maximum f_ede
    # cosh^-2 gives asymptotic decay ~ a^(3(1+w)) post-transition
    omega_phi = f_ede / (np.cosh(dx) ** 2)
    return np.asarray(omega_phi, dtype=np.float64)


def solve_background_history(
    cosmology: EDECosmology,
    _z_min: float = 0.0,
    z_max: float = 10000.0,
    num_z: int = 400,
) -> BackgroundHistory:
    """Compute an approximate background history for development use only."""
    # Redshift grid logarithmically spaced from z=0 to z_max
    z_grid = np.geomspace(1.0, z_max + 1.0, num_z) - 1.0
    z_grid[0] = 0.0  # Ensure exact z=0 node
    a_grid = 1.0 / (1.0 + z_grid)

    om0 = cosmology.Omega_m0
    or0 = cosmology.Omega_r0

    # Base LCDM Friedmann without scalar field
    inv_a = 1.0 / a_grid
    h2_lcdm_ratio = om0 * inv_a**3 + or0 * inv_a**4 + (1.0 - om0 - or0)

    # EDE fraction
    omega_phi = compute_ede_energy_density_fraction(
        z_grid,
        cosmology.f_ede,
        cosmology.z_c,
        cosmology.potential_index,
        cosmology.theta_i,
    )

    # Total expansion rate H(z): 1 - Omega_phi = (H_lcdm / H_tot)^2
    h_ratio2 = h2_lcdm_ratio / np.maximum(1e-4, 1.0 - omega_phi)
    h_kms_mpc = cosmology.H0 * np.sqrt(h_ratio2)

    # Energy density fractions
    om_z = om0 * inv_a**3 / h_ratio2
    or_z = or0 * inv_a**4 / h_ratio2
    ol_z = (1.0 - om0 - or0) / h_ratio2

    # Renormalize to ensure exact cosmic flatness sum == 1.0
    tot = om_z + or_z + ol_z + omega_phi
    om_z /= tot
    or_z /= tot
    ol_z /= tot
    op_z = omega_phi / tot

    return BackgroundHistory(
        z=z_grid,
        a=a_grid,
        H_z=h_kms_mpc,
        omega_m=om_z,
        omega_r=or_z,
        omega_lambda=ol_z,
        omega_phi=op_z,
    )


def eisenstein_hu_transfer(
    k_h_mpc: FloatArray,
    omega_m: float,
    omega_b: float,
    _h: float = 0.67,
) -> FloatArray:
    """Compute an approximate Eisenstein--Hu zero-baryon transfer function."""
    # Shape parameter Gamma
    theta_cmb = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / omega_m) / np.sqrt(1.0 + 10.0 * omega_b**0.75)
    alpha_gamma = (
        1.0
        - 0.328 * np.log(431.0 * omega_m) * (omega_b / omega_m)
        + 0.38 * np.log(22.3 * omega_m) * (omega_b / omega_m) ** 2
    )
    gamma_eff = omega_m * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_h_mpc * s) ** 4))

    q = k_h_mpc * (theta_cmb**2) / gamma_eff
    l0 = np.log(2.0 * np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return np.asarray(l0 / (l0 + c0 * q**2), dtype=np.float64)


def solve_linear_growth(
    bg: BackgroundHistory,
    k_h_mpc: FloatArray,
    target_redshifts: tuple[float, ...],
    f_ede: float,
) -> FloatArray:
    """Solve an approximate subhorizon linear growth equation.

    Returns growth factor matrix of shape (num_redshifts, num_k) normalized so that D(k, z=0) = 1.
    """
    ln_a_nodes = np.log(bg.a)
    h_nodes = bg.H_z
    om_nodes = bg.omega_m

    # Sort in increasing order of ln a for PchipInterpolator
    sort_idx = np.argsort(ln_a_nodes)
    ln_a_sorted = ln_a_nodes[sort_idx]
    h_sorted = h_nodes[sort_idx]
    om_sorted = om_nodes[sort_idx]

    # Interpolators for background in ln a
    h_spline = PchipInterpolator(ln_a_sorted, np.log(h_sorted))
    om_spline = PchipInterpolator(ln_a_sorted, om_sorted)

    def growth_ode(ln_a: float, y: list[float]) -> list[float]:
        # y[0] = delta, y[1] = d delta / dln a
        delta, d_delta = y
        dln_h = float(h_spline.derivative()(ln_a))
        om = float(om_spline(ln_a))
        # d^2 delta / dln a^2 = - (2 + dln H / dln a) d delta / dln a + 1.5 * Omega_m * delta
        d2_delta = -(2.0 + dln_h) * d_delta + 1.5 * om * delta
        return [d_delta, d2_delta]

    # Integrate from early matter domination ln a = -6 (z ~ 400) to ln a = 0 (z = 0)
    ln_a_span = (-6.0, 0.0)
    y0 = [np.exp(-6.0), np.exp(-6.0)]  # Growing mode delta ~ a

    eval_ln_a = np.array(
        [float(np.log(1.0 / (1.0 + z))) for z in target_redshifts], dtype=np.float64
    )
    sort_order = np.argsort(eval_ln_a)
    eval_ln_a_sorted = eval_ln_a[sort_order]
    eval_ln_a_sorted = np.clip(eval_ln_a_sorted, -5.999, 0.0)

    sol = solve_ivp(
        growth_ode,
        ln_a_span,
        y0,
        t_eval=eval_ln_a_sorted,
        rtol=1e-6,
        atol=1e-8,
        method="RK45",
    )

    inv_sort = np.argsort(sort_order)
    delta_eval = sol.y[0][inv_sort]
    # Normalize so D(z=0) = 1.0
    idx_z0 = target_redshifts.index(0.0) if 0.0 in target_redshifts else -1
    norm = delta_eval[idx_z0] if idx_z0 >= 0 else 1.0
    d_base = delta_eval / norm

    # Incorporate scale-dependent EDE suppression plateau (Derivation 15):
    # D_eff(k, z) = D_base(z) * [1 - 0.5 * f_ede * ln(1 + k/0.1) / ln(1 + 10/0.1)]
    # for high-k modes that entered horizon during active EDE
    num_z = len(target_redshifts)
    num_k = k_h_mpc.size
    growth_matrix = np.zeros((num_z, num_k), dtype=np.float64)

    kc = 0.2  # characteristic transition scale h/Mpc
    suppression_k = 1.0 - 0.6 * f_ede * (np.log1p(k_h_mpc / kc) / np.log1p(100.0 / kc))
    suppression_k = np.clip(suppression_k, 0.5, 1.0)

    for i, z in enumerate(target_redshifts):
        # Suppression is active primarily at low-z after the EDE phase has passed
        # At high z >> z_c, perturbations are identical to primordial
        w_z = 1.0 / (1.0 + (z / 3000.0) ** 2)
        growth_matrix[i, :] = d_base[i] * (1.0 - (1.0 - suppression_k) * w_z)

    # Re-normalize strictly so D_eff(k, z=0) = 1.0
    if idx_z0 >= 0:
        growth_matrix /= growth_matrix[idx_z0 : idx_z0 + 1, :]

    return growth_matrix


def generate_linear_theory_result(
    request: LinearTheoryRequest,
    solver_name: str = "embedded_ede_physics",
    solver_version: str = "v1.0.0",
) -> LinearTheoryResult:
    """Generate explicitly non-production linear-theory data for development only."""
    cosmo = request.cosmology
    bg = solve_background_history(cosmo)

    # Build k-grid
    k_grid = np.geomspace(request.k_min_h_mpc, request.k_max_h_mpc, request.num_k_points)

    # Primordial spectrum P_prim(k) = 2pi^2 / k^3 * A_s * (k / k_pivot)^(n_s - 1)
    k_pivot_h = 0.05 / cosmo.h
    p_prim = (2.0 * np.pi**2 / k_grid**3) * cosmo.A_s * (k_grid / k_pivot_h) ** (cosmo.n_s - 1.0)

    # Transfer function T(k)
    t_k = eisenstein_hu_transfer(k_grid, cosmo.omega_m, cosmo.omega_b, cosmo.h)

    # Linear power spectrum at z=0: P(k, 0) = P_prim(k) * T^2(k) * norm
    # Normalize to produce standard sigma_8 ~ 0.81
    pk_z0_raw = p_prim * t_k**2
    # Standard normalization factor to match Planck amplitude (sigma_8 ~ 0.811)
    norm_factor = 2.9447e8 * (cosmo.h**4)
    pk_z0 = pk_z0_raw * norm_factor

    # Compute growth across request redshifts
    growth_matrix = solve_linear_growth(bg, k_grid, request.redshifts, cosmo.f_ede)

    # 2D power grid P(k, z) = P(k, 0) * D^2(k, z)
    pk_grid = np.zeros((len(request.redshifts), k_grid.size), dtype=np.float64)
    for i in range(len(request.redshifts)):
        pk_grid[i, :] = pk_z0 * (growth_matrix[i, :] ** 2)

    power_grid = MatterPowerGrid(
        k_h_mpc=k_grid,
        redshifts=np.array(request.redshifts, dtype=np.float64),
        pk_grid=pk_grid,
    )

    manifest = SolverManifest(
        solver_name=solver_name,
        solver_version=solver_version,
        config_hash=compute_sha256({"cosmo": str(cosmo), "redshifts": list(request.redshifts)}),
        input_parameters={"f_ede": cosmo.f_ede, "h": cosmo.h, "omega_m": cosmo.omega_m},
        raw_units={"k": "h/Mpc", "P": "(Mpc/h)^3", "H": "km/s/Mpc"},
        execution_time_seconds=0.05,
        stdout_summary="Development-only analytic generator; not Boltzmann-solver output.",
    )

    return LinearTheoryResult(
        cosmology=cosmo,
        background=bg,
        power_grid=power_grid,
        manifest=manifest,
    )
