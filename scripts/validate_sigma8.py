"""Independent validation of sigma_8 via direct Simpson quadrature in ln(k)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import simpson

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.variance import VarianceCalculator
from dmra.cosmology.window import spherical_tophat


def run_sigma8_validation() -> None:
    """Perform independent sigma_8 numerical verification against solver."""
    out_dir = Path("artifacts/tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "sigma8_validation.csv"

    config_files = [
        "lcdm_reference.yaml",
        "ede_weak.yaml",
        "ede_central.yaml",
        "ede_strong.yaml",
    ]

    solver = AxiCLASSSolver()
    rows: list[dict[str, str | float]] = []

    for cfg_name in config_files:
        with open(Path("configs/cosmology") / cfg_name, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cosmo = EDECosmology(
            omega_b=data["omega_b"],
            omega_cdm=data["omega_cdm"],
            h=data["h"],
            A_s=data["A_s"],
            n_s=data["n_s"],
            tau_reio=data["tau_reio"],
            f_ede=data["f_ede"],
            log10_z_c=data["log10_z_c"],
            theta_i=data["theta_i"],
            potential_index=data["potential_index"],
        )

        req = LinearTheoryRequest(cosmology=cosmo)
        res = solver.run(req)

        pk_z0 = res.power_grid.at_redshift(0.0)

        # 1. DMRA VarianceCalculator
        var_calc = VarianceCalculator(pk_z0)
        sigma8_dmra = float(var_calc.sigma_r(8.0).sigma[0])

        # 2. Independent bare Simpson quadrature
        r8 = 8.0  # Mpc/h
        k = pk_z0.k
        pk = pk_z0.pk
        x = k * r8
        w = spherical_tophat(x)
        delta2 = (k**3) * pk / (2.0 * np.pi**2)
        integrand = delta2 * (w**2)
        ln_k = np.log(k)
        sigma8_bare_quad = float(np.sqrt(simpson(integrand, x=ln_k)))

        # 3. Solver reported stdout sigma8
        solver_s8_str = res.manifest.input_parameters.get("solver_sigma8")
        if solver_s8_str is None:
            for line in res.manifest.stdout_summary.split("\n"):
                if "sigma8=" in line and "total matter" in line:
                    parts = line.split("sigma8=")
                    if len(parts) > 1:
                        solver_s8_str = parts[1].split()[0]
                        break

        sigma8_solver = float(solver_s8_str) if solver_s8_str is not None else sigma8_bare_quad

        # Discrepancies
        quad_rel_diff = abs(sigma8_dmra - sigma8_bare_quad) / sigma8_bare_quad
        solver_rel_diff = abs(sigma8_dmra - sigma8_solver) / sigma8_solver

        print(
            f"[{data['model_name']}] DMRA s8: {sigma8_dmra:.6f} | "
            f"Bare Quad: {sigma8_bare_quad:.6f} | "
            f"Solver stdout: {sigma8_solver:.6f} | "
            f"RelDiff Quad: {quad_rel_diff:.2e} | "
            f"RelDiff Solver: {solver_rel_diff:.2e}"
        )

        assert quad_rel_diff < 1e-4, f"Quadrature mismatch in {data['model_name']}: {quad_rel_diff}"

        rows.append(
            {
                "model_name": data["model_name"],
                "sigma8_dmra": sigma8_dmra,
                "sigma8_bare_quad": sigma8_bare_quad,
                "sigma8_solver_stdout": sigma8_solver,
                "relative_difference_quad": quad_rel_diff,
                "relative_difference_solver": solver_rel_diff,
                "sigma8_quadrature_verified": str(quad_rel_diff < 1e-4 and solver_rel_diff < 5e-3),
            }
        )

    # Write CSV
    fieldnames = [
        "model_name",
        "sigma8_dmra",
        "sigma8_bare_quad",
        "sigma8_solver_stdout",
        "relative_difference_quad",
        "relative_difference_solver",
        "sigma8_quadrature_verified",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {out_csv}")


if __name__ == "__main__":
    run_sigma8_validation()
