"""Validate scale-dependent linear growth and generate growth_scale_dependence.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.growth import GrowthDiagnostics
from dmra.cosmology.ede.model import EDECosmology


def run_growth_validation() -> None:
    """Execute growth validation script across benchmark redshifts and wavenumbers."""
    out_dir = Path("artifacts/tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "growth_scale_dependence.csv"

    # Evaluate central EDE model
    with open("configs/cosmology/ede_central.yaml", encoding="utf-8") as f:
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
    solver = AxiCLASSSolver()
    res = solver.run(req)

    growth = GrowthDiagnostics.from_power_grid(res.power_grid, k_ref_h_mpc=0.05)

    test_ks = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    test_zs = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    rows: list[dict[str, str | float]] = []

    for z in test_zs:
        idx_z = int(np.argmin(np.abs(growth.redshifts - z)))
        for k in test_ks:
            idx_k = int(np.argmin(np.abs(growth.k_h_mpc - k)))
            d_val = float(growth.d_eff_grid[idx_z, idx_k])
            eps_val = float(growth.epsilon_d_grid[idx_z, idx_k])

            rows.append(
                {
                    "redshift": z,
                    "wavenumber_h_mpc": k,
                    "d_eff": d_val,
                    "epsilon_d": eps_val,
                    "abs_epsilon_d_percent": abs(eps_val) * 100.0,
                }
            )

    # Write CSV
    fieldnames = [
        "redshift",
        "wavenumber_h_mpc",
        "d_eff",
        "epsilon_d",
        "abs_epsilon_d_percent",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {out_csv}")


if __name__ == "__main__":
    run_growth_validation()
