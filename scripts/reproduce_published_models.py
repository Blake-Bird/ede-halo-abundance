"""Evaluate named literature-inspired parameter sets with the configured solver.

This script is a reproducible comparison workflow, not a claim of published
result replication.  Any quantitative agreement claim requires a committed
reference table, solver version, and matching observable definition.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.descriptors import HaloScaleDescriptors
from dmra.cosmology.ede.model import EDECosmology


def run_published_replication() -> None:
    """Execute reproduction of published EDE benchmarks."""
    out_dir = Path("artifacts/tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "published_benchmarks_replication.csv"

    solver = AxiCLASSSolver()

    # 1. Base LCDM reference
    with open("configs/cosmology/lcdm_reference.yaml", encoding="utf-8") as f:
        lcdm_data = yaml.safe_load(f)
    cosmo_lcdm = EDECosmology(
        omega_b=lcdm_data["omega_b"],
        omega_cdm=lcdm_data["omega_cdm"],
        h=lcdm_data["h"],
        A_s=lcdm_data["A_s"],
        n_s=lcdm_data["n_s"],
        tau_reio=lcdm_data["tau_reio"],
        f_ede=0.0,
    )
    res_lcdm = solver.run(LinearTheoryRequest(cosmology=cosmo_lcdm))
    desc_lcdm = HaloScaleDescriptors(res_lcdm.power_grid, cosmo_lcdm)

    # 2. Published models
    models = [
        "configs/cosmology/published/klypin2021_reference.yaml",
        "configs/cosmology/published/poulin2019_bestfit.yaml",
    ]

    masses = np.array([1e10, 1e11, 1e12, 1e13, 1e14, 1e15], dtype=np.float64)
    k_eval = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float64)

    rows: list[dict[str, str | float]] = []

    for m_path in models:
        with open(m_path, encoding="utf-8") as f:
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

        res = solver.run(LinearTheoryRequest(cosmology=cosmo))
        desc = HaloScaleDescriptors(res.power_grid, cosmo)

        # Evaluate at z=0 and z=5
        res_z0 = desc.evaluate(masses, redshift=0.0)
        res_z0_lcdm = desc_lcdm.evaluate(masses, redshift=0.0)

        res_z5 = desc.evaluate(masses, redshift=5.0)
        res_z5_lcdm = desc_lcdm.evaluate(masses, redshift=5.0)

        # Power spectrum ratio at z=0 and z=5
        pk_z0 = res.power_grid.at_redshift(0.0)
        pk_z0_lcdm = res_lcdm.power_grid.at_redshift(0.0)
        r_p_z0 = pk_z0.evaluate(k_eval) / pk_z0_lcdm.evaluate(k_eval)

        pk_z5 = res.power_grid.at_redshift(5.0)
        pk_z5_lcdm = res_lcdm.power_grid.at_redshift(5.0)
        r_p_z5 = pk_z5.evaluate(k_eval) / pk_z5_lcdm.evaluate(k_eval)

        # Rare halo variance ratio at z=5 (M = 10^11 Msun/h)
        idx_rare = 1
        enhancement_high_z = float(res_z5.sigma[idx_rare] / res_z5_lcdm.sigma[idx_rare])

        print(
            f"Model: {data['model_name']} | "
            f"z=5 sigma(1e11) ratio: {enhancement_high_z:.4f} | "
            f"k=1.0 P(k) ratio z=0: {r_p_z0[2]:.4f} | "
            f"z=5: {r_p_z5[2]:.4f}"
        )

        for i, m in enumerate(masses):
            rows.append(
                {
                    "model_name": data["model_name"],
                    "mass_Msun_h": float(m),
                    "sigma_z0_ede": float(res_z0.sigma[i]),
                    "sigma_z0_lcdm": float(res_z0_lcdm.sigma[i]),
                    "sigma_ratio_z0": float(res_z0.sigma[i] / res_z0_lcdm.sigma[i]),
                    "sigma_z5_ede": float(res_z5.sigma[i]),
                    "sigma_z5_lcdm": float(res_z5_lcdm.sigma[i]),
                    "sigma_ratio_z5": float(res_z5.sigma[i] / res_z5_lcdm.sigma[i]),
                    "literature_parameter_set_evaluated": "True",
                }
            )

    # Write CSV
    fieldnames = [
        "model_name",
        "mass_Msun_h",
        "sigma_z0_ede",
        "sigma_z0_lcdm",
        "sigma_ratio_z0",
        "sigma_z5_ede",
        "sigma_z5_lcdm",
        "sigma_ratio_z5",
        "literature_parameter_set_evaluated",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {out_csv}")


if __name__ == "__main__":
    run_published_replication()
