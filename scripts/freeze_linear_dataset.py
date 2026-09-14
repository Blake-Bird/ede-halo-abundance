"""Freeze linear theory and halo-scale dataset into Parquet and compute SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.descriptors import HaloScaleDescriptors
from dmra.cosmology.ede.growth import GrowthDiagnostics
from dmra.cosmology.ede.model import EDECosmology


def freeze_dataset() -> None:
    """Freeze processed cosmological data artifacts and compute cryptographic checksums."""
    out_dir = Path("artifacts/processed/linear_theory").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = AxiCLASSSolver()

    # Load central EDE model
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
    res = solver.run(req)

    # 1. Background History Table
    bg = res.background
    df_bg = pd.DataFrame(
        {
            "redshift": bg.z,
            "scale_factor": bg.a,
            "hubble_kms_mpc": bg.H_z,
            "omega_m": bg.omega_m,
            "omega_r": bg.omega_r,
            "omega_lambda": bg.omega_lambda,
            "omega_phi": bg.omega_phi,
        }
    )
    bg_file = out_dir / "background_history.parquet"
    df_bg.to_parquet(bg_file, index=False)

    # 2. Matter Power Grid Table
    grid = res.power_grid
    rows_grid: list[dict[str, float]] = []
    for i, z in enumerate(grid.redshifts):
        for j, k in enumerate(grid.k_h_mpc):
            rows_grid.append(
                {
                    "redshift": float(z),
                    "wavenumber_h_mpc": float(k),
                    "pk_mpc_h3": float(grid.pk_grid[i, j]),
                }
            )
    df_grid = pd.DataFrame(rows_grid)
    grid_file = out_dir / "matter_power_grid.parquet"
    df_grid.to_parquet(grid_file, index=False)

    # 3. Growth Diagnostics Table
    growth = GrowthDiagnostics.from_power_grid(grid)
    rows_growth: list[dict[str, float]] = []
    for i, z in enumerate(growth.redshifts):
        for j, k in enumerate(growth.k_h_mpc):
            rows_growth.append(
                {
                    "redshift": float(z),
                    "wavenumber_h_mpc": float(k),
                    "d_eff": float(growth.d_eff_grid[i, j]),
                    "epsilon_d": float(growth.epsilon_d_grid[i, j]),
                }
            )
    df_growth = pd.DataFrame(rows_growth)
    growth_file = out_dir / "growth_diagnostics.parquet"
    df_growth.to_parquet(growth_file, index=False)

    # 4. Halo Scale Descriptors Table
    desc = HaloScaleDescriptors(grid, cosmo)
    masses = np.geomspace(1e8, 1e16, 50)
    rows_desc: list[dict[str, float]] = []
    for z in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        res_desc = desc.evaluate(masses, redshift=z)
        rows_desc.extend(
            [
                {
                    "redshift": float(z),
                    "mass_Msun_h": float(res_desc.mass[idx_m]),
                    "radius_mpc_h": float(res_desc.radius[idx_m]),
                    "sigma": float(res_desc.sigma[idx_m]),
                    "dln_sigma_inv_dln_m": float(res_desc.dln_sigma_inv_dln_m[idx_m]),
                    "nu": float(res_desc.nu[idx_m]),
                    "n_eff": float(res_desc.n_eff[idx_m]),
                }
                for idx_m in range(masses.size)
            ]
        )
    df_desc = pd.DataFrame(rows_desc)
    desc_file = out_dir / "halo_scale_descriptors.parquet"
    df_desc.to_parquet(desc_file, index=False)

    # 5. Provenance Metadata
    meta = {
        "dataset_name": "LINEAR_THEORY_V1",
        "cosmology": "ede_central",
        "parameters": {
            "omega_b": cosmo.omega_b,
            "omega_cdm": cosmo.omega_cdm,
            "h": cosmo.h,
            "f_ede": cosmo.f_ede,
            "log10_z_c": cosmo.log10_z_c,
            "theta_i": cosmo.theta_i,
        },
        "manifest": res.manifest.to_dict(),
    }
    meta_file = out_dir / "provenance_metadata.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 6. Compute SHA-256 Checksums
    files_to_hash = [
        bg_file,
        grid_file,
        growth_file,
        desc_file,
        meta_file,
    ]
    checksum_lines: list[str] = []
    for p in files_to_hash:
        hasher = hashlib.sha256()
        hasher.update(p.read_bytes())
        digest = hasher.hexdigest()
        checksum_lines.append(f"{digest}  {p.name}")

    chk_file = out_dir / "checksums.sha256"
    chk_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"Dataset successfully frozen to {out_dir}")
    print(f"Checksums saved to {chk_file}")


if __name__ == "__main__":
    freeze_dataset()
