"""Audit parameter mapping, shooting accuracy, and LCDM zero-limit invariant across benchmark models."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.class_ede import CLASSEDESolver
from dmra.cosmology.ede.history import EDEHistory
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.ede.parameter_map import to_axiclass_dict, to_class_ede_dict


def run_parameter_mapping_audit() -> None:
    """Execute parameter mapping and shooting audit across all benchmark configs."""
    configs_dir = Path("configs/cosmology").resolve()
    out_dir = Path("artifacts/tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "parameter_mapping_audit.csv"

    config_files = [
        "lcdm_reference.yaml",
        "ede_weak.yaml",
        "ede_central.yaml",
        "ede_strong.yaml",
    ]

    solver_axi = AxiCLASSSolver()
    solver_ede = CLASSEDESolver()

    results_rows: list[dict[str, str | float]] = []

    # First evaluate pure LCDM baseline
    with open(configs_dir / "lcdm_reference.yaml", encoding="utf-8") as f:
        lcdm_data = yaml.safe_load(f)
    lcdm_cosmo = EDECosmology(
        omega_b=lcdm_data["omega_b"],
        omega_cdm=lcdm_data["omega_cdm"],
        h=lcdm_data["h"],
        A_s=lcdm_data["A_s"],
        n_s=lcdm_data["n_s"],
        tau_reio=lcdm_data["tau_reio"],
        f_ede=lcdm_data["f_ede"],
    )
    req_lcdm = LinearTheoryRequest(cosmology=lcdm_cosmo)
    res_lcdm = solver_axi.run(req_lcdm)
    bg_lcdm = res_lcdm.background

    # Audit: Test zero EDE limit against flat LCDM background with radiation
    omega_m = lcdm_cosmo.Omega_m0
    omega_l = lcdm_cosmo.Omega_lambda0
    omega_r = float(bg_lcdm.omega_r[0])
    z_test = np.linspace(0.0, 10.0, 100)
    a_test = 1.0 / (1.0 + z_test)
    h_analytic = lcdm_cosmo.H0 * np.sqrt(omega_m * a_test**-3 + omega_l + omega_r * a_test**-4)
    h_solver = bg_lcdm.hubble(z_test)
    max_lcdm_rel_err = float(np.max(np.abs(h_solver - h_analytic) / h_analytic))

    print(f"LCDM Zero-Limit Invariant: max |Delta H / H| = {max_lcdm_rel_err:.3e}")
    assert max_lcdm_rel_err < 1e-4, f"LCDM zero-limit failed: {max_lcdm_rel_err}"

    for cfg_name in config_files:
        cfg_path = configs_dir / cfg_name
        with open(cfg_path, encoding="utf-8") as f:
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
        axi_res = solver_axi.run(req)
        ede_res = solver_ede.run(req)

        axi_hist = EDEHistory.from_background(axi_res.background, cosmo.f_ede, cosmo.z_c)
        ede_hist = EDEHistory.from_background(ede_res.background, cosmo.f_ede, cosmo.z_c)
        # Cross-check parameter dictionary keys
        axi_dict = to_axiclass_dict(cosmo)
        ede_dict = to_class_ede_dict(cosmo)
        assert len(axi_dict) > 0
        assert len(ede_dict) > 0
        assert ede_hist.f_ede_realized >= 0.0

        results_rows.append(
            {
                "model_name": data["model_name"],
                "requested_f_ede": cosmo.f_ede,
                "realized_f_ede": axi_hist.f_ede_realized,
                "f_ede_error_percent": axi_hist.f_ede_shooting_error * 100.0,
                "requested_z_c": cosmo.z_c,
                "realized_z_c": axi_hist.z_c_realized,
                "z_c_error_percent": axi_hist.z_c_shooting_error * 100.0,
                "fwhm_ln_1pz": axi_hist.fwhm_ln_1pz,
                "cross_solver_h0_diff": abs(axi_res.background.H_z[0] - ede_res.background.H_z[0]),
                "lcdm_zero_limit_verified": str(max_lcdm_rel_err < 1e-4),
            }
        )

    # Write CSV table
    fieldnames = list(results_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Successfully generated {out_csv}")


if __name__ == "__main__":
    run_parameter_mapping_audit()
