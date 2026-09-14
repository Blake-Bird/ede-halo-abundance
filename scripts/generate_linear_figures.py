"""Generate publication figures for EDE linear theory and halo-scale descriptors."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.descriptors import HaloScaleDescriptors
from dmra.cosmology.ede.growth import GrowthDiagnostics
from dmra.cosmology.ede.model import EDECosmology


def setup_matplotlib() -> None:
    """Set aesthetic publication styling for figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.8,
        }
    )


def make_all_figures() -> None:
    """Generate canonical linear theory figures."""
    setup_matplotlib()
    out_dir = Path("artifacts/figures").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = AxiCLASSSolver()

    # Load cosmologies
    def load_cosmo(cfg_name: str) -> EDECosmology:
        with open(Path("configs/cosmology") / cfg_name, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return EDECosmology(
            omega_b=d["omega_b"],
            omega_cdm=d["omega_cdm"],
            h=d["h"],
            A_s=d["A_s"],
            n_s=d["n_s"],
            tau_reio=d["tau_reio"],
            f_ede=d["f_ede"],
            log10_z_c=d.get("log10_z_c", 3.5),
            theta_i=d.get("theta_i", 2.8),
            potential_index=d.get("potential_index", 3),
        )

    cosmo_lcdm = load_cosmo("lcdm_reference.yaml")
    cosmo_weak = load_cosmo("ede_weak.yaml")
    cosmo_central = load_cosmo("ede_central.yaml")
    cosmo_strong = load_cosmo("ede_strong.yaml")

    res_lcdm = solver.run(LinearTheoryRequest(cosmology=cosmo_lcdm))
    res_weak = solver.run(LinearTheoryRequest(cosmology=cosmo_weak))
    res_central = solver.run(LinearTheoryRequest(cosmology=cosmo_central))
    res_strong = solver.run(LinearTheoryRequest(cosmology=cosmo_strong))

    # --- Fig 01: Scalar field potential V(phi) vs theta ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    theta = np.linspace(-np.pi, np.pi, 500)
    for n_pow in [1, 2, 3]:
        v_pot = (1.0 - np.cos(theta)) ** n_pow
        ax.plot(theta, v_pot / (2.0**n_pow), label=f"$n = {n_pow}$")
    ax.set_xlabel(r"Misalignment Angle $\theta \equiv \phi / f$")
    ax.set_ylabel(r"$V(\theta) / V_{\rm max}$")
    ax.set_title(r"Axion-like Potential $V(\phi) = m^2 f^2 [1 - \cos(\phi/f)]^n$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "01_axion_potential_shapes.png")
    plt.close(fig)

    # --- Fig 02: Omega_phi(z) history ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for res, lab, col in [
        (res_weak, "Weak ($f_{\\rm EDE}=0.05$)", "#2ca02c"),
        (res_central, "Central ($f_{\\rm EDE}=0.10$)", "#1f77b4"),
        (res_strong, "Strong ($f_{\\rm EDE}=0.15$)", "#d62728"),
    ]:
        z = res.background.z
        mask = (z >= 1.0) & (z <= 1e5)
        ax.plot(1.0 + z[mask], res.background.omega_phi[mask], label=lab, color=col)
    ax.set_xscale("log")
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel(r"$\Omega_\phi(z) \equiv \rho_\phi / \rho_{\rm tot}$")
    ax.set_title(r"Early Dark Energy Fraction History $\Omega_\phi(z)$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "02_omega_phi_history.png")
    plt.close(fig)

    # --- Fig 03: Fractional Hubble Expansion Perturbation Delta H / H ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    z_ref = res_lcdm.background.z
    h_ref = res_lcdm.background.H_z
    for res, lab, col in [
        (res_weak, "Weak", "#2ca02c"),
        (res_central, "Central", "#1f77b4"),
        (res_strong, "Strong", "#d62728"),
    ]:
        z_curr = res.background.z
        h_curr = res.background.H_z
        mask = (z_curr >= 1.0) & (z_curr <= 1e5)
        h_interp = np.interp(z_curr[mask], z_ref, h_ref)
        delta_h = (h_curr[mask] - h_interp) / h_interp
        ax.plot(1.0 + z_curr[mask], delta_h * 100.0, label=lab, color=col)
    ax.set_xscale("log")
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel(r"$\Delta H(z) / H_{\Lambda\rm CDM}(z)\;[\%]$")
    ax.set_title(r"Hubble Expansion Rate Perturbation $\Delta H / H$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "03_hubble_expansion_perturbation.png")
    plt.close(fig)

    # --- Fig 04: Linear Matter Power Spectrum P(k, z=0) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    pk_lcdm = res_lcdm.power_grid.at_redshift(0.0)
    ax.plot(
        pk_lcdm.k,
        pk_lcdm.pk,
        label=r"$\Lambda\mathrm{CDM}$ Reference",
        color="black",
        linestyle="--",
    )
    for res, lab, col in [
        (res_weak, "Weak EDE", "#2ca02c"),
        (res_central, "Central EDE", "#1f77b4"),
        (res_strong, "Strong EDE", "#d62728"),
    ]:
        pk_m = res.power_grid.at_redshift(0.0)
        ax.plot(pk_m.k, pk_m.pk, label=lab, color=col)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_m(k, z=0)\;[(\mathrm{Mpc}/h)^3]$")
    ax.set_title(r"Linear Matter Power Spectra at $z=0$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "04_matter_power_spectra_z0.png")
    plt.close(fig)

    # --- Fig 05: Power Spectrum Ratio P_EDE / P_LCDM at z=0 and z=5 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    k_eval = np.geomspace(1e-3, 10.0, 300)
    for z_target, ax_target, tit in [(0.0, ax1, "$z=0$"), (5.0, ax2, "$z=5$")]:
        pk_l_grid = res_lcdm.power_grid.at_redshift(z_target)
        pk_l_vals = pk_l_grid.evaluate(k_eval)
        for res, lab, col in [
            (res_weak, "Weak EDE", "#2ca02c"),
            (res_central, "Central EDE", "#1f77b4"),
            (res_strong, "Strong EDE", "#d62728"),
        ]:
            pk_curr = res.power_grid.at_redshift(z_target).evaluate(k_eval)
            ax_target.plot(k_eval, pk_curr / pk_l_vals, label=lab, color=col)
        ax_target.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
        ax_target.set_xscale("log")
        ax_target.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
        ax_target.set_title(f"Power Ratio at {tit}")
        ax_target.legend(frameon=True)
    ax1.set_ylabel(r"$P_{\rm EDE}(k, z) / P_{\Lambda\rm CDM}(k, z)$")
    fig.tight_layout()
    fig.savefig(out_dir / "05_power_spectrum_ratio_z0_z5.png")
    plt.close(fig)

    # --- Fig 06: Linear Mass Variance sigma(M) at z=0 and z=5 ---
    desc_lcdm = HaloScaleDescriptors(res_lcdm.power_grid, cosmo_lcdm)
    desc_central = HaloScaleDescriptors(res_central.power_grid, cosmo_central)
    masses = np.geomspace(1e8, 1e16, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sig_l_0 = desc_lcdm.evaluate(masses, redshift=0.0).sigma
    sig_c_0 = desc_central.evaluate(masses, redshift=0.0).sigma
    sig_l_5 = desc_lcdm.evaluate(masses, redshift=5.0).sigma
    sig_c_5 = desc_central.evaluate(masses, redshift=5.0).sigma

    ax1.plot(masses, sig_l_0, label=r"$\Lambda\mathrm{CDM}\;(z=0)$", color="black", linestyle="--")
    ax1.plot(masses, sig_c_0, label=r"Central EDE $(z=0)$", color="#1f77b4")
    ax1.plot(masses, sig_l_5, label=r"$\Lambda\mathrm{CDM}\;(z=5)$", color="gray", linestyle="--")
    ax1.plot(masses, sig_c_5, label=r"Central EDE $(z=5)$", color="#ff7f0e")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Halo Mass $M\;[M_\odot / h]$")
    ax1.set_ylabel(r"Linear Variance $\sigma(M, z)$")
    ax1.set_title(r"Mass Variance $\sigma(M, z)$")
    ax1.legend(frameon=True)

    # Ratio
    ax2.plot(masses, sig_c_0 / sig_l_0, label="$z=0$", color="#1f77b4")
    ax2.plot(masses, sig_c_5 / sig_l_5, label="$z=5$", color="#ff7f0e")
    ax2.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"Halo Mass $M\;[M_\odot / h]$")
    ax2.set_ylabel(r"$\sigma_{\rm EDE}(M) / \sigma_{\Lambda\rm CDM}(M)$")
    ax2.set_title(r"Variance Enhancement Ratio")
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_dir / "06_variance_sigma_m.png")
    plt.close(fig)

    # --- Fig 07: Effective Growth Scale-Dependence Diagnostic ---
    growth_central = GrowthDiagnostics.from_power_grid(res_central.power_grid, k_ref_h_mpc=0.01)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for z_t in [0.5, 1.0, 2.0, 3.0, 5.0]:
        idx_z = int(np.argmin(np.abs(growth_central.redshifts - z_t)))
        eps = growth_central.epsilon_d_grid[idx_z, :]
        ax.plot(growth_central.k_h_mpc, eps * 100.0, label=f"$z={z_t}$")
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.5)
    ax.axhspan(-0.5, 0.5, color="green", alpha=0.1, label=r"$\pm 0.5\%$ Bound")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(
        r"$\epsilon_D(k, z) \equiv D_{\rm eff}(k,z)/D_{\rm eff}(k_{\rm ref},z) - 1\;[\%]$"
    )
    ax.set_title(r"Effective Growth Scale-Dependence $\epsilon_D(k, z)$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "07_growth_scale_dependence.png")
    plt.close(fig)

    print(f"Publication figures successfully generated in {out_dir}")


if __name__ == "__main__":
    make_all_figures()
