"""Generate HMF-foundation figures from a deterministic reference spectrum.

The built-in spectrum is intentionally a smooth numerical test spectrum, not a
claim about the physical LCDM power spectrum. Replace it with CLASS data in the
external-validation pass before using the figures in research communication.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.growth import solve_linear_growth
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.cosmology.window import spherical_tophat
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import PressSchechter, ShethTormen, Tinker2008Delta200MeanZ0
from dmra.statistics.peak_height import peak_height


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out = Path("artifacts/hmf_foundations/figures")
    out.mkdir(parents=True, exist_ok=True)

    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    k = np.geomspace(1.0e-6, 1.0e4, 8193)
    pk = 3.0e3 * k / (1.0 + (k / 25.0) ** 8)
    power = TabulatedPowerSpectrum(k, pk)

    x = np.linspace(0.0, 30.0, 4000)
    fig, ax = plt.subplots()
    ax.plot(x, spherical_tophat(x))
    ax.set_xlabel(r"$kR$")
    ax.set_ylabel(r"$W(kR)$")
    ax.set_title("Spherical top-hat Fourier window")
    save(fig, out / "01_tophat_window.png")

    fig, ax = plt.subplots()
    ax.loglog(k, pk)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    ax.set_title("Numerical reference power spectrum")
    save(fig, out / "02_power_spectrum.png")

    fig, ax = plt.subplots()
    ax.loglog(k, power.dimensionless_power)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.set_title("Dimensionless power")
    save(fig, out / "03_dimensionless_power.png")

    masses_for_integrand = np.array([1.0e9, 1.0e12, 1.0e15])
    radii_for_integrand = cosmology.mass_to_radius(masses_for_integrand)
    fig, ax = plt.subplots()
    for mass, radius in zip(masses_for_integrand, radii_for_integrand, strict=True):
        integrand = power.dimensionless_power * spherical_tophat(k * radius) ** 2
        ax.loglog(k, integrand, label=rf"$M={mass:.0e}\,M_\odot/h$")
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)W^2(kR)$")
    ax.set_title("Contribution to the variance per logarithmic k interval")
    ax.legend()
    save(fig, out / "04_variance_integrands.png")

    mass = np.geomspace(1.0e8, 1.0e16, 512)
    variance = VarianceCalculator(power).sigma_m(mass, cosmology)

    fig, ax = plt.subplots()
    ax.loglog(mass, variance.sigma)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel(r"$\sigma(M)$")
    ax.set_title("Top-hat-smoothed variance")
    save(fig, out / "05_sigma_mass.png")

    nu = peak_height(variance.sigma)
    fig, ax = plt.subplots()
    ax.loglog(mass, nu)
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel(r"$\nu(M)$")
    ax.set_title("Peak height")
    save(fig, out / "06_peak_height.png")

    sigma_grid = np.geomspace(0.1, 20.0, 2048)
    fig, ax = plt.subplots()
    for model in (PressSchechter(), ShethTormen()):
        ax.loglog(peak_height(sigma_grid), model.evaluate(sigma_grid), label=model.name)
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel("multiplicity")
    ax.set_title("Analytic multiplicity models")
    ax.legend()
    save(fig, out / "07_multiplicity.png")

    calculator = HaloMassFunctionCalculator(rho_m0=cosmology.rho_m0)
    hmf_results = [
        calculator.evaluate(mass, variance.sigma, model)
        for model in (PressSchechter(), ShethTormen(), Tinker2008Delta200MeanZ0())
    ]

    fig, ax = plt.subplots()
    for result in hmf_results:
        ax.loglog(result.mass, result.dn_dln_mass, label=result.model_name)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel(r"$dn/d\ln M\,[(h/\mathrm{Mpc})^3]$")
    ax.set_title("Halo mass function on the numerical reference spectrum")
    ax.legend()
    save(fig, out / "08_hmf_models.png")

    ps = hmf_results[0]
    fig, ax = plt.subplots()
    for result in hmf_results[1:]:
        ax.semilogx(result.mass, result.dn_dln_mass / ps.dn_dln_mass, label=result.model_name)
    ax.axhline(1.0, linewidth=1.0)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel("ratio to Press-Schechter")
    ax.set_title("Mass-function model ratios")
    ax.legend()
    save(fig, out / "09_hmf_ratios.png")

    a = np.geomspace(1.0e-3, 1.0, 512)
    growth_lcdm = solve_linear_growth(cosmology, a)
    growth_eds = solve_linear_growth(FlatLambdaCDM(omega_m0=1.0, h=0.674), a)

    fig, ax = plt.subplots()
    ax.loglog(a, growth_lcdm.D, label="flat LCDM reference")
    ax.loglog(a, growth_eds.D, label="Einstein-de Sitter")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$D(a)$")
    ax.set_title("Normalized linear growth")
    ax.legend()
    save(fig, out / "10_growth_factor.png")

    fig, ax = plt.subplots()
    ax.semilogx(a, growth_lcdm.f, label="flat LCDM reference")
    ax.semilogx(a, growth_eds.f, label="Einstein-de Sitter")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$f=d\ln D/d\ln a$")
    ax.set_title("Linear growth rate")
    ax.legend()
    save(fig, out / "11_growth_rate.png")


if __name__ == "__main__":
    main()
