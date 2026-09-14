"""Generate halo mass function foundation figures from a deterministic reference spectrum."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.growth import solve_linear_growth
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.cosmology.window import spherical_tophat
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import PressSchechter, ShethTormen, Tinker2008Delta200MeanZ0
from dmra.statistics.peak_height import peak_height


def save(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out = Path("artifacts/figures")
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
    save(fig, out / "hmf_01_tophat_window.png")

    fig, ax = plt.subplots()
    ax.loglog(k, pk)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    ax.set_title("Numerical reference power spectrum")
    save(fig, out / "hmf_02_power_spectrum.png")

    fig, ax = plt.subplots()
    ax.loglog(k, power.dimensionless_power)
    ax.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$\Delta^2(k)$")
    ax.set_title(r"Dimensionless power $\Delta^2(k)$")
    save(fig, out / "hmf_03_dimensionless_power.png")

    radii = np.geomspace(0.05, 30.0, 200)
    variance_result = VarianceCalculator(power).sigma_r(radii)
    mass = cosmology.radius_to_mass(radii)

    fig, ax = plt.subplots()
    ax.loglog(mass, variance_result.sigma)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel(r"$\sigma(M)$")
    ax.set_title(r"Mass variance $\sigma(M)$")
    save(fig, out / "hmf_04_mass_variance.png")

    a = np.geomspace(1.0e-3, 1.0, 400)
    growth = solve_linear_growth(cosmology, a)
    fig, ax = plt.subplots()
    ax.plot(a, growth.D, label=r"$D(a)$")
    ax.plot(a, growth.f, label=r"$f(a)=\mathrm{d}\ln D/\mathrm{d}\ln a$")
    ax.set_xlabel(r"scale factor $a$")
    ax.set_ylabel("growth")
    ax.set_title(r"Linear growth for $\Omega_{m,0}=0.315$")
    ax.legend()
    save(fig, out / "hmf_05_linear_growth.png")

    sigma_samples = np.geomspace(0.1, 4.0, 200)
    ps = PressSchechter().evaluate(sigma_samples)
    st = ShethTormen().evaluate(sigma_samples)
    tinker = Tinker2008Delta200MeanZ0().evaluate(sigma_samples)

    fig, ax = plt.subplots()
    ax.plot(sigma_samples, ps, label="Press-Schechter")
    ax.plot(sigma_samples, st, label="Sheth-Tormen")
    ax.plot(sigma_samples, tinker, label=r"Tinker (2008, $\Delta=200m$, $z=0$)")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$f(\sigma)$")
    ax.set_title(r"Multiplicity models $f(\sigma)$")
    ax.legend()
    save(fig, out / "hmf_06_multiplicity_comparison.png")

    nu = peak_height(sigma_samples)
    fig, ax = plt.subplots()
    ax.plot(nu, nu * ps, label="Press-Schechter")
    ax.plot(nu, nu * st, label="Sheth-Tormen")
    ax.set_xlabel(r"peak height $\nu=\delta_c/\sigma$")
    ax.set_ylabel(r"$\nu f(\nu)$")
    ax.set_title(r"Multiplicity per logarithmic peak height $\nu$")
    ax.legend()
    save(fig, out / "hmf_07_peak_height_multiplicity.png")

    calculator = HaloMassFunctionCalculator(rho_m0=cosmology.rho_m0)
    hmf_ps = calculator.evaluate(mass, variance_result.sigma, PressSchechter())
    hmf_st = calculator.evaluate(mass, variance_result.sigma, ShethTormen())
    hmf_tinker = calculator.evaluate(mass, variance_result.sigma, Tinker2008Delta200MeanZ0())

    fig, ax = plt.subplots()
    ax.loglog(mass, hmf_ps.dn_dln_mass, label="Press-Schechter")
    ax.loglog(mass, hmf_st.dn_dln_mass, label="Sheth-Tormen")
    ax.loglog(mass, hmf_tinker.dn_dln_mass, label="Tinker (2008)")
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel(r"$\mathrm{d} n/\mathrm{d}\ln M\,[(h/\mathrm{Mpc})^3]$")
    ax.set_title("Halo mass functions")
    ax.legend()
    save(fig, out / "hmf_08_halo_mass_function.png")

    fig, ax = plt.subplots()
    ax.semilogx(mass, variance_result.max_edge_ratio)
    ax.set_xlabel(r"$M\,[M_\odot/h]$")
    ax.set_ylabel("boundary / peak ratio")
    ax.set_title(r"Finite-support edge diagnostic for $\sigma(M)$")
    save(fig, out / "hmf_09_edge_diagnostic.png")
    print(f"HMF figures successfully saved to {out}")


if __name__ == "__main__":
    main()
