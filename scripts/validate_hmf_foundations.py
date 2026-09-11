"""Deterministic HMF-foundation smoke validation using a smooth synthetic spectrum."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.growth import solve_linear_growth
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import PressSchechter, ShethTormen, Tinker2008Delta200MeanZ0


def main() -> None:
    output = Path("artifacts/hmf_foundations")
    output.mkdir(parents=True, exist_ok=True)

    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    k = np.geomspace(1.0e-6, 1.0e4, 32769)
    pk = 3.0e3 * k / (1.0 + (k / 25.0) ** 8)
    power = TabulatedPowerSpectrum(k, pk)

    mass = np.geomspace(1.0e8, 1.0e16, 512)
    variance = VarianceCalculator(power).sigma_m(mass, cosmology)

    if float(np.max(variance.max_edge_ratio)) >= 1.0e-7:
        raise RuntimeError(
            "synthetic validation spectrum does not adequately support the mass grid"
        )
    if np.any(np.diff(variance.sigma) >= 0.0):
        raise RuntimeError("sigma(M) must decrease across the validation domain")

    hmf = HaloMassFunctionCalculator(rho_m0=cosmology.rho_m0)
    models = (PressSchechter(), ShethTormen(), Tinker2008Delta200MeanZ0())
    results = {model.name: hmf.evaluate(mass, variance.sigma, model) for model in models}

    a = np.geomspace(1.0e-3, 1.0, 256)
    growth = solve_linear_growth(cosmology, a)

    summary = {
        "mass_min_msun_h": float(mass[0]),
        "mass_max_msun_h": float(mass[-1]),
        "sigma_min": float(variance.sigma[-1]),
        "sigma_max": float(variance.sigma[0]),
        "max_variance_edge_ratio": float(np.max(variance.max_edge_ratio)),
        "D_at_a1": float(growth.D[-1]),
        "f_at_a1": float(growth.f[-1]),
        "hmf_finite": {
            name: bool(np.all(np.isfinite(result.dn_dln_mass))) for name, result in results.items()
        },
    }

    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
