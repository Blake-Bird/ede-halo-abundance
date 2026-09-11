"""Optional independent variance check against Colossus.

Install with:
    python -m pip install -e '.[validation]'

The comparison feeds the *same Colossus-generated P(k) samples* to the local
variance integrator and then compares against Colossus's independent sigma(R)
calculation. The purpose is to test the integration/filter implementation, not
to compare two different transfer-function models.
"""

from __future__ import annotations

import json

import numpy as np

from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator


def main() -> None:
    try:
        from colossus.cosmology import cosmology
    except ImportError as exc:
        raise SystemExit("Colossus is not installed. Install the 'validation' extra.") from exc

    params = {
        "flat": True,
        "H0": 67.4,
        "Om0": 0.315,
        "Ob0": 0.0493,
        "sigma8": 0.811,
        "ns": 0.965,
    }
    reference = cosmology.setCosmology("dmra_hmf_foundations_validation", params)

    k = np.geomspace(1.0e-6, 1.0e4, 65537)
    pk = reference.matterPowerSpectrum(k, z=0.0, model="eisenstein98")
    local = VarianceCalculator(TabulatedPowerSpectrum(k, pk))

    radius = np.geomspace(0.1, 50.0, 160)
    ours = local.sigma_r(radius)
    theirs = np.asarray(reference.sigma(radius, z=0.0, ps_args={"model": "eisenstein98"}))

    relative = np.abs(ours.sigma / theirs - 1.0)
    interior = ours.max_edge_ratio < 1.0e-8
    if not np.any(interior):
        raise RuntimeError("no radius survived the finite-k support diagnostic")

    report = {
        "max_relative_error_supported": float(np.max(relative[interior])),
        "median_relative_error_supported": float(np.median(relative[interior])),
        "supported_radius_count": int(np.sum(interior)),
        "total_radius_count": int(radius.size),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
