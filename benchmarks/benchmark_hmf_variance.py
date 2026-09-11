"""Small reproducible performance benchmark for the HMF variance kernel."""

from __future__ import annotations

import json
import time

import numpy as np

from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator


def main() -> None:
    k = np.geomspace(1.0e-6, 1.0e4, 8193)
    pk = 2.0e3 * k / (1.0 + (k / 25.0) ** 8)
    power = TabulatedPowerSpectrum(k, pk)
    radii = np.geomspace(0.03, 50.0, 256)

    rows: list[dict[str, float | int]] = []
    for chunk_size in (32, 64, 128, 256):
        calculator = VarianceCalculator(power, chunk_size=chunk_size)
        calculator.sigma_r(radii[:32])  # warm-up allocations and SciPy dispatch
        start = time.perf_counter()
        result = calculator.sigma_r(radii)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "chunk_size": chunk_size,
                "seconds": elapsed,
                "radii_per_second": radii.size / elapsed,
                "max_edge_ratio": float(np.max(result.max_edge_ratio)),
            }
        )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
