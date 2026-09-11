"""Comprehensive performance and precision benchmark proving DMRA improvements.

Measures:
1. Power spectrum interpolation caching speedup.
2. Top-hat window precision and catastrophic cancellation avoidance.
3. Analytical Jacobian exactness vs. numerical spline differentiation.
4. Cosmological background expansion evaluation speedup.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.interpolate import PchipInterpolator

from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.cosmology.window import spherical_tophat, spherical_tophat_derivative
from dmra.statistics.hmf import HaloMassFunctionCalculator


def benchmark_power_interpolation() -> dict[str, float]:
    k = np.geomspace(1.0e-4, 1.0e2, 2048)
    pk = 1.0e4 * k * np.exp(-k / 10.0)
    spectrum = TabulatedPowerSpectrum(k, pk)
    query = np.geomspace(1.0e-3, 1.0e1, 100)

    # Cached evaluation (our improved implementation)
    start = time.perf_counter()
    n_iters = 2000
    for _ in range(n_iters):
        spectrum.evaluate(query)
    t_cached = (time.perf_counter() - start) / n_iters

    # Uncached evaluation (the previous naive implementation rebuilding PCHIP)
    start = time.perf_counter()
    for _ in range(n_iters):
        interp = PchipInterpolator(np.log(k), np.log(pk))
        _ = np.exp(interp(np.log(query)))
    t_uncached = (time.perf_counter() - start) / n_iters

    return {
        "cached_eval_microseconds": t_cached * 1.0e6,
        "uncached_eval_microseconds": t_uncached * 1.0e6,
        "speedup_factor": t_uncached / t_cached,
    }


def benchmark_window_precision() -> dict[str, float]:
    # At x = 1e-3, direct formula 3*(sin(x) - x*cos(x))/x^3 cancels digits:
    x_test = np.array([1.0e-5, 1.0e-4, 1.0e-3, 0.01, 0.05, 0.08])
    w_improved = spherical_tophat(x_test)
    dw_improved = spherical_tophat_derivative(x_test)

    # Naive direct evaluation without Horner polynomial
    with np.errstate(divide="ignore", invalid="ignore"):
        w_direct = 3.0 * (np.sin(x_test) - x_test * np.cos(x_test)) / x_test**3
    diff_at_1e3 = float(np.abs(w_improved[2] - w_direct[2]))

    # Execution time for 100k samples
    large_x = np.geomspace(1.0e-6, 30.0, 100_000)
    start = time.perf_counter()
    for _ in range(20):
        _ = spherical_tophat(large_x)
    t_eval = (time.perf_counter() - start) / 20

    return {
        "cancellation_error_at_1e-3_if_unprotected": diff_at_1e3,
        "eval_time_100k_points_ms": t_eval * 1.0e3,
        "w_at_origin": float(spherical_tophat([0.0])[0]),
        "dw_at_origin": float(dw_improved[0]),
    }


def benchmark_background_speedup() -> dict[str, float]:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    a = np.geomspace(1.0e-3, 1.0, 50_000)

    # Fast multiplication
    start = time.perf_counter()
    for _ in range(100):
        _ = cosmology.e2(a)
    t_fast = (time.perf_counter() - start) / 100

    # Naive power operator
    start = time.perf_counter()
    for _ in range(100):
        _ = 0.315 * a**-3 + 0.685
    t_naive = (time.perf_counter() - start) / 100

    return {
        "fast_reciprocal_ms": t_fast * 1.0e3,
        "naive_power_ms": t_naive * 1.0e3,
        "speedup_factor": t_naive / t_fast,
    }


def benchmark_jacobian_accuracy() -> dict[str, float]:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    k = np.geomspace(1.0e-6, 1.0e4, 8193)
    pk = 3.0e3 * k / (1.0 + (k / 25.0) ** 8)
    power = TabulatedPowerSpectrum(k, pk)
    calc = VarianceCalculator(power)

    masses = np.geomspace(1.0e9, 1.0e15, 128)
    res = calc.sigma_m(masses, cosmology, compute_derivative=True)
    analytical_jac = res.dln_sigma_inv_dln_m
    assert analytical_jac is not None

    pchip_jac = HaloMassFunctionCalculator.logarithmic_jacobian(masses, res.sigma)

    rel_error = np.abs(analytical_jac - pchip_jac) / analytical_jac

    return {
        "median_relative_discrepancy": float(np.median(rel_error[10:-10])),
        "max_spline_discretization_error": float(np.max(rel_error[10:-10])),
    }


def main() -> None:
    results = {
        "power_spectrum_interpolation": benchmark_power_interpolation(),
        "spherical_tophat_precision": benchmark_window_precision(),
        "background_expansion_speedup": benchmark_background_speedup(),
        "jacobian_precision": benchmark_jacobian_accuracy(),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
