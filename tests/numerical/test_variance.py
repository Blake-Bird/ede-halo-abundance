import numpy as np

from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator


def _smooth_test_spectrum(amplitude: float) -> TabulatedPowerSpectrum:
    k = np.geomspace(1.0e-6, 1.0e4, 32769)
    pk = amplitude * k / (1.0 + (k / 30.0) ** 8)
    return TabulatedPowerSpectrum(k, pk)


def _sampled_smooth_spectrum(sample_count: int) -> TabulatedPowerSpectrum:
    k = np.geomspace(1.0e-6, 1.0e4, sample_count)
    pk = 1.0e3 * k / (1.0 + (k / 30.0) ** 8)
    return TabulatedPowerSpectrum(k, pk)


def test_sigma_scales_as_square_root_of_power_amplitude() -> None:
    radii = np.geomspace(0.05, 30.0, 64)
    sigma1 = VarianceCalculator(_smooth_test_spectrum(1.0e3)).sigma_r(radii).sigma
    sigma4 = VarianceCalculator(_smooth_test_spectrum(4.0e3)).sigma_r(radii).sigma
    np.testing.assert_allclose(sigma4, 2.0 * sigma1, rtol=4.0e-13, atol=0.0)


def test_variance_edge_diagnostic_is_small_for_well_supported_scales() -> None:
    radii = np.geomspace(0.1, 10.0, 32)
    result = VarianceCalculator(_smooth_test_spectrum(1.0e3)).sigma_r(radii)
    assert float(np.max(result.max_edge_ratio)) < 1.0e-8


def test_log_k_quadrature_converges_under_grid_refinement() -> None:
    """A resolved spectrum must be stable when its log-k grid is refined."""
    radii = np.geomspace(0.1, 10.0, 32)
    resolved = VarianceCalculator(_sampled_smooth_spectrum(32769)).sigma_r(radii).sigma
    reference = VarianceCalculator(_sampled_smooth_spectrum(65537)).sigma_r(radii).sigma
    np.testing.assert_allclose(resolved, reference, rtol=2.0e-10, atol=0.0)
