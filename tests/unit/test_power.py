import numpy as np
import pytest

from dmra.cosmology.power import TabulatedPowerSpectrum


def test_amplitude_rescaling() -> None:
    k = np.geomspace(1.0e-4, 1.0e2, 128)
    pk = 1.0e4 * k * np.exp(-k / 10.0)
    spectrum = TabulatedPowerSpectrum(k, pk)
    scaled = spectrum.rescale_amplitude(4.0)
    np.testing.assert_allclose(scaled.pk, 4.0 * spectrum.pk)


def test_extrapolation_is_rejected_by_default() -> None:
    k = np.geomspace(1.0e-4, 1.0e2, 128)
    pk = 1.0e4 * k * np.exp(-k / 10.0)
    spectrum = TabulatedPowerSpectrum(k, pk)
    with pytest.raises(ValueError, match="outside"):
        spectrum.evaluate(np.array([1.0e-5]))
