import numpy as np

from dmra.cosmology.background import FlatLambdaCDM


def test_mass_radius_round_trip() -> None:
    cosmology = FlatLambdaCDM(omega_m0=0.315, h=0.674)
    mass = np.geomspace(1.0e8, 1.0e16, 256)
    reconstructed = cosmology.radius_to_mass(cosmology.mass_to_radius(mass))
    np.testing.assert_allclose(reconstructed, mass, rtol=2.0e-15, atol=0.0)


def test_eds_e2() -> None:
    cosmology = FlatLambdaCDM(omega_m0=1.0, h=0.7)
    a = np.geomspace(1.0e-3, 1.0, 64)
    np.testing.assert_allclose(cosmology.e2(a), a**-3, rtol=1.0e-15, atol=0.0)
