"""Unit tests for UnitGatekeeper and dimensional consistency."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import simpson  # type: ignore[import-untyped]

from dmra.cosmology.boltzmann.units import SPEED_OF_LIGHT_KM_S, UnitGatekeeper


def test_class_to_project_power() -> None:
    """Test scaling laws k_h = k/h and P_h = P * h^3."""
    h = 0.70
    k_mpc = np.array([0.01, 0.1, 1.0])
    pk_mpc3 = np.array([10000.0, 1000.0, 10.0])

    k_h, pk_h = UnitGatekeeper.class_to_project_power(k_mpc, pk_mpc3, h)

    np.testing.assert_allclose(k_h, k_mpc / h)
    np.testing.assert_allclose(pk_h, pk_mpc3 * (h**3))


def test_round_trip_conversion() -> None:
    """Test that round trip conversion reproduces input to machine precision."""
    h = 0.6736
    k_orig = np.geomspace(1e-4, 50.0, 100)
    pk_orig = 1000.0 * (k_orig / 0.05) ** (-1.5)

    k_h, pk_h = UnitGatekeeper.class_to_project_power(k_orig, pk_orig, h)
    k_back, pk_back = UnitGatekeeper.project_to_class_power(k_h, pk_h, h)

    np.testing.assert_allclose(k_back, k_orig, rtol=1e-15)
    np.testing.assert_allclose(pk_back, pk_orig, rtol=1e-15)


def test_variance_integral_invariance() -> None:
    r"""Test physical invariant: \int k^2 dk P(k) is identical in CLASS vs DMRA units."""
    h = 0.72
    k_mpc = np.geomspace(1e-3, 10.0, 500)
    # Simple toy power spectrum P(k) = A / (1 + (k/k_eq)^2)
    pk_mpc = 5000.0 / (1.0 + (k_mpc / 0.05) ** 2)

    # In CLASS units: I_class = \int k^2 P(k) dk
    integrand_class = k_mpc**2 * pk_mpc
    integral_class = float(simpson(integrand_class, x=k_mpc))

    # In DMRA project units:
    k_h, pk_h = UnitGatekeeper.class_to_project_power(k_mpc, pk_mpc, h)
    integrand_project = k_h**2 * pk_h
    integral_project = float(simpson(integrand_project, x=k_h))

    # Because (k/h)^2 * dk/h * P * h^3 = k^2 dk P, the integrals must match identically
    assert pytest.approx(integral_project, rel=1e-10) == integral_class


def test_hubble_conversion() -> None:
    """Test Hubble conversion from 1/Mpc to km/s/Mpc."""
    # Suppose H = 1 / 3000 Mpc^-1
    h_inv_mpc = np.array([1.0 / 3000.0])
    h_kms = UnitGatekeeper.hubble_inv_mpc_to_kms_mpc(h_inv_mpc)

    expected = SPEED_OF_LIGHT_KM_S / 3000.0
    np.testing.assert_allclose(h_kms, [expected])


def test_unit_gatekeeper_invalid_inputs() -> None:
    """Test error handling for non-positive h or invalid values."""
    with pytest.raises(ValueError, match="dimensionless Hubble parameter h must be positive"):
        UnitGatekeeper.class_to_project_power([0.1, 0.2], [100.0, 50.0], h=-0.7)

    with pytest.raises(ValueError, match="strictly positive"):
        UnitGatekeeper.class_to_project_power([0.1, -0.2], [100.0, 50.0], h=0.7)
