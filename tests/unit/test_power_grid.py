"""Unit tests for MatterPowerGrid and BackgroundHistory."""

from __future__ import annotations

import numpy as np
import pytest

from dmra.cosmology.boltzmann.result import BackgroundHistory, MatterPowerGrid
from dmra.cosmology.power import TabulatedPowerSpectrum


@pytest.fixture
def sample_power_grid() -> MatterPowerGrid:
    """Fixture providing a synthetic 2D matter power grid across 3 redshifts."""
    k = np.geomspace(1e-3, 10.0, 50)
    redshifts = np.array([0.0, 1.0, 2.0])

    # Power spectrum decreasing with redshift: P(k, z) = P_0(k) / (1 + z)^2
    pk_0 = 2000.0 / (1.0 + (k / 0.05) ** 1.8)
    pk_grid = np.array([pk_0 / (1.0 + z) ** 2 for z in redshifts])

    return MatterPowerGrid(k_h_mpc=k, redshifts=redshifts, pk_grid=pk_grid)


def test_power_grid_exact_redshift_slice(sample_power_grid: MatterPowerGrid) -> None:
    """Test extracting an exact grid node returns a valid TabulatedPowerSpectrum."""
    pk_z0 = sample_power_grid.at_redshift(0.0)
    assert isinstance(pk_z0, TabulatedPowerSpectrum)
    np.testing.assert_allclose(pk_z0.k, sample_power_grid.k_h_mpc)
    np.testing.assert_allclose(pk_z0.pk, sample_power_grid.pk_grid[0])


def test_power_grid_interpolated_redshift_slice(sample_power_grid: MatterPowerGrid) -> None:
    """Test intermediate redshift interpolation is smooth and bounded."""
    pk_interp = sample_power_grid.at_redshift(0.5)
    assert isinstance(pk_interp, TabulatedPowerSpectrum)

    pk_0 = sample_power_grid.at_redshift(0.0).pk
    pk_1 = sample_power_grid.at_redshift(1.0).pk

    # Power at z=0.5 must lie strictly between z=1.0 and z=0.0
    assert np.all(pk_interp.pk < pk_0)
    assert np.all(pk_interp.pk > pk_1)


def test_power_grid_extrapolation_rejection(sample_power_grid: MatterPowerGrid) -> None:
    """Test that requesting z outside support raises ValueError."""
    with pytest.raises(ValueError, match="lies outside tabulated bounds"):
        sample_power_grid.at_redshift(3.5)


def test_power_grid_extrapolation_permitted(sample_power_grid: MatterPowerGrid) -> None:
    """Test that requesting z outside support with extrapolate=True succeeds and scales smoothly."""
    pk_extrap = sample_power_grid.at_redshift(3.0, extrapolate=True)
    assert isinstance(pk_extrap, TabulatedPowerSpectrum)
    pk_2 = sample_power_grid.at_redshift(2.0).pk
    assert np.all(pk_extrap.pk < pk_2)


def test_power_grid_effective_growth(sample_power_grid: MatterPowerGrid) -> None:
    """Test effective growth factor normalization and monotonicity."""
    d0 = sample_power_grid.growth_factor(0.0)
    assert pytest.approx(d0) == 1.0

    d1 = sample_power_grid.growth_factor(1.0)
    d2 = sample_power_grid.growth_factor(2.0)

    # In our toy model D(z) = 1 / (1+z)
    assert pytest.approx(d1, rel=1e-3) == 0.5
    assert pytest.approx(d2, rel=1e-3) == 1.0 / 3.0
    assert d0 > d1 > d2


def test_background_history_flatness_and_hubble() -> None:
    """Test BackgroundHistory flatness verification and PCHIP Hubble interpolation."""
    z = np.linspace(0.0, 5.0, 50)
    a = 1.0 / (1.0 + z)
    # Flat LambdaCDM toy densities
    om0 = 0.3
    ol0 = 0.7
    h_z = 70.0 * np.sqrt(om0 * (1.0 + z) ** 3 + ol0)

    om_z = om0 * (1.0 + z) ** 3 / (om0 * (1.0 + z) ** 3 + ol0)
    ol_z = ol0 / (om0 * (1.0 + z) ** 3 + ol0)
    or_z = np.zeros_like(z)
    op_z = np.zeros_like(z)

    bg = BackgroundHistory(
        z=z,
        a=a,
        H_z=h_z,
        omega_m=om_z,
        omega_r=or_z,
        omega_lambda=ol_z,
        omega_phi=op_z,
    )

    assert bg.validate_flatness(tolerance=1e-5) is True

    # Check Hubble interpolation at z=0 and z=1.5
    np.testing.assert_allclose(bg.hubble(0.0), 70.0, rtol=1e-4)
    expected_h15 = 70.0 * np.sqrt(om0 * 2.5**3 + ol0)
    np.testing.assert_allclose(bg.hubble(1.5), expected_h15, rtol=1e-4)


def test_background_history_flatness_violation() -> None:
    """Test that BackgroundHistory raises ValueError when flatness is broken."""
    z = np.array([0.0, 1.0])
    a = np.array([1.0, 0.5])
    h = np.array([70.0, 120.0])
    om = np.array([0.3, 0.6])
    ol = np.array([0.7, 0.4])
    orad = np.array([0.0, 0.0])
    ophi = np.array([0.1, 0.1])  # Causes sum = 1.1

    bg = BackgroundHistory(
        z=z,
        a=a,
        H_z=h,
        omega_m=om,
        omega_r=orad,
        omega_lambda=ol,
        omega_phi=ophi,
    )

    with pytest.raises(ValueError, match="cosmic flatness violated"):
        bg.validate_flatness(tolerance=1e-4)


def test_background_history_ede_peak_detection() -> None:
    """Test peak extraction of realized f_ede and z_c."""
    z = np.linspace(100.0, 10000.0, 100)
    a = 1.0 / (1.0 + z)
    h_z = 70.0 * (1.0 + z) ** 1.5

    # Gaussian bump in ln(1+z) centered at z_c = 3500
    ln_1pz = np.log1p(z)
    ln_zc = np.log1p(3500.0)
    op = 0.10 * np.exp(-0.5 * ((ln_1pz - ln_zc) / 0.5) ** 2)

    om = 0.3 * np.ones_like(z)
    ol = 0.7 * np.ones_like(z) - op
    orad = np.zeros_like(z)

    bg = BackgroundHistory(
        z=z,
        a=a,
        H_z=h_z,
        omega_m=om,
        omega_r=orad,
        omega_lambda=ol,
        omega_phi=op,
    )

    f_peak, z_peak = bg.omega_phi_peak()
    assert pytest.approx(f_peak, rel=1e-2) == 0.10
    assert pytest.approx(z_peak, rel=0.05) == 3500.0
    assert bg.fwhm_ln_1pz() > 0.0
