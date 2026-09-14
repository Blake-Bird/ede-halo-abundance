"""Unit tests for EDECosmology physical domain model."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from dmra.cosmology.ede.model import EDECosmology


def test_ede_model_valid_inputs() -> None:
    """Test valid instantiation of EDECosmology and property derivations."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        A_s=2.1e-9,
        n_s=0.9649,
        tau_reio=0.0544,
        f_ede=0.10,
        log10_z_c=3.55,
        theta_i=2.83,
        potential_index=3,
    )

    assert cosmo.omega_b == 0.02237
    assert cosmo.omega_cdm == 0.1200
    assert cosmo.h == 0.6736
    assert cosmo.f_ede == 0.10
    assert cosmo.is_ede is True
    assert pytest.approx(cosmo.omega_m) == 0.02237 + 0.1200
    assert pytest.approx(cosmo.H0) == 67.36
    assert pytest.approx(cosmo.z_c) == 10.0**3.55
    assert pytest.approx(cosmo.Omega_m0) == (0.02237 + 0.1200) / (0.6736**2)
    assert pytest.approx(cosmo.ln10_As) == math.log(1.0e10 * 2.1e-9)

    # Conversion to FlatLambdaCDM
    lcdm = cosmo.to_flat_lambdacdm()
    assert pytest.approx(lcdm.omega_m0) == cosmo.Omega_m0
    assert pytest.approx(lcdm.h) == cosmo.h


def test_ede_model_zero_ede_flag() -> None:
    """Test is_ede flag is False when f_ede is zero."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.0,
    )
    assert cosmo.is_ede is False


def test_ede_model_invalid_negative_f_ede() -> None:
    """Test that negative f_ede raises ValueError."""
    with pytest.raises(ValueError, match="f_ede must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            f_ede=-0.05,
        )


def test_ede_model_invalid_excessive_f_ede() -> None:
    """Test that f_ede > 0.50 raises ValueError."""
    with pytest.raises(ValueError, match="f_ede must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            f_ede=0.55,
        )


def test_ede_model_invalid_z_c_range() -> None:
    """Test that log10_z_c outside [2, 5] raises ValueError."""
    with pytest.raises(ValueError, match="log10_z_c must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            log10_z_c=1.5,
        )

    with pytest.raises(ValueError, match="log10_z_c must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            log10_z_c=5.5,
        )


def test_ede_model_invalid_theta_i_range() -> None:
    """Test that theta_i outside [0, pi] raises ValueError."""
    with pytest.raises(ValueError, match="theta_i must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            theta_i=-0.1,
        )

    with pytest.raises(ValueError, match="theta_i must lie in"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            theta_i=3.5,
        )


def test_ede_model_invalid_potential_index() -> None:
    """Test that potential_index < 1 raises ValueError."""
    with pytest.raises(ValueError, match="potential_index must be a positive integer"):
        EDECosmology(
            omega_b=0.022,
            omega_cdm=0.12,
            h=0.70,
            potential_index=0,
        )


def test_ede_model_immutability() -> None:
    """Test that EDECosmology is frozen and cannot be mutated."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.10,
    )

    with pytest.raises(FrozenInstanceError):
        cosmo.f_ede = 0.15  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        cosmo.h = 0.72  # type: ignore[misc]
