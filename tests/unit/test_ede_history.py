"""Unit tests for EDEHistory extraction and shooting error diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.history import EDEHistory
from dmra.cosmology.ede.model import EDECosmology

pytestmark = pytest.mark.external_solver


def test_ede_history_extraction() -> None:
    """Test extraction of peak fraction, transition redshift, and FWHM."""
    cosmo = EDECosmology(
        omega_b=0.02260,
        omega_cdm=0.1300,
        h=0.7210,
        f_ede=0.10,
        log10_z_c=3.55,
        theta_i=2.83,
    )
    solver = AxiCLASSSolver()
    res = solver.run(LinearTheoryRequest(cosmology=cosmo))
    hist = EDEHistory.from_background(res.background, cosmo.f_ede, cosmo.z_c)

    assert isinstance(hist.f_ede_realized, float)
    assert 0.09 < hist.f_ede_realized < 0.11
    assert abs(np.log10(hist.z_c_realized) - 3.55) < 0.1
    assert hist.fwhm_ln_1pz > 0.0
    assert hist.f_ede_shooting_error < 0.05
    assert hist.z_c_shooting_error < 0.05


def test_ede_history_zero_limit() -> None:
    """Test EDEHistory handles zero EDE cleanly without division by zero."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.0,
    )
    solver = AxiCLASSSolver()
    res = solver.run(LinearTheoryRequest(cosmology=cosmo))
    hist = EDEHistory.from_background(res.background, cosmo.f_ede, cosmo.z_c)

    assert hist.f_ede_realized == 0.0
    assert hist.f_ede_shooting_error == 0.0
    assert hist.z_c_shooting_error == 0.0


def test_compute_delta_h_over_h() -> None:
    """Test relative Hubble expansion perturbation calculation."""
    cosmo_ede = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.10,
        log10_z_c=3.55,
    )
    cosmo_lcdm = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.0,
    )

    solver = AxiCLASSSolver()
    res_ede = solver.run(LinearTheoryRequest(cosmology=cosmo_ede))
    res_lcdm = solver.run(LinearTheoryRequest(cosmology=cosmo_lcdm))

    z_eval = np.array([0.0, 1.0, 10.0, 3000.0])
    delta_h = EDEHistory.compute_delta_h_over_h(res_ede.background, res_lcdm.background, z_eval)

    assert delta_h.shape == (4,)
    # At z=0, delta_H should be close to 0 or positive due to higher h
    assert np.all(np.isfinite(delta_h))
