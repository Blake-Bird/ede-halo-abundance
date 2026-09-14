"""Unit tests for parameter mapping layer to AxiCLASS and CLASS_EDE."""

from __future__ import annotations

import math

import pytest

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.ede.parameter_map import (
    build_axiclass_ini_content,
    build_class_ede_ini_content,
    to_axiclass_dict,
    to_class_ede_dict,
)


def test_parameter_map_axiclass_keys() -> None:
    """Unit Test 6: Verify correct backend parameter dictionary generated for AxiCLASS."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.10,
        log10_z_c=3.55,
        theta_i=2.83,
        potential_index=3,
    )

    d = to_axiclass_dict(cosmo)
    assert d["omega_b"] == 0.02237
    assert d["omega_cdm"] == 0.1200
    assert d["h"] == 0.6736
    assert d["fraction_axion_ac"] == 0.10
    assert d["log10_axion_ac"] == pytest.approx(-math.log10(1.0 + 10.0**3.55))
    assert d["scf_parameters"] == "2.8300000000000001, 0.0"
    assert d["n_axion"] == 3
    assert d["do_shooting"] == "yes"

    req = LinearTheoryRequest(cosmology=cosmo)
    ini = build_axiclass_ini_content(req)
    assert "fraction_axion_ac = 0.1" in ini
    assert "omega_b = 0.02237" in ini


def test_parameter_map_class_ede_keys() -> None:
    """Unit Test 7: Verify correct backend parameter dictionary generated for CLASS_EDE."""
    cosmo = EDECosmology(
        omega_b=0.02237,
        omega_cdm=0.1200,
        h=0.6736,
        f_ede=0.12,
        log10_z_c=3.53,
        theta_i=2.81,
        potential_index=3,
    )

    d = to_class_ede_dict(cosmo)
    assert d["omega_b"] == 0.02237
    assert d["omega_cdm"] == 0.1200
    assert d["h"] == 0.6736
    assert d["fEDE"] == 0.12
    assert d["log10z_c"] == 3.53
    assert d["thetai_scf"] == 2.81
    assert d["Omega_Lambda"] == 0

    req = LinearTheoryRequest(cosmology=cosmo)
    ini = build_class_ede_ini_content(req)
    assert "fEDE = 0.12" in ini
    assert "Omega_Lambda = 0" in ini
