"""Unit tests for LinearTheoryRequest, BoltzmannSolver protocol, and SolverManifest."""

from __future__ import annotations

import pytest

from dmra.cosmology.boltzmann.manifest import SolverManifest, compute_sha256
from dmra.cosmology.boltzmann.protocol import (
    CANONICAL_REDSHIFTS,
    BoltzmannSolver,
    LinearTheoryRequest,
)
from dmra.cosmology.boltzmann.result import (
    BackgroundHistory,
    LinearTheoryResult,
    MatterPowerGrid,
)
from dmra.cosmology.ede.model import EDECosmology


def test_linear_theory_request_defaults() -> None:
    """Test standard defaults and validation of LinearTheoryRequest."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736)
    req = LinearTheoryRequest(cosmology=cosmo)

    assert req.cosmology == cosmo
    assert req.redshifts == CANONICAL_REDSHIFTS
    assert req.k_min_h_mpc == 1.0e-4
    assert req.k_max_h_mpc == 1.0e2
    assert req.num_k_points == 500
    assert req.gauge == "synchronous"


def test_linear_theory_request_invalid_k_bounds() -> None:
    """Test validation rejection when k_max <= k_min."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736)
    with pytest.raises(ValueError, match="k_max_h_mpc must be strictly greater"):
        LinearTheoryRequest(cosmology=cosmo, k_min_h_mpc=1.0, k_max_h_mpc=0.5)


def test_linear_theory_request_invalid_gauge() -> None:
    """Test validation rejection of unsupported gauge."""
    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736)
    with pytest.raises(ValueError, match="gauge must be 'synchronous' or 'newtonian'"):
        LinearTheoryRequest(cosmology=cosmo, gauge="conformal")  # type: ignore[arg-type]


def test_solver_manifest_deterministic_hashing() -> None:
    """Test that identical configs generate identical SHA-256 digests."""
    cfg1 = {"omega_b": 0.02237, "omega_cdm": 0.1200, "h": 0.6736, "f_ede": 0.10}
    # Keys in different order
    cfg2 = {"f_ede": 0.10, "h": 0.6736, "omega_cdm": 0.1200, "omega_b": 0.02237}

    hash1 = compute_sha256(cfg1)
    hash2 = compute_sha256(cfg2)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_mock_solver_conforms_to_protocol() -> None:
    """Verify a minimal BoltzmannSolver implementation satisfies the Protocol."""

    class MockSolver:
        @property
        def name(self) -> str:
            return "mock_boltzmann"

        @property
        def version(self) -> str:
            return "v1.0.0-test"

        def run(self, request: LinearTheoryRequest) -> LinearTheoryResult:
            import numpy as np

            z = np.array([0.0, 1.0])
            a = 1.0 / (1.0 + z)
            h = np.array([70.0, 120.0])
            om = np.array([0.3, 0.6])
            ol = np.array([0.7, 0.4])
            orad = np.zeros_like(z)
            ophi = np.zeros_like(z)
            bg = BackgroundHistory(
                z=z, a=a, H_z=h, omega_m=om, omega_r=orad, omega_lambda=ol, omega_phi=ophi
            )

            k = np.geomspace(1e-3, 10.0, 20)
            pk = np.ones((len(z), len(k))) * 1000.0
            grid = MatterPowerGrid(k_h_mpc=k, redshifts=z, pk_grid=pk)

            manifest = SolverManifest(
                solver_name=self.name,
                solver_version=self.version,
                config_hash=compute_sha256({"test": 1}),
                input_parameters={"test": 1},
                raw_units={"k": "h/Mpc", "P": "(Mpc/h)^3"},
                execution_time_seconds=0.01,
            )

            return LinearTheoryResult(
                cosmology=request.cosmology,
                background=bg,
                power_grid=grid,
                manifest=manifest,
            )

    solver = MockSolver()
    assert isinstance(solver, BoltzmannSolver)

    cosmo = EDECosmology(omega_b=0.02237, omega_cdm=0.1200, h=0.6736)
    req = LinearTheoryRequest(cosmology=cosmo, redshifts=(0.0, 1.0))
    res = solver.run(req)

    assert isinstance(res, LinearTheoryResult)
    assert res.manifest.solver_name == "mock_boltzmann"
    assert res.background.validate_flatness() is True
    assert res.power_grid.at_redshift(0.0).k.size == 20
