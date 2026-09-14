"""Inputs recognized by the pinned AxiCLASS and CLASS_EDE source revisions."""

from __future__ import annotations

import math
from typing import Any

from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.model import EDECosmology


def _common(c: EDECosmology) -> dict[str, Any]:
    if c.omega_r0 != 0:
        raise ValueError("custom omega_r0 is unsupported; radiation uses T_cmb and N_ur")
    return {
        "omega_b": c.omega_b,
        "omega_cdm": c.omega_cdm,
        "h": c.h,
        "A_s": c.A_s,
        "n_s": c.n_s,
        "tau_reio": c.tau_reio,
        "T_cmb": 2.7255,
        "N_ur": 3.044,
        "N_ncdm": 0,
        "output": "mPk,dTk",
        "format": "class",
        "headers": "yes",
        "write background": "yes",
        "write thermodynamics": "yes",
        "write parameters": "yes",
        "write warnings": "yes",
    }


def to_axiclass_dict(cosmology: EDECosmology) -> dict[str, Any]:
    """Use native axion shooting parameters without unrecognized aliases."""
    p = _common(cosmology)
    p["fourier_verbose"] = 1
    if cosmology.f_ede > 0:
        p.update(
            {
                "scf_potential": "axion",
                "n_axion": cosmology.potential_index,
                "log10_axion_ac": -math.log10(1 + cosmology.z_c),
                "fraction_axion_ac": cosmology.f_ede,
                "scf_parameters": f"{cosmology.theta_i:.17g}, 0.0",
                "scf_evolve_as_fluid": "no",
                "scf_evolve_like_axionCAMB": "no",
                "do_shooting": "yes",
                "do_shooting_scf": "yes",
                "scf_has_perturbations": "yes",
                "attractor_ic_scf": "no",
                "include_scf_in_delta_m": "no",
                "include_scf_in_delta_cb": "no",
            }
        )
    return p


def to_class_ede_dict(cosmology: EDECosmology) -> dict[str, Any]:
    """CLASS_EDE includes the cosmological constant inside its scalar potential."""
    p = _common(cosmology)
    p["fourier_verbose"] = 1
    if cosmology.f_ede > 0:
        p.update(
            {
                "Omega_Lambda": 0,
                "Omega_fld": 0,
                "Omega_scf": -1,
                "fEDE": cosmology.f_ede,
                "log10z_c": cosmology.log10_z_c,
                "thetai_scf": cosmology.theta_i,
                "n_scf": cosmology.potential_index,
                "CC_scf": 1,
                "scf_parameters": "1, 1, 1, 1, 1, 0.0",
                "scf_tuning_index": 3,
                "attractor_ic_scf": "no",
            }
        )
    return p


def _serialize(request: LinearTheoryRequest, p: dict[str, Any]) -> str:
    p.update(
        {
            "root": "output/run_",
            "z_pk": ", ".join(f"{z:.17g}" for z in request.redshifts),
            "P_k_max_h/Mpc": request.k_max_h_mpc * 1.05,
            "gauge": request.gauge,
        }
    )
    # Densify the native grid before guarded resampling onto the requested grid.
    p["k_scalar_k_per_decade_for_pk"] = 100
    return "\n".join(f"{key} = {value}" for key, value in p.items()) + "\n"


def build_axiclass_ini_content(request: LinearTheoryRequest) -> str:
    return _serialize(request, to_axiclass_dict(request.cosmology))


def build_class_ede_ini_content(request: LinearTheoryRequest) -> str:
    return _serialize(request, to_class_ede_dict(request.cosmology))
