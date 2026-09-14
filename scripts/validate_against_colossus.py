"""Physical LCDM calibration against Colossus and hmf, with retained evidence."""

from __future__ import annotations

import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from hmf.mass_function import fitting_functions

from dmra.constants import DEFAULT_DELTA_C
from dmra.cosmology.background import FlatLambdaCDM
from dmra.cosmology.boltzmann.manifest import compute_sha256
from dmra.cosmology.power import TabulatedPowerSpectrum
from dmra.cosmology.variance import VarianceCalculator
from dmra.statistics.hmf import HaloMassFunctionCalculator
from dmra.statistics.multiplicity import PressSchechter, ShethTormen, Tinker2008Delta200MeanZ0


def main() -> None:
    config_path = Path("configs/validation/hmf_reference.json")
    c = json.loads(config_path.read_text())
    out = Path("validation/hmf")
    out.mkdir(parents=True, exist_ok=True)
    reference = cosmology.setCosmology(
        c["name"],
        dict(
            flat=True,
            H0=100 * c["h"],
            Om0=c["omega_m0"],
            Ob0=c["omega_b0"],
            sigma8=c["sigma8"],
            ns=c["n_s"],
            interpolation=False,
            persistence="",
        ),
    )
    background = FlatLambdaCDM(c["omega_m0"], c["h"])
    mass = np.geomspace(c["mass_min"], c["mass_max"], c["mass_points"])
    radius = background.mass_to_radius(mass)
    k = np.geomspace(c["k_min"], c["k_max"], c["k_points"])
    pk_seed = reference.matterPowerSpectrum(k, model="eisenstein98")
    # Colossus consumes logged samples. Read its normalized samples back before
    # handing them to DMRA so both integrations see precisely the same P(k).
    seed_path = out / "colossus_seed_power.txt"
    np.savetxt(seed_path, np.column_stack([np.log10(k), np.log10(pk_seed)]))
    reference = cosmology.setCosmology(
        f"{c['name']}_tabulated",
        dict(
            flat=True,
            H0=100 * c["h"],
            Om0=c["omega_m0"],
            Ob0=c["omega_b0"],
            sigma8=c["sigma8"],
            ns=c["n_s"],
            interpolation=False,
            persistence="",
        ),
    )
    ps_args = {"path": str(seed_path)}
    pk = reference.matterPowerSpectrum(k, **ps_args)
    spectrum = TabulatedPowerSpectrum(k, pk)
    result = VarianceCalculator(spectrum).sigma_r(radius)
    their_sigma = reference.sigma(radius, ps_args=ps_args)
    sigma_error = np.abs(result.sigma / their_sigma - 1)
    np.savetxt(
        out / "physical_power.csv",
        np.column_stack([k, pk]),
        delimiter=",",
        header="k_h_Mpc,P_Mpc_h_cubed",
        comments="",
    )
    np.savetxt(
        out / "sigma_comparison.csv",
        np.column_stack(
            [mass, radius, result.sigma, their_sigma, sigma_error, result.max_edge_ratio]
        ),
        delimiter=",",
        header="mass_Msun_h,radius_Mpc_h,dmra_sigma,colossus_sigma,relative_error,edge_ratio",
        comments="",
    )
    comparisons = {}
    calc = HaloMassFunctionCalculator(rho_m0=background.rho_m0)
    for label, local, remote, hmf_type in [
        ("PS", PressSchechter(), "press74", fitting_functions.PS),
        ("ST", ShethTormen(), "sheth99", fitting_functions.ST),
        ("Tinker2008", Tinker2008Delta200MeanZ0(), "tinker08", fitting_functions.Tinker08),
    ]:
        f_local = local.evaluate(result.sigma)
        kwargs = {"deltac_args": {"corrections": False}} if label != "Tinker2008" else {}
        f_colossus = mass_function.massFunction(
            result.sigma,
            0,
            q_in="sigma",
            q_out="f",
            model=remote,
            mdef="200m" if label == "Tinker2008" else "fof",
            ps_args=ps_args,
            **kwargs,
        )
        hmf_error: float | None = None
        if label != "Tinker2008":
            f_hmf = hmf_type(
                nu2=(DEFAULT_DELTA_C / result.sigma) ** 2,
                m=mass,
                delta_c=DEFAULT_DELTA_C,
            ).fsigma
            hmf_error = float(np.max(np.abs(f_local / f_hmf - 1)))
        else:
            # hmf's Tinker class applies its own calibration interpolation.
            # Colossus is the independent Tinker check at explicitly 200m.
            f_hmf = np.full_like(f_local, np.nan)
        h = calc.evaluate(mass, result.sigma, local, jacobian=result.dln_sigma_inv_dln_m)
        # The independent abundance includes Colossus's own sigma and Jacobian.
        reference.interpolation = True
        h_colossus = mass_function.massFunction(
            mass,
            0,
            model=remote,
            q_out="dndlnM",
            mdef="200m" if label == "Tinker2008" else "fof",
            **kwargs,
        )
        reference.interpolation = False
        comparisons[label] = {
            "colossus_multiplicity_max_relative": float(np.max(np.abs(f_local / f_colossus - 1))),
            "hmf_multiplicity_max_relative": hmf_error,
            "colossus_abundance_max_relative": float(
                np.max(np.abs(h.dn_dln_mass / h_colossus - 1))
            ),
        }
        np.savetxt(
            out / f"{label}_comparison.csv",
            np.column_stack([mass, f_local, f_colossus, f_hmf, h.dn_dln_mass, h_colossus]),
            delimiter=",",
            header="mass,dmra_f,colossus_f,hmf_f,dmra_dndlnM,colossus_dndlnM",
            comments="",
        )
    convergence = []
    for stride in [16, 8, 4, 2, 1]:
        v = VarianceCalculator(TabulatedPowerSpectrum(k[::stride], pk[::stride])).sigma_r(radius)
        convergence.append(
            [
                len(k[::stride]),
                float(np.max(np.abs(v.sigma / result.sigma - 1))),
                float(np.max(np.abs(v.dln_sigma_inv_dln_m / result.dln_sigma_inv_dln_m - 1))),
            ]
        )
    np.savetxt(
        out / "grid_convergence.csv",
        convergence,
        delimiter=",",
        header="k_points,sigma_relative,jacobian_relative",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogx(mass, sigma_error)
    axes[0].set(
        xlabel="Mass [Msun/h]",
        ylabel="Relative sigma error",
        title="Colossus independent quadrature",
    )
    conv = np.array(convergence[:-1])
    axes[1].loglog(conv[:, 0], conv[:, 1], "-o", label="sigma")
    axes[1].loglog(conv[:, 0], conv[:, 2], "-o", label="Jacobian")
    axes[1].set(
        xlabel="k samples", ylabel="Error vs finest grid", title="Physical LCDM convergence"
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=180)
    plt.close(fig)
    passed = bool(
        np.max(sigma_error) < c["sigma_relative_tolerance"]
        and np.max(result.max_edge_ratio) < c["edge_ratio_tolerance"]
        and convergence[-2][1] < c["grid_sigma_tolerance"]
        and convergence[-2][2] < c["grid_jacobian_tolerance"]
        and all(
            all(
                value < c["hmf_relative_tolerance"]
                for value in metrics.values()
                if value is not None
            )
            for metrics in comparisons.values()
        )
    )
    report = {
        "passed": passed,
        "config": c,
        "config_sha256": compute_sha256(config_path.read_bytes()),
        "max_sigma_relative_error": float(np.max(sigma_error)),
        "max_edge_ratio": float(np.max(result.max_edge_ratio)),
        "comparisons": comparisons,
        "convergence": convergence,
        "packages": {p: version(p) for p in ["numpy", "scipy", "colossus", "hmf"]},
        "python": platform.python_version(),
        "conventions": "The local integrator is verified by grid convergence. Colossus documents its Eisenstein-Hu variance approximation as accurate to about 2%; its 2% comparison threshold tests cross-package compatibility rather than an identical quadrature. Collapse corrections are disabled; ST uses analytic A rather than rounded 0.3222. Colossus HMF uses its interpolation and critical-density constant.",
        "artifacts": {
            p.name: compute_sha256(p.read_bytes())
            for p in out.iterdir()
            if p.is_file() and p.name != "report.json"
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit("HMF calibration gate failed; inspect retained report")


if __name__ == "__main__":
    main()
