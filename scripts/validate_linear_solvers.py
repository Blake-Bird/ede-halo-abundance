"""Execute predeclared linear-theory comparisons and retain every solver run."""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dmra.cosmology.boltzmann.manifest import compute_sha256
from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.ede.axiclass import AxiCLASSSolver
from dmra.cosmology.ede.class_ede import CLASSEDESolver
from dmra.cosmology.ede.descriptors import HaloScaleDescriptors
from dmra.cosmology.ede.growth import GrowthDiagnostics
from dmra.cosmology.ede.history import EDEHistory
from dmra.cosmology.ede.model import EDECosmology
from dmra.cosmology.variance import VarianceCalculator


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a / b - 1)))


def main() -> None:
    gate_path = Path("configs/validation/linear_gate.json")
    gate = json.loads(gate_path.read_text())
    out = Path("validation/linear")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in gate["models"]:
        values = yaml.safe_load(Path(f"configs/cosmology/{model}.yaml").read_text())
        values.pop("model_name")
        cosmology = EDECosmology(**values)
        request = LinearTheoryRequest(
            cosmology,
            redshifts=tuple(gate["redshifts"]),
            k_min_h_mpc=gate["k_min"],
            k_max_h_mpc=gate["k_max"],
            num_k_points=gate["k_points"],
        )
        results = []
        shooting = []
        sigma8_errors = []
        for name, adapter in [("axiclass", AxiCLASSSolver), ("class_ede", CLASSEDESolver)]:
            print(f"Running {model} with {name}", flush=True)
            result = adapter(timeout_seconds=240).run(request)
            results.append(result)
            source = Path(result.manifest.provenance["run_directory"])
            archive = out / f"{model}_{name}.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(source, arcname=f"{model}_{name}")
            shutil.copy2(source / "manifest.json", out / f"{model}_{name}_manifest.json")
            history = EDEHistory.from_background(result.background, cosmology.f_ede, cosmology.z_c)
            shooting.append(
                {
                    "solver": name,
                    "f_ede": history.f_ede_realized,
                    "z_c": history.z_c_realized,
                    "fraction_error": history.f_ede_shooting_error,
                    "redshift_error": history.z_c_shooting_error,
                }
            )
            actual = result.manifest.input_parameters.get("solver_sigma8")
            if actual is None:
                raise RuntimeError(f"{name} did not report sigma8; no self-comparison fallback")
            sigma8 = float(VarianceCalculator(result.power_grid.at_redshift(0)).sigma_r(8).sigma[0])
            sigma8_errors.append(abs(sigma8 / actual - 1))
            growth = GrowthDiagnostics.from_power_grid(result.power_grid)
            a, f = growth.linear_growth_rate()
            np.savetxt(
                out / f"{model}_{name}_growth.csv",
                np.column_stack([a, f]),
                delimiter=",",
                header="a,f_from_power",
                comments="",
            )
        a, b = results
        z = np.geomspace(1, 100001, 400) - 1
        k = np.geomspace(gate["compare_k_min"], gate["compare_k_max"], 100)
        masses = np.geomspace(1e9, 1e15, 80)
        power_errors = []
        sigma_errors = []
        growth_errors = []
        for redshift in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
            power_errors.append(
                relative(
                    a.power_grid.at_redshift(redshift).evaluate(k),
                    b.power_grid.at_redshift(redshift).evaluate(k),
                )
            )
            sa = HaloScaleDescriptors(a.power_grid, cosmology).evaluate(masses, redshift).sigma
            sb = HaloScaleDescriptors(b.power_grid, cosmology).evaluate(masses, redshift).sigma
            sigma_errors.append(relative(sa, sb))
            da = np.sqrt(
                a.power_grid.at_redshift(redshift).evaluate(k)
                / a.power_grid.at_redshift(0).evaluate(k)
            )
            db = np.sqrt(
                b.power_grid.at_redshift(redshift).evaluate(k)
                / b.power_grid.at_redshift(0).evaluate(k)
            )
            growth_errors.append(relative(da, db))
        row = {
            "model": model,
            "H_relative": relative(a.background.hubble(z), b.background.hubble(z)),
            "P_relative": max(power_errors),
            "sigma_relative": max(sigma_errors),
            "D_relative": max(growth_errors),
            "sigma8_relative": max(sigma8_errors),
            "shooting": shooting,
        }
        row["passed"] = bool(
            row["H_relative"] < gate["hubble_relative_tolerance"]
            and row["P_relative"] < gate["power_relative_tolerance"]
            and row["sigma_relative"] < gate["sigma_relative_tolerance"]
            and row["D_relative"] < gate["growth_relative_tolerance"]
            and row["sigma8_relative"] < gate["sigma8_relative_tolerance"]
            and all(
                h["fraction_error"] < gate["shooting_fraction_relative_tolerance"]
                and h["redshift_error"] < gate["shooting_redshift_relative_tolerance"]
                for h in shooting
            )
        )
        rows.append(row)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].semilogx(1 + z, a.background.hubble(z) / b.background.hubble(z) - 1)
        axes[0].set(xlabel="1+z", ylabel="H_AxiCLASS/H_CLASS_EDE - 1")
        for redshift in [0, 5]:
            axes[1].semilogx(
                k,
                a.power_grid.at_redshift(redshift).evaluate(k)
                / b.power_grid.at_redshift(redshift).evaluate(k)
                - 1,
                label=f"z={redshift}",
            )
        axes[1].set(xlabel="k [h/Mpc]", ylabel="P_AxiCLASS/P_CLASS_EDE - 1")
        axes[1].legend()
        fig.suptitle(model)
        fig.tight_layout()
        fig.savefig(out / f"{model}_comparison.png", dpi=160)
        plt.close(fig)
        report = {
            "passed": all(r["passed"] for r in rows),
            "complete": len(rows) == len(gate["models"]),
            "gate": gate,
            "gate_sha256": compute_sha256(gate_path.read_bytes()),
            "models": rows,
        }
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    if not all(r["passed"] for r in rows):
        raise SystemExit(
            "Linear gate failed; retained evidence must be reviewed without changing tolerances"
        )


if __name__ == "__main__":
    main()
