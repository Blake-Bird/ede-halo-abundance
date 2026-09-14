"""Shared execution and provenance for the two pinned CLASS forks."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from dmra.cosmology.boltzmann.manifest import SolverManifest, compute_sha256
from dmra.cosmology.boltzmann.protocol import LinearTheoryRequest
from dmra.cosmology.boltzmann.result import LinearTheoryResult, MatterPowerGrid
from dmra.cosmology.boltzmann.subprocess import SandboxedExecutionEngine
from dmra.cosmology.ede.parameter_map import build_axiclass_ini_content, build_class_ede_ini_content
from dmra.cosmology.ede.parser import (
    build_background_history_from_table,
    extract_solver_sigma8,
    parse_class_power_spectrum,
    parse_class_table,
)
from dmra.cosmology.power import TabulatedPowerSpectrum


def git_value(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


class CLASSRunner:
    """Execute a real solver and retain all raw files plus a JSON manifest.

    Missing binaries fail explicitly. The former approximate fallback has been
    removed because it could return data labeled with a real solver's name.
    """

    name = ""
    checkout = ""
    command_name = ""

    def __init__(
        self,
        executable_path: Path | str | None = None,
        scratch_root: Path | str | None = None,
        timeout_seconds: float = 120,
        allow_mock: bool = False,
    ) -> None:
        if allow_mock:
            raise ValueError("analytic fallback is no longer supported by scientific adapters")
        if executable_path is None:
            candidates = [
                Path("external") / self.checkout / "class",
                Path("external/bin") / self.command_name,
            ]
            found = shutil.which(self.command_name)
            if found:
                candidates.append(Path(found))
            executable = next(
                (p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None
            )
        else:
            executable = Path(executable_path)
        if executable is None or not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"{self.name} executable missing; see external/manifests")
        self._executable = executable.resolve()
        self.engine = SandboxedExecutionEngine(
            scratch_root=scratch_root or f"scratch/{self.checkout}", timeout_seconds=timeout_seconds
        )

    @property
    def version(self) -> str:
        return git_value(self._executable.parent, "rev-parse", "HEAD")

    def run(self, request: LinearTheoryRequest) -> LinearTheoryResult:
        run_dir = self.engine.create_run_directory()
        out = run_dir / "output"
        out.mkdir()
        ini = (
            build_axiclass_ini_content(request)
            if self.name == "AxiCLASS"
            else build_class_ede_ini_content(request)
        )
        (run_dir / "input.ini").write_text(ini, encoding="utf-8")
        execution = self.engine.execute([self._executable, "input.ini"], cwd=run_dir)

        def unique(pattern: str) -> Path:
            files = list(out.glob(pattern))
            if len(files) != 1 or files[0].stat().st_size == 0:
                raise ValueError(f"expected one nonempty {pattern}, found {len(files)}")
            return files[0]

        bg_file = unique("*_background.dat")
        unique("*_thermodynamics.dat")
        unique("*_parameters.ini")
        columns, data = parse_class_table(bg_file)
        bg = build_background_history_from_table(data, columns, request.cosmology.h)
        bg.validate_flatness()
        k = np.geomspace(request.k_min_h_mpc, request.k_max_h_mpc, request.num_k_points)
        powers = []
        for i, _z in enumerate(request.redshifts):
            tag = f"*_z{i + 1}_" if len(request.redshifts) > 1 else "*_"
            pk_file = unique(tag + "pk.dat")
            unique(tag + "tk.dat")
            k_native, pk_native = parse_class_power_spectrum(pk_file, request.cosmology.h)
            powers.append(TabulatedPowerSpectrum(k_native, pk_native).evaluate(k))
        unused_files = list(out.glob("*unused_parameters*"))
        for unused in unused_files:
            lines = [
                line
                for line in unused.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if lines:
                raise ValueError(f"solver did not recognize input parameters: {lines}")
        inputs: dict[str, Any] = asdict(request.cosmology)
        s8 = extract_solver_sigma8(execution.stdout)
        if s8 is not None:
            inputs["solver_sigma8"] = s8
        root = Path.cwd()
        source_files = [*sorted(Path("src").rglob("*.py")), Path("pyproject.toml")]
        source_hashes = {str(p): compute_sha256(p.read_bytes()) for p in source_files}
        output_hashes = {
            str(p.relative_to(run_dir)): compute_sha256(p.read_bytes())
            for p in sorted(run_dir.rglob("*"))
            if p.is_file()
        }
        build_path = Path("validation/builds") / (
            "axiclass.json" if self.name == "AxiCLASS" else "class_ede.json"
        )
        build_record = json.loads(build_path.read_text()) if build_path.exists() else None
        if build_record and build_record["executable_sha256"] != compute_sha256(
            self._executable.read_bytes()
        ):
            raise ValueError("executable no longer matches its recorded build")
        provenance = {
            "request": asdict(request),
            "executable_sha256": compute_sha256(self._executable.read_bytes()),
            "solver_source_status": git_value(self._executable.parent, "status", "--porcelain"),
            "solver_repository": git_value(self._executable.parent, "remote", "get-url", "origin"),
            "build_environment": build_record or "unrecorded build",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: version(p) for p in ["numpy", "scipy", "PyYAML"]},
            "source_hashes": source_hashes,
            "output_hashes": output_hashes,
            "run_directory": str(run_dir),
            "command": [str(self._executable), "input.ini"],
        }
        manifest = SolverManifest(
            solver_name=self.name,
            solver_version=self.version,
            config_hash=compute_sha256(ini),
            input_parameters=inputs,
            raw_units={"k": "h/Mpc", "P": "(Mpc/h)^3", "H": "1/Mpc"},
            execution_time_seconds=execution.execution_time_seconds,
            stdout_summary=execution.stdout[-4000:],
            stderr_summary=execution.stderr[-4000:],
            git_commit=git_value(root, "rev-parse", "HEAD"),
            provenance=provenance,
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return LinearTheoryResult(
            cosmology=request.cosmology,
            background=bg,
            power_grid=MatterPowerGrid(k, np.asarray(request.redshifts), np.asarray(powers)),
            manifest=manifest,
        )
