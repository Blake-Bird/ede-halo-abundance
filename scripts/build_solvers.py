"""Build the pinned source revisions and record the actual compiler invocations."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from dmra.cosmology.boltzmann.manifest import compute_sha256
from dmra.cosmology.ede.runner import git_value


def main() -> None:
    for label, checkout in [("axiclass", "AxiCLASS"), ("class_ede", "class_ede")]:
        pin = json.loads(Path(f"external/manifests/{label}.json").read_text())
        source = Path("external") / checkout
        if not source.exists():
            subprocess.run(["git", "clone", pin["repository"], str(source)], check=True)
            subprocess.run(
                ["git", "-C", str(source), "checkout", "--detach", pin["commit"]], check=True
            )
        if git_value(source, "rev-parse", "HEAD") != pin["commit"] or git_value(
            source, "status", "--porcelain"
        ):
            raise RuntimeError(f"{source} must be clean at pinned revision {pin['commit']}")
        cc = "clang" if platform.system() == "Darwin" else "gcc"
        cpp = "clang++" if platform.system() == "Darwin" else "g++"
        command = [
            "make",
            "-B",
            "-j2",
            "class",
            f"CC={cc}",
            f"CPP={cpp} -std=c++11",
            "OMPFLAG=-pthread",
            "OPTFLAG=-O3",
        ]
        out = Path("validation/builds")
        out.mkdir(parents=True, exist_ok=True)
        log = out / f"{label}.log"
        with log.open("w") as stream:
            subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)
        record = {
            "commit": pin["commit"],
            "command": command,
            "platform": platform.platform(),
            "compiler": subprocess.check_output([cc, "--version"], text=True),
            "executable_sha256": compute_sha256((source / "class").read_bytes()),
            "build_log": str(log),
            "build_log_sha256": compute_sha256(log.read_bytes()),
        }
        (out / f"{label}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Built {label} at {pin['commit']}", flush=True)


if __name__ == "__main__":
    main()
