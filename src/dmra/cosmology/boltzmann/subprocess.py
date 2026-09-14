"""Sandboxed subprocess execution engine for external Boltzmann solvers."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class BoltzmannExecutionError(RuntimeError):
    """Raised when an external Boltzmann solver fails, crashes, or times out."""


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    """Artifacts and execution metrics captured from a sandboxed run."""

    exit_code: int
    stdout: str
    stderr: str
    execution_time_seconds: float
    scratch_dir: Path
    output_files: tuple[Path, ...]


class SandboxedExecutionEngine:
    """Manages isolated scratch directories, execution timeouts, and stream logging.

    Parameters
    ----------
    scratch_root:
        Base path under which unique run directories are allocated.
    timeout_seconds:
        Default wall-clock limit before terminating child processes.
    cleanup_on_success:
        If True, cleans up scratch directories on successful execution.
        Defaults to False so outputs remain available for downstream inspection.
    """

    def __init__(
        self,
        scratch_root: Path | str = "scratch",
        *,
        timeout_seconds: float = 60.0,
        cleanup_on_success: bool = False,
    ) -> None:
        self.scratch_root = Path(scratch_root).resolve()
        self.timeout_seconds = float(timeout_seconds)
        self.cleanup_on_success = cleanup_on_success

    def create_run_directory(self, prefix: str = "run_") -> Path:
        """Create a dedicated, isolated scratch workspace."""
        run_id = f"{prefix}{uuid.uuid4().hex[:12]}"
        run_dir = self.scratch_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def execute(
        self,
        command: Sequence[str | Path],
        *,
        cwd: Path | str | None = None,
        expected_outputs: Sequence[str | Path] = (),
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a Boltzmann command under strict sandbox and timeout guarantees.

        Parameters
        ----------
        command:
            Executable and arguments list.
        cwd:
            Working directory for execution. If None, allocates a new scratch dir.
        expected_outputs:
            Relative or absolute file paths that must exist and be non-empty upon completion.
        timeout:
            Override timeout in seconds.
        env:
            Custom environment variables dictionary.

        Returns
        -------
        ExecutionResult:
            Structured record of exit status, stream logs, timing, and output files.
        """
        run_timeout = self.timeout_seconds if timeout is None else float(timeout)
        run_dir = Path(cwd).resolve() if cwd is not None else self.create_run_directory()
        run_dir.mkdir(parents=True, exist_ok=True)

        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        cmd_strings = [str(c) for c in command]
        t_start = time.perf_counter()

        stdout_text = ""
        stderr_text = ""
        exit_code = -1

        try:
            process = subprocess.Popen(
                cmd_strings,
                cwd=run_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=exec_env,
            )
            stdout_text, stderr_text = process.communicate(timeout=run_timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired as exc:
            process.kill()
            stdout_text, stderr_text = process.communicate()
            t_elapsed = time.perf_counter() - t_start
            raise BoltzmannExecutionError(
                f"Boltzmann solver timed out after {t_elapsed:.2f}s "
                f"(limit: {run_timeout}s). Command: {' '.join(cmd_strings)}\n"
                f"Stderr tail: {stderr_text[-1000:]}"
            ) from exc
        except Exception as exc:
            t_elapsed = time.perf_counter() - t_start
            raise BoltzmannExecutionError(
                f"Failed to launch command {' '.join(cmd_strings)}: {exc}"
            ) from exc

        t_elapsed = time.perf_counter() - t_start

        # Persist stdout/stderr to disk in run directory
        (run_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
        (run_dir / "stderr.log").write_text(stderr_text, encoding="utf-8")

        if exit_code != 0:
            raise BoltzmannExecutionError(
                f"Boltzmann engine exited with non-zero status {exit_code}.\n"
                f"Command: {' '.join(cmd_strings)}\n"
                f"Stderr:\n{stderr_text[-2000:]}\n"
                f"Stdout:\n{stdout_text[-2000:]}"
            )

        # Verify expected outputs exist and are non-empty
        validated_outputs: list[Path] = []
        for expected in expected_outputs:
            out_path = Path(expected)
            if not out_path.is_absolute():
                out_path = run_dir / out_path

            if not out_path.exists():
                raise BoltzmannExecutionError(
                    f"Execution completed with code 0 but expected output '{out_path.name}' "
                    f"was not created in {run_dir}."
                )
            if out_path.stat().st_size == 0:
                raise BoltzmannExecutionError(
                    f"Output file '{out_path.name}' was created but is empty (0 bytes)."
                )
            validated_outputs.append(out_path)

        if self.cleanup_on_success and cwd is None:
            shutil.rmtree(run_dir, ignore_errors=True)

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            execution_time_seconds=t_elapsed,
            scratch_dir=run_dir,
            output_files=tuple(validated_outputs),
        )
