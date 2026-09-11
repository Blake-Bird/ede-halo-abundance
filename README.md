# DMRA — Halo Abundance in Early Dark Energy Cosmologies

[![CI](https://github.com/Blake-Bird/ede-halo-abundance/actions/workflows/ci.yml/badge.svg)](https://github.com/Blake-Bird/ede-halo-abundance/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dark-matter halos are the nonlinear structures that host galaxies and galaxy clusters. Their abundance as a function of mass—the halo mass function (HMF)—is one of the main bridges between a cosmological model and observable structure.


This repository addresses a foundational cosmological question:

> If an Early Dark Energy (EDE) episode changes how structure grows, when do standard halo-mass-function prescriptions stop being accurate, and what is the smallest physically meaningful correction needed to restore predictive accuracy?

The eventual program will compare EDE and matched control cosmologies across redshift with N-body simulations, then build a fast, validated HMF model. This repository does **not** yet claim an EDE result. It is the numerical foundation required to make that later comparison worth trusting.

## What is here now

An independent Python implementation of the calculation that sits underneath essentially every HMF prediction:

```text
linear matter power spectrum P(k)
        ↓ top-hat smoothing
mass variance sigma(M)
        ↓ collapse threshold
peak height nu
        ↓ multiplicity model + Jacobian
halo abundance dn / dln(M)
```

The implementation includes:

- explicit h-scaled units and one frozen peak-height convention;
- a numerically stable spherical top-hat window with analytical derivatives;
- guarded log-log interpolation of tabulated power spectra—no accidental extrapolation;
- log-wavenumber variance integration with exact analytical Jacobians and finite-support diagnostics;
- a high-accuracy flat-LambdaCDM growth reference;
- Press–Schechter, Sheth–Tormen, and calibrated Tinker (2008) `Delta=200m, z=0` reference fits;
- separate unit, numerical, and physics tests; and
- optional independent checks against Colossus.

The derivation, assumptions, and source trail are part of the project rather than an afterthought:

- [Theoretical Derivations & Foundations (PDF)](docs/theory/hmf_foundations.pdf) — 9-page formal derivation paper
- [Reading guide](docs/READING.md) — primary literature notes & code connections
- [Equation and software provenance](docs/SOURCES.md) — mathematical attribution
- [Notation and units](docs/NOTATION.md) — frozen conventions


## Run it

```bash
python -m pip install -e '.[dev]'
pytest -q
python scripts/validate_hmf_foundations.py
```

For the optional independent variance comparison:

```bash
python -m pip install -e '.[validation]'
python scripts/validate_against_colossus.py
```

Quality checks used in continuous integration:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```

## Repository guide

- `src/dmra/` — the scientific implementation.
- `tests/` — behavioral, numerical, and physics invariants.
- `scripts/` — reproducible validation and figure-generation entry points.
- `benchmarks/` — performance and precision measurements for the variance kernel.
- `docs/` — derivations, notation, reading notes, and provenance.

Generated figures, local environments, raw simulation outputs, checkpoints, and machine-specific files are intentionally excluded from Git.

## Numerical Scope & Methodology

The codebase emphasizes strict numerical correctness: Tinker coefficients are explicitly scoped to their calibration domain; finite k-domain boundaries produce diagnostics rather than silent power extrapolations; and external packages (such as Colossus) are used as independent validation benchmarks rather than core dependencies. Before this project can support an EDE fitting function, it still needs physical EDE linear theory, controlled N-body simulations, convergence campaigns, halo-finder cross-checks, and out-of-distribution validation.

