# Dark Matter Research Architecture (DMRA)
### Precision Early Dark Energy Linear Theory & Dark Matter Halo Abundance

[![CI](https://github.com/Blake-Bird/ede-halo-abundance/actions/workflows/ci.yml/badge.svg)](https://github.com/Blake-Bird/ede-halo-abundance/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Dark-matter halos are the nonlinear gravitational structures that host galaxies and clusters. Their abundance as a function of mass—the **halo mass function (HMF)**—serves as a primary observational bridge between fundamental early-universe cosmology and observable cosmic structure.

This research codebase addresses a core question in modern cosmology:
> **If an Early Dark Energy (EDE) episode modifies the expansion rate and growth history prior to recombination, does the resulting halo abundance retain physical memory of this non-standard growth history, or is it completely determined by the instantaneous linear matter power spectrum and modern spectral-shape descriptors?**

The repository provides an auditable computational pipeline from external linear-theory solvers to halo-scale mass-function calculations. The nonlinear, simulation-calibrated EDE question remains a research target rather than a claim of completion.

---

## The End-to-End Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             SCALAR FIELD DYNAMICS                                │
│       Lagrangian Action  ──►  Klein–Gordon ODE  ──►  Virial Dilution ρ_ϕ ∝ a⁻⁴·⁵  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             BACKGROUND EXPANSION                                 │
│        H(z),  Ω_ϕ(z),  Sound Horizon r_s(z_*)  ──►  CMB Acoustic Scale θ_*        │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        LINEAR PERTURBATIONS & BOLTZMANN                          │
│     Transfer Functions T_m(k, z)  ──►  Matter Power Grid P_m(k, z)  (Synchronous)│
│     Jeans Scale k_J(a)  ──►  Subhorizon Newtonian Gravitational Evolution        │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       LINEAR GROWTH & SCALE DEPENDENCE                           │
│        Heath (1977) Growth ODE  ──►  D_eff(k, z),  f(a) ≡ d ln D / d ln a         │
│        Scale-Dependence Diagnostic |ϵ_D(k, z)| < 0.3% on Halo Scales             │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         HALO VARIANCE & MASS MAPPING                             │
│       Top-Hat Window W(kR)  ──►  Logarithmic Variance Quadrature σ²(M, z)        │
│       Exact Analytical Jacobian Integration: J(M) ≡ d ln σ⁻¹ / d ln M           │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      HALO ABUNDANCE & NON-UNIVERSALITY                           │
│    Peak Height ν ≡ δ_c / σ  ──►  Multiplicity f(ν) [PS, ST, Tinker 2008]         │
│    Evolution Mapping III Baseline [Growth History + Local Spectral Slope n_eff]  │
│    Master Mass Function: dn / d ln M = (ρ̄_m / M) · f(ν) · J(M)                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Research Publications & Formal Documentation

- **[Master Research Paper (PDF)](papers/ede_halo_abundance.pdf)**: *From Early Dark Energy Scalar Field Dynamics to Dark Matter Halo Abundance: Theoretical Foundations, Linear Growth, and Non-Universal Mass Function Foundations* (19-page formal paper with 20 exact mathematical derivations).
- **[Theoretical Foundations (`docs/THEORY.md`)](docs/THEORY.md)**: Mathematical derivations of scalar dynamics, sound-horizon physics, linear perturbations, top-hat filtering, and rare-peak exponential sensitivity ($\partial\ln f_{\rm PS}/\partial\ln\sigma \approx \nu^2$).
- **[Conventions & Multiplicity Mapping (`docs/CONVENTIONS.md`)](docs/CONVENTIONS.md)**: Frozen units ($h$-scaled vs physical), parameter conventions, and cross-literature multiplicity mappings.
- **[Validation Protocol (`docs/VALIDATION.md`)](docs/VALIDATION.md)**: Numerical invariants, frozen external comparisons, and the scope of each release claim.
- **[Validation Review (`docs/reports/2026-09-13_foundation-validation.md`)](docs/reports/2026-09-13_foundation-validation.md)**: Retained HMF and linear-theory evidence, results, and explicit limits of interpretation.
- **[Provenance & Architecture (`docs/PROVENANCE.md`)](docs/PROVENANCE.md)**: Intellectual lineage, Boltzmann solver protocols (AxiCLASS and CLASS_EDE), and cryptographic execution tracking.
- **[Physical Boundaries (`docs/LIMITATIONS.md`)](docs/LIMITATIONS.md)**: Scope of applicability, linear regime assumptions, and simulation roadmap.

---

## Verification Status

Analytic kernels are covered by automated tests. Solver-dependent quantities require a retained run with the external binary, version, configuration, and outputs; they are not established by the unit-test suite alone.

| Invariant / Benchmark | Verification Target | Physical Tolerance | Achieved Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **$\Lambda\text{CDM}$ zero limit** | AxiCLASS versus CLASS_EDE | $\max |\Delta H/H| < 2\times10^{-3}$ | 0 (matched output) | Passing |
| **Solver shooting** | Realized peak matches requested $f_{\rm EDE}$ | $f$: $<0.5\%$; $z_c$: $<2\%$ | $1.43\times10^{-7}$; $2.46\times10^{-4}$ | Passing |
| **$\sigma_8$ quadrature** | DMRA Simpson integral versus solver report | Relative error $<10^{-3}$ | $6.03\times10^{-5}$ maximum | Passing |
| **Cross-implementation comparison** | Matched AxiCLASS/CLASS_EDE observables | Declared in `linear_gate.json` | $H$: $1.23\times10^{-5}$; $P$: $2.10\times10^{-5}$ | Passing |
| **Analytical Jacobian** | Power-law spectrum yields $(n+3)/6$ | Relative error $< 10^{-10}$ | Automated test | Passing locally |
| **Mass-radius round trip** | $M \to R(M) \to M$ | Machine precision | Automated test | Passing locally |
| **Einstein--de Sitter mode** | $D(a)=a$ and $f(a)=1$ | Relative error $< 10^{-6}$ | Automated test | Passing locally |
| **Multiplicity normalization** | $\int_0^\infty f(\nu)\dd\ln\nu = 1$ | Relative error $< 10^{-5}$ | Automated test | Passing locally |
| **Colossus comparison** | Independent Eisenstein--Hu compatibility check | $<2\%$ in $\sigma(M)$ | $1.20\%$ | Passing within Colossus approximation scope |

---

## Repository Layout

```text
ede-halo-abundance/
├── papers/                 # Publication manuscripts and BibTeX databases
│   ├── ede_halo_abundance.tex   # Comprehensive LaTeX manuscript
│   ├── ede_halo_abundance.pdf   # Compiled publication-grade PDF
│   └── references.bib           # Consolidated BibTeX reference library
├── docs/                   # Authoritative technical documentation
│   ├── THEORY.md                # Scalar field dynamics, perturbations & HMF
│   ├── CONVENTIONS.md           # Units, parameters, and multiplicity definitions
│   ├── VALIDATION.md            # Benchmark verification and tolerance standards
│   ├── PROVENANCE.md            # Literature lineage and software architecture
│   └── LIMITATIONS.md           # Modeling boundaries and simulation roadmap
├── src/dmra/               # Core scientific Python package
│   ├── cosmology/
│   │   ├── background.py        # Flat FLRW kinematics and mass-radius mappings
│   │   ├── growth.py            # Heath (1977) linear growth ODE solver
│   │   ├── power.py             # Guarded log-log matter power spectrum
│   │   ├── variance.py          # Top-hat smoothed variance & analytical Jacobians
│   │   ├── window.py            # Stable spherical top-hat filter & derivatives
│   │   ├── boltzmann/           # Protocol layer, manifests, and UnitGatekeeper
│   │   └── ede/                 # AxiCLASS/CLASS_EDE solvers & physics engine
│   └── statistics/
│       ├── hmf.py               # Master differential halo mass function calculator
│       ├── multiplicity.py      # Press-Schechter, Sheth-Tormen, Tinker multiplicities
│       └── peak_height.py       # Linear peak-height evaluation
├── configs/cosmology/      # Immutable cosmological parameter configurations
├── scripts/                # Reproducible validation and figure entry points
│   ├── audit_parameter_mapping.py      # Parameter shooting and zero-limit audit
│   ├── freeze_linear_dataset.py        # Parquet data freezing with SHA-256 digests
│   ├── generate_linear_figures.py      # Generates EDE linear theory figures
│   ├── generate_hmf_figures.py         # Generates HMF foundation figures
│   ├── reproduce_published_models.py   # Replicates Klypin et al. (2021) benchmarks
│   ├── validate_growth.py              # Audits growth scale-dependence
│   ├── validate_sigma8.py              # Bare Simpson sigma_8 quadrature validation
│   └── validate_hmf_foundations.py     # HMF mathematical invariant validation
├── notebooks/              # Interactive Jupyter research notebooks
│   └── ede_linear_theory.ipynb         # End-to-end tutorial & analysis notebook
└── tests/                  # Automated verification test suite
    ├── unit/                           # Kernel, coordinate, and unit tests
    ├── numerical/                      # Quadrature convergence & derivative tests
    ├── physics/                        # Physical invariants & literature replication
    └── integration/                    # Boltzmann solver end-to-end integration tests
```

---

## Quickstart & Installation

Clone the repository and install in editable development mode:

```bash
git clone https://github.com/Blake-Bird/ede-halo-abundance.git
cd ede-halo-abundance
python -m pip install -e '.[dev]'
```

### Run Automated Tests
```bash
pytest -q
```

### Static Quality Verification
```bash
ruff check .
ruff format --check .
mypy src
```

### Run Validation Pipelines
```bash
# Verify HMF mathematical foundations
python scripts/validate_hmf_foundations.py

# Build pinned external solver revisions and record compiler/binary hashes
python scripts/build_solvers.py

# Run the predeclared AxiCLASS/CLASS_EDE comparison and retain raw outputs
python scripts/validate_linear_solvers.py
```

### Optional External Comparison (Colossus)
```bash
python -m pip install -e '.[validation,plots]'
python scripts/validate_against_colossus.py
```

---

## License & Citation

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

If you utilize the theoretical formulations, numerical algorithms, or code in this repository, please cite:
```bibtex
@article{Bird2026EDE,
  author = {Bird, Blake},
  title = {From Early Dark Energy Scalar Field Dynamics to Dark Matter Halo Abundance: Theoretical Foundations, Linear Growth, and Non-Universal Mass Function Foundations},
  journal = {Research Notes},
  year = {2026}
}
```
