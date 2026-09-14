# Provenance, Software Architecture, and Cryptographic Tracking

This document outlines the intellectual heritage, software provenance, and cryptographic execution infrastructure of the Dark Matter Research Architecture (`DMRA`).

---

## 1. Intellectual Lineage & Theoretical Context

The theoretical foundations of `dmra` synthesise five decades of cosmological structure formation and perturbation theory:

### Excursion Set Theory & Multiplicity Formulations
The analytical mapping from Gaussian linear density perturbations to nonlinear collapsed objects originates in **Press & Schechter (1974)**, who formulated spherical collapse thresholding on smoothed density fields to obtain the universal multiplicity $f(\nu) = \sqrt{2/\pi}\nu e^{-\nu^2/2}$ (implemented in `dmra.statistics.multiplicity.PressSchechter`). **Bond, Cole, Efstathiou, & Kaiser (1991)** resolved the cloud-in-cloud ambiguity by casting excursion-set collapse as a Brownian random walk in linear variance $\sigma^2$ with an absorbing barrier at $\delta_c = 1.686$, establishing the theoretical foundation for unit mass-fraction normalization. Recognizing the inadequacy of spherical collapse against tidal shear, **Sheth & Tormen (1999, 2002)** introduced ellipsoidal collapse with a moving barrier calibrated to early $N$-body simulations; `dmra.statistics.multiplicity.ShethTormen` preserves their exact gamma-function normalization identity.

### Precision Non-Universality & Modern Descriptors
Precision $N$-body campaigns systematically demonstrated that the assumption of a universal multiplicity function $f(\sigma)$ breaks down at the $10\text{--}20\%$ level across redshift and cosmological background. **Tinker et al. (2008)** calibrated simulation fits across $z=0\text{--}2.5$ for spherical-overdensity definitions, proving that multiplicity parameters evolve systematically with expansion history (`dmra.statistics.multiplicity.Tinker2008Delta200MeanZ0`). To address this non-universality causally, **Ondaro-Mallea et al. (2022)** separated growth history $f(z)$ from instantaneous power spectrum shape, demonstrating that halo abundance retains physical memory of its dynamical past. Most recently, **Fiorilli et al. (2026)** (*Evolution Mapping III*) restored percent-level universality across diverse cosmological models by parameterizing multiplicity via recent growth history and the local linear spectral slope $n_{\rm eff}(M) \equiv d\ln P / d\ln k|_{k=1/R(M)}$. This modern parameterization serves as the primary benchmark against which any claimed EDE halo signature must be tested.

### Early Dark Energy Dynamics
The dynamical background engine relies on **Turner (1983)**, who proved that scalar fields oscillating in steep monomial potentials $V(\phi) \propto \phi^{2n}$ exhibit a cycle-averaged virial equation of state $\langle w_\phi \rangle = (n-1)/(n+1)$. For the canonical $n=3$ axion-like potential introduced by **Poulin et al. (2018, 2019)** to alleviate the Hubble tension ($H_0$), this yields $\langle w_\phi \rangle = 1/2$ and an asymptotic energy density dilution $\rho_\phi(a) \propto a^{-4.5}$, decaying faster than radiation and leaving late-time cosmological expansion unperturbed. This scalar field physics is directly parameterized in `dmra.cosmology.ede.model.EDECosmology`.

---

## 2. Solver Architecture: Protocol & Backends

The Boltzmann solver layer operates under a strict protocol-driven architecture:
- **`LinearTheoryProtocol` (`dmra.cosmology.boltzmann.protocol`):** Structural typing interface requiring any solver to accept a `LinearTheoryRequest` and return a validated `LinearTheoryResult`.
- **`AxiCLASSSolver` (`dmra.cosmology.ede.axiclass`):** Production C-backend interface executing Poulin et al.'s AxiCLASS Boltzmann code with full Klein--Gordon field integration.
- **`CLASS_EDESolver` (`dmra.cosmology.ede.class_ede`):** Alternative C-backend interface executing the CLASS_EDE implementation (Hill et al. 2020) for cross-solver validation.
- **Analytic development utility (`dmra.cosmology.ede.physics_engine`):** A deliberately simplified background and growth generator. Production adapters refuse analytic fallbacks; this utility is not a Boltzmann solver and cannot support scientific or cross-solver claims.

---

## 3. Cryptographic Execution Tracking & Manifests

Every solver execution writes a JSON `SolverManifest` alongside its raw output. It contains:
- `solver_name`: Canonical engine identifier.
- `solver_version`: Pinned git commit hash of the solver repository.
- `config_hash`: SHA-256 digest of the exact generated solver input file.
- `input_parameters`: Key-value map of physical cosmological inputs.
- `raw_units`: Dictionary mapping input and output variables to their native units prior to ingestion.
- `execution_time_seconds`: High-resolution wall-clock duration of the Boltzmann run.
- `stdout_summary` & `stderr_summary`: Captured diagnostic tails from standard streams.
- `git_commit`: Current repository commit hash under which the evaluation took place.
- `provenance`: executable SHA-256, pinned-source status and remote, build record, Python/platform/package versions, code and raw-output hashes, exact command, and run-directory path.

The retained release runs are indexed by `validation/linear/report.json`; each archive contains the generated parameters, background, thermodynamics, transfer, power, logs, and manifest. AxiCLASS and CLASS_EDE are two CLASS-family implementations, so agreement between them is cross-implementation evidence rather than independence from a shared numerical lineage.
