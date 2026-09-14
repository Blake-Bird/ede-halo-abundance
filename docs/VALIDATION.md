# Validation Protocol and Quantitative Physical Invariants

This document specifies validation criteria and required evidence for the Dark Matter Research Architecture (`DMRA`). Numerical values should be reported only with the retained run artifacts and revision that produced them.

---

## 1. Physical & Numerical Invariants

Analytic kernels are covered by automated tests. Solver-dependent checks require a retained external-solver run and must not be inferred from the unit-test suite.

| Invariant / Benchmark | Verification Target | Physical Tolerance | Achieved Precision | Status |
| :--- | :--- | :--- | :--- | :--- |
| **$\Lambda\text{CDM}$ zero limit** | AxiCLASS versus CLASS_EDE background | $\max |\Delta H/H| < 2\times10^{-3}$ | 0 on the retained grid | Passing |
| **Solver shooting** | Realized peak matches requested $f_{\rm EDE}$ | $f$: $<0.5\%$; $z_c$: $<2\%$ | $1.43\times10^{-7}$; $2.46\times10^{-4}$ | Passing |
| **$\sigma_8$ quadrature** | Simpson integral agrees with solver report | Relative error $<10^{-3}$ | $6.03\times10^{-5}$ maximum | Passing |
| **Cross-implementation consistency** | Matched AxiCLASS/CLASS_EDE observables | Declared before execution | $H$: $1.23\times10^{-5}$; $P$: $2.10\times10^{-5}$ | Passing |
| **Analytical Jacobian** | Power-law spectrum yields $(n+3)/6$ | Relative error $< 10^{-10}$ | Automated test | Passing locally |
| **Mass-radius round trip** | $M \to R(M) \to M$ | Machine precision | Automated test | Passing locally |
| **Einstein--de Sitter mode** | $D(a)=a$ and $f(a)=1$ | Relative error $< 10^{-6}$ | Automated test | Passing locally |
| **Multiplicity normalization** | $\int_0^\infty f(\nu)\dd\ln\nu = 1$ | Relative error $< 10^{-5}$ | Automated test | Passing locally |
| **Colossus comparison** | Eisenstein--Hu compatibility in $\sigma(M)$ | Relative error $<2\%$ | $1.20\%$ maximum | Passing within reference approximation scope |

---

## 2. Independent Reference Validations

### (i) Continuous $\Lambda\mathrm{CDM}$ Zero-Limit
To verify that the Early Dark Energy solver continuous limit does not introduce parasitic background energy or numerical offsets, we evaluate an EDE model with $f_{\rm EDE} = 0.0$ against an analytical flat $\Lambda\mathrm{CDM}$ background across $z \in [0, 10]$:
The retained zero-limit result is in `validation/linear/report.json`, with raw solver archives and manifests in `validation/linear/`.

### (ii) Simpson $\sigma_8$ Quadrature Verification
The linear matter power spectrum $P(k, z=0)$ from the Boltzmann solver is ingested into the DMRA `VarianceCalculator`. We independently integrate $\sigma_8$ using logarithmic Simpson quadrature:
$$\sigma_8^2 = \int_{-\infty}^\infty \frac{k^3 P(k, z=0)}{2\pi^2} W^2(k \cdot 8\,h^{-1}\mathrm{Mpc}) \dd\ln k.$$
Comparison against the external solver's reported value is a required integration check. The independently evaluated integral and solver report use the total-matter convention. The retained four-model gate has a maximum relative discrepancy of $6.03\times10^{-5}$, below its predeclared $10^{-3}$ tolerance.

### (iii) High-Redshift Rare-Halo Variance Enhancement
Named parameter sets can be evaluated at $z=5$, but an agreement claim requires a published target observable, matched parameter definitions, and a retained comparison table. The workflow in `scripts/reproduce_published_models.py` is therefore an exploratory comparison, not literature-replication validation.

### (iv) Colossus Benchmark Parity
With optional validation dependencies installed (`pip install -e '.[validation,plots]'`), `scripts/validate_against_colossus.py` evaluates a frozen Planck-like flat LCDM configuration. The retained run is `validation/hmf/report.json` and its CSV tables and figure are alongside it. The local quadrature converges to $5.54\times10^{-10}$ in $\sigma$ and $4.90\times10^{-7}$ in the Jacobian on a 4,097-point grid relative to the 65,537-point reference. Colossus's Eisenstein--Hu variance approximation differs by at most 1.20%; this is assessed against a 2% compatibility threshold, not represented as an identical-integrand comparison.

---

## 3. Regression Testing Suite

The project enforces three levels of automated continuous integration:
1. **Unit Tests (`tests/unit/`):** Test individual numerical kernels, coordinate transforms, window functions, and Taylor expansion thresholds.
2. **Numerical Tests (`tests/numerical/`):** Test convergence under grid refinement, analytical Jacobian recovery, and growth factor ODE accuracy.
3. **Physics Invariant Tests (`tests/physics/`):** Test physical limits, mass fraction normalizations, solver round-trips, and literature replication.
4. **External-solver tests (`tests/integration/` and marked physics tests):** Test end-to-end Boltzmann execution and manifest generation against pinned local binaries. They are marked `external_solver` and are deliberately excluded from clean hosted CI; `scripts/validate_linear_solvers.py` is the retained release gate.
