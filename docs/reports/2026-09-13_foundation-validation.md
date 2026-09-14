# Foundation and Linear-Theory Validation Review

## Scientific objective

Establish an auditable path from a linear matter spectrum through top-hat variance, peak height, multiplicity, and differential halo abundance; then demonstrate that the EDE linear-theory interface executes two pinned CLASS-family implementations with explicit conventions and retained evidence.

## Decision rule

The HMF release gate requires a frozen flat-LCDM configuration, a converged local calculation, and an external-reference comparison with a declared tolerance. The linear gate is defined in `configs/validation/linear_gate.json` before execution. It evaluates four named configurations, $0\leq z\leq30$, and $10^{-4}\leq k/(h\,\mathrm{Mpc}^{-1})\leq100$; its tolerances are $2\times10^{-3}$ in $H$, $10^{-2}$ in $P$, $5\times10^{-3}$ in $\sigma$ and effective growth, $10^{-3}$ in $\sigma_8$, $5\times10^{-3}$ in realized EDE fraction, and $2\times10^{-2}$ in realized transition redshift.

## Evidence retained

- HMF configuration and report: `configs/validation/hmf_reference.json`, `validation/hmf/report.json`.
- HMF tables and convergence figure: `validation/hmf/`.
- Solver gate and report: `configs/validation/linear_gate.json`, `validation/linear/report.json`.
- Raw solver outputs, input files, logs, thermodynamics, transfer functions, spectra, hashes, and manifests: `validation/linear/*.tar.gz` and `validation/linear/*_manifest.json`.
- Frozen solver source revisions: AxiCLASS `ae9609e1f96ddebbb6cc8a46de94512514c68781`; CLASS_EDE `5a131c91d657dd9a7c6364cc45b038710f8d0d97`.

## Results

The HMF gate passes. Refining the log-$k$ grid from 4,097 to 65,537 points changes $\sigma(M)$ by at most $5.54\times10^{-10}$ and the analytical Jacobian by $4.90\times10^{-7}$. The largest finite-support edge ratio is $1.50\times10^{-11}$. With the same Planck-like LCDM parameters and Eisenstein--Hu reference family, the maximum Colossus $\sigma(M)$ discrepancy is 1.20%, within the predeclared 2% compatibility threshold. Press--Schechter multiplicity matches exactly; the maximum retained abundance differences from Colossus are 0.311% (PS), 0.288% (ST), and 0.288% (Tinker 2008).

The linear gate passes all four configurations. Across the retained model set, the largest AxiCLASS/CLASS_EDE differences are $1.23\times10^{-5}$ in $H$, $2.10\times10^{-5}$ in $P$, $1.31\times10^{-6}$ in $\sigma(M)$, and $1.12\times10^{-5}$ in effective growth. The largest independent $\sigma_8$ discrepancy relative to a solver-reported value is $6.03\times10^{-5}$. For nonzero EDE configurations, the largest realized shooting errors are $1.43\times10^{-7}$ in fraction and $2.46\times10^{-4}$ in transition redshift.

## What these results establish

The numerical HMF implementation is converged for the declared configuration and mass range, and its Press--Schechter, Sheth--Tormen, and scoped Tinker transformations are externally compatible at the stated level. The EDE wrapper writes and retains real solver products, rejects missing or unrecognized solver parameters, records executable/source/output hashes, and maps those outputs into $H(a)$, effective growth, $P(k,z)$, $\sigma(M,z)$, and $\nu(M,z)$. The two pinned implementations agree far inside the declared gate.

## What these results do not establish

This is not an EDE nonlinear-halo result, a simulation calibration, a literature replication, or evidence of a new physical effect. Colossus's Eisenstein--Hu path is an external compatibility check rather than a bitwise identical quadrature implementation. AxiCLASS and CLASS_EDE share CLASS ancestry, so their agreement is cross-implementation validation, not complete algorithmic independence. No claim about HMF universality, EDE residual halo memory, or publishable novelty follows from this gate alone.

## Reproduction

Install the validation and plotting extras, build the exact external revisions with `python scripts/build_solvers.py`, then run `python scripts/validate_against_colossus.py` and `python scripts/validate_linear_solvers.py`. The JSON reports include configuration hashes and output hashes; raw archives make the displayed metrics independently auditable.

## Remaining uncertainty

The next scientifically meaningful step is a matched cosmology design followed by a controlled nonlinear simulation error budget. That stage needs an independent code family or a reference likelihood/transfer implementation beyond the shared CLASS lineage, multiple seeds, halo-definition matching, and a preregistered systematic-error threshold.
