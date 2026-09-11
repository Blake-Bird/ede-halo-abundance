# Provenance: equations, numerics, and validation

The code in `src/dmra/` is original project code. It is not copied from CLASS, Colossus, `hmf`, SciPy, or another HMF repository. This page records where the scientific expressions and numerical choices come from.

| Component | Primary source | What enters this repository |
|---|---|---|
| Press–Schechter | Press & Schechter (1974), ApJ 187, 425; DOI [10.1086/152650](https://doi.org/10.1086/152650) | thresholded-Gaussian mass-accounting framework |
| Excursion set | Bond et al. (1991), ApJ 379, 440; DOI [10.1086/170520](https://doi.org/10.1086/170520) | first-crossing interpretation of cloud-in-cloud |
| Sheth–Tormen | Sheth & Tormen (1999); Sheth, Mo & Tormen (2001) | multiplicity form and ellipsoidal-collapse motivation |
| Tinker fit | Tinker et al. (2008), ApJ 688, 709; [arXiv:0803.2706](https://arxiv.org/abs/0803.2706) | `Delta=200m`, `z=0` reference calibration only |
| HMF non-universality | Fiorilli et al. (2026), [arXiv:2511.16730](https://arxiv.org/abs/2511.16730) | a modern baseline for growth-history and spectral-shape dependence |
| Shape-preserving interpolation | Fritsch & Butland (1984) | rationale for PCHIP where slopes influence HMF predictions |

## Numerical choices

The variance identity is integrated in `ln(k)`, where the integrand is `Delta^2(k) W^2(kR)`. This respects the fact that cosmological spectra span decades in wavenumber. The code does not extrapolate a tabulated spectrum; it reports the boundary integrand relative to its peak as a diagnostic. That diagnostic is necessary but not sufficient evidence of convergence—important results should also include explicit k-range and resolution studies.

The growth reference uses SciPy’s adaptive `DOP853` integrator with a matter-era growing-mode initial condition. The HMF derivative uses PCHIP in `(ln M, ln sigma)` so a generic cubic spline cannot introduce oscillatory slopes.

## Independent validation

CLASS, Colossus, and `hmf` are validation tools, not runtime dependencies of the scientific core. A comparison is meaningful only after matching: cosmological parameters, units, power-spectrum normalization, redshift, window, collapse convention, halo mass definition, and fit calibration. Record the exact external version, input configuration, safe domain, and observed discrepancy with the result.

The BibTeX entries used by the derivation are in [docs/theory/references.bib](theory/references.bib).
