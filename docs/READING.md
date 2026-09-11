# Reading the halo mass function literature

This is a working reading map, not a list of citations to collect. For each paper, write down: the quantity it defines, its conventions, the assumptions that make it work, the regime where it can fail, and one concrete consequence for this code.

## The essential papers

### Press & Schechter (1974): the first mass-accounting argument

Read this for the bridge from a Gaussian smoothed density field to the fraction of matter in collapsed objects. Re-derive the thresholded Gaussian probability and identify the original factor-of-two issue. The important question is not “what is the final formula?” but “why does smoothing attach a mass scale to a random field?”

**Code connection:** `PressSchechter` supplies only a multiplicity. The common density factor and `d ln(sigma^-1) / d ln(M)` Jacobian belong in `HaloMassFunctionCalculator`, because they are transformations, not properties of Press–Schechter.

### Bond, Cole, Efstathiou & Kaiser (1991): cloud-in-cloud and first crossing

Study the random walk in `S = sigma^2`, the absorbing barrier, and the first-crossing distribution. This is the conceptual resolution of why counting threshold exceedances at each smoothing scale double-counts nested objects.

**Code connection:** the package does not simulate excursion-set walks, but its normalization tests are mass-accounting tests. The derivation explains why that matters.

### Sheth & Tormen (1999); Sheth, Mo & Tormen (2001): moving barriers

Read the 1999 paper for the fitted multiplicity and the 2001 paper for the ellipsoidal-collapse interpretation. Translate notation before comparing formulas: this project defines `nu = delta_c / sigma`, while papers often use its square.

**Code connection:** `ShethTormen` derives its normalization from `p`, has a tested Press–Schechter limit, and never hides the convention conversion.

### Tinker et al. (2008): calibration is conditional

Focus on the halo definition, overdensity, redshift dependence, and calibration domain before looking at coefficients. The point of Tinker is not that four numbers are “the HMF”; it is that a fit inherits the simulation definition and conditions that produced it.

**Code connection:** `Tinker2008Delta200MeanZ0` is intentionally long. Its name prevents the common mistake of applying one calibrated point across all redshifts and mass definitions.

### Fiorilli et al. (2026), *Evolution Mapping III*: the modern baseline to beat

This work models HMF non-universality using recent structure-formation history and local linear-power-spectrum shape. It raises the standard for an EDE result: showing disagreement with an old universal fit is not enough. A credible EDE correction must add information beyond already effective history- and shape-based descriptors.

## Software to study without copying

**CLASS** is the future source of controlled linear-theory outputs. Study its explicit inputs, precision controls, and reference configurations.

**Colossus** and **hmf** are independent reference implementations. Use them to compare matched definitions and discover convention mistakes. Do not transplant their implementation into this package; independent agreement is the point of the comparison.

**SciPy** supplies mature numerical primitives. Here, `DOP853` solves the smooth growth ODE and PCHIP controls derivative overshoot in the HMF Jacobian. Numerical libraries provide algorithms, not the project’s physics assumptions.

## A useful note-taking discipline

Label every equation in your derivation as one of: **definition**, **derived result**, **physical assumption**, **empirical calibration**, or **numerical choice**. That single habit prevents a large fraction of cosmology-code mistakes.
