# Physical Scope and Modeling Boundaries

This document defines the operational scope, domain of validity, and physical modeling boundaries for the linear theory and halo-scale variance architecture in `dmra`.

---

## 1. Domain of Physical Applicability

### Linear Regime Boundaries
The linear theory engine evaluates the background expansion $H(a)$, scalar field energy fraction $\Omega_\phi(a)$, and matter clustering power spectrum $P_m(k, z)$ strictly within the regime of linear cosmological perturbation theory. This description is physically robust for wavenumbers $k \lesssim 10\,h\,\mathrm{Mpc}^{-1}$ at high redshift ($z \gtrsim 5$) and $k \lesssim 0.1\,h\,\mathrm{Mpc}^{-1}$ at $z=0$. Nonlinear mode-coupling, halo assembly bias, and virial velocity dispersions fall outside linear perturbation theory and cannot be predicted by Boltzmann solvers alone; they require full $N$-body simulations.

### Matter Sector Approximations
The halo variance and mass function calculations treat cold dark matter and baryons within the standard single-fluid collisionless approximation on halo scales ($M \sim 10^8\text{--}10^{16}\,M_\odot/h$). The default benchmark models assume three standard massless/relativistic neutrino species ($N_{\rm eff} = 3.044$); massive neutrino free-streaming cutoffs and warm dark matter particle thermal velocities are deliberately excluded from the baseline EDE configurations to isolate scalar field dynamics.

### Filter Topology
Mass variance $\sigma^2(M, z)$ and the exact analytical Jacobian $J(M)$ are evaluated exclusively using comoving spherical top-hat filtering in real space. Alternative topologies (such as sharp-$k$ or Gaussian filtering) are omitted from default mass-radius relations to preserve exact consistency with excursion-set and spherical-collapse definitions.

---

## 2. Multiplicity Model Boundaries

- **Tinker (2008) Calibration:** The `Tinker2008Delta200MeanZ0` implementation is strictly scoped to its empirical calibration domain ($\Delta = 200$ relative to mean background matter density at $z=0$). Redshift evolution and alternative overdensity definitions ($\Delta_{\rm crit}$, virial overdensity) are intentionally scoped until full halo-finding machinery is integrated.
- **Evolution Mapping Baseline:** The linear descriptors computed by DMRA ($P_m(k,z), D_{\rm eff}(k,z), f(a), \sigma(M,z), n_{\rm eff}(M,z)$) provide the exact inputs required by modern non-universal models (such as Evolution Mapping III; Fiorilli et al. 2026), providing an auditable reference baseline.

---

## 3. Scientific Roadmap: Causal Matching & $N$-Body Simulations

The linear theory engine developed here serves as the input operator $\mathcal L(\theta)$ for the downstream causal matching experiment:
1. **Cosmological Pairing:** Identifying degenerate cosmology pairs $(A, B)$ where linear power spectra match at target redshift $z_\star$ while growth histories diverge.
2. **$N$-Body Initial Conditions:** Generating initial conditions using the audited linear power grids and scale-dependent growth factors.
3. **Halo Mass Function Extraction:** Measuring simulated halo abundance with Friends-of-Friends and spherical-overdensity halo finders to rigorously isolate early dynamical memory.
