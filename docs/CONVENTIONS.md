# Conventions, Units, and Notation

This document freezes all mathematical definitions, coordinate systems, dimensional units, and parameter conventions across the Dark Matter Research Architecture (`DMRA`).

---

## 1. Dimensional Units ($h$-Scaled System)

All internal scientific operations in `dmra` adhere to the canonical $h$-scaled astrophysical unit system:

| Physical Quantity | Symbol | Internal Dimension / Units |
| :--- | :--- | :--- |
| Comoving Wavenumber | $k$ | $h\,\mathrm{Mpc}^{-1}$ |
| Matter Power Spectrum | $P(k)$ | $(\mathrm{Mpc}/h)^3$ |
| Dimensionless Power Spectrum | $\Delta^2(k) \equiv k^3P(k)/(2\pi^2)$ | Dimensionless |
| Comoving Smoothing Radius | $R$ | $\mathrm{Mpc}/h$ |
| Lagrangian Halo Mass | $M$ | $M_\odot / h$ |
| Mean Comoving Matter Density | $\bar\rho_{m,0}$ | $(M_\odot/h) / (\mathrm{Mpc}/h)^3$ |
| Critical Density Today | $\rho_{\mathrm{crit},0} \equiv 3H_0^2/(8\pi G)$ | $2.77536627 \times 10^{11}\,(M_\odot/h)/(\mathrm{Mpc}/h)^3$ |
| Linear Mass Variance | $\sigma(M, z)$ | Dimensionless |
| Differential Number Density | $\dd n/\dd\ln M$ | $(h/\mathrm{Mpc})^3$ |
| Dimensionless Hubble Expansion | $E(a) \equiv H(a)/H_0$ | Dimensionless |
| Physical Hubble Parameter | $H(z)$ | $\mathrm{km\,s^{-1}\,Mpc^{-1}}$ |

---

## 2. Unit Gatekeeper Transformations

Boltzmann solvers (such as $\text{CLASS}$ and $\text{AxiCLASS}$) natively output physical units ($1/\mathrm{Mpc}$ and $\mathrm{Mpc}^3$), whereas halo abundance models operate in $h$-scaled units. The `UnitGatekeeper` enforces the following exact conversions:

### Wavenumber & Power Spectrum
$$k_{h} = \frac{k_{\mathrm{CLASS}}}{h}, \qquad P_{h}(k_h) = P_{\mathrm{CLASS}}(k_{\mathrm{CLASS}}) \times h^3.$$

### Variance Invariance Identity
The smoothed variance integral is an invariant scalar under this transformation:
$$\sigma^2(R) = \frac{1}{2\pi^2}\int_0^\infty k_{\mathrm{CLASS}}^2 P_{\mathrm{CLASS}}(k_{\mathrm{CLASS}}) W^2(k_{\mathrm{CLASS}} R_{\mathrm{phys}}) \dd k_{\mathrm{CLASS}} = \frac{1}{2\pi^2}\int_0^\infty k_h^2 P_h(k_h) W^2(k_h R_h) \dd k_h.$$

### Hubble Parameter
$$H(z)\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}] = c\;[\mathrm{km/s}] \times H_{\mathrm{CLASS}}(z)\;[\mathrm{Mpc}^{-1}], \qquad c = 299792.458\,\mathrm{km/s}.$$

---

## 3. Peak-Height & Multiplicity Conventions: Cross-Literature Mapping

Cosmological literature contains several conflicting conventions for multiplicity and differential abundance. Confusing these causes $\mathcal O(1)$ discrepancies. We freeze the exact relationships:

### (i) Peak-Height Variable
- **DMRA Convention:** $\nu \equiv \frac{\delta_c(z)}{\sigma(M, z)}$, with standard spherical collapse barrier $\delta_c \simeq 1.68647$.
- **Excursion-Set Literature (e.g. Sheth & Tormen 1999):** $\nu' \equiv \frac{\delta_c^2}{\sigma^2} = \nu^2$. Under this square convention, $\dd\nu' = 2\nu\dd\nu$.

### (ii) Multiplicity Functions
- The differential mass fraction per logarithmic unit of peak height is:
  $$f(\nu) \equiv \frac{\dd F}{\dd\ln\nu}.$$
- In our convention, multiplicity expressed in terms of variance is identical:
  $$f(\sigma) \equiv \frac{\dd F}{\dd\ln\sigma^{-1}} = f(\nu).$$
- Authors quoting $\nu f(\nu)$ are expressing the differential mass fraction per linear $\nu$:
  $$\nu f(\nu) = \frac{\dd F}{\dd\ln\nu}.$$

### (iii) Differential Abundance Bases
Number densities are related by:
$$\frac{\dd n}{\dd\ln M} = M\frac{\dd n}{\dd M}, \qquad \frac{\dd n}{\dd\log_{10} M} = \ln(10)\,\frac{\dd n}{\dd\ln M} \simeq 2.302585\,\frac{\dd n}{\dd\ln M}.$$
Comparisons against simulation histograms binned in $\log_{10} M$ must always account for the $\ln(10)$ factor.

---

## 4. Cosmological & EDE Parameter Definitions

### Standard Cosmological Parameters
- $\omega_b \equiv \Omega_b h^2$: Physical baryon density parameter.
- $\omega_{\rm cdm} \equiv \Omega_{\rm cdm} h^2$: Physical cold dark matter density parameter.
- $h \equiv H_0 / (100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$: Reduced Hubble constant.
- $A_s$: Primordial scalar curvature perturbation amplitude at pivot $k_* = 0.05\,\mathrm{Mpc}^{-1}$.
- $n_s$: Primordial scalar spectral index.
- $\tau_{\rm reio}$: Optical depth to reionization.

### Early Dark Energy Parameters
- $f_{\rm EDE} \equiv \max_z \left[ \rho_\phi(z) / \rho_{\rm tot}(z) \right]$: Peak fractional energy contribution.
- $\log_{10} z_c$: Redshift of maximum fractional EDE energy contribution.
- $\theta_i \equiv \phi_i / f$: Initial dimensionless scalar field misalignment angle.
- $n$: Power in axion-like potential $V(\phi) = m^2f^2[1 - \cos(\phi/f)]^n$ (canonically $n=3$).
- $m$: Axion mass scale (determined by solver shooting).
- $f$: Axion decay constant (determined by solver shooting).
