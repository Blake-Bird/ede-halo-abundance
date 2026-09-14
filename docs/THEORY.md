# Theoretical Foundations: Early Dark Energy & Halo Statistics

This document provides the complete mathematical and physical architecture of the Dark Matter Research Architecture (`DMRA`), connecting Early Dark Energy (EDE) scalar field dynamics to dark matter halo abundance.

---

## 1. Physical Motivation: The Hubble Tension & EDE
Measurements of the local expansion rate from Type Ia supernovae calibrated by Cepheids (SH0ES; Riess et al. 2022) yield $H_0 = 73.04 \pm 1.04\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$, diverging by $>5\sigma$ from the Planck $\Lambda\mathrm{CDM}$ inference ($H_0 = 67.36 \pm 0.54\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$).

Early Dark Energy resolves this tension by introducing a scalar field $\phi$ that contributes $f_{\rm EDE} \sim 8\text{--}12\%$ to the cosmic energy density around matter-radiation equality ($z_c \sim 10^{3.5}$) before diluting rapidly. The elevated expansion rate reduces the comoving acoustic sound horizon:
$$r_s(z_*) = \int_{z_*}^\infty \frac{c_s(z)}{H(z)}\dd z.$$
Preserving the exquisitely measured CMB acoustic scale $\theta_* = r_s(z_*) / D_A(z_*)$ requires a corresponding decrease in the angular diameter distance $D_A(z_*)$, which necessitates an increase in the late-time Hubble constant $H_0$.

---

## 2. Scalar Field Action & Equations of Motion
Consider a real scalar field $\phi$ minimally coupled to gravity with action:
$$S_\phi = \int \dd^4x \sqrt{-g} \left[ -\frac{1}{2}g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi - V(\phi) \right].$$

Varying with respect to $\phi$ in flat FLRW spacetime yields the homogeneous Klein--Gordon equation:
$$\ddot\phi + 3H\dot\phi + V_{,\phi} = 0.$$

The energy density and pressure are:
$$\rho_\phi = \frac{1}{2}\dot\phi^2 + V(\phi), \qquad p_\phi = \frac{1}{2}\dot\phi^2 - V(\phi),$$
giving the instantaneous equation of state:
$$w_\phi = \frac{\frac{1}{2}\dot\phi^2 - V(\phi)}{\frac{1}{2}\dot\phi^2 + V(\phi)}.$$

---

## 3. The Axion Potential & Turner (1983) Virial Dilution
Canonical EDE adopts the periodic cosine potential (Poulin et al. 2018, 2019):
$$V(\phi) = m^2f^2\left[ 1 - \cos\left(\frac{\phi}{f}\right) \right]^n,$$
where $m$ is the axion mass scale, $f$ is the symmetry-breaking decay constant, and $n=3$ is the canonical exponent. Defining the misalignment angle $\theta \equiv \phi/f$:
- **Frozen Regime ($z \gg z_c$):** When $H \gg m_{\rm eff} \equiv \sqrt{V_{,\phi\phi}}$, Hubble friction overdamps field motion ($\dot\phi \approx 0$). The field remains frozen on its potential with $w_\phi \approx -1$, behaving as dark energy.
- **Oscillatory Virial Regime ($z \ll z_c$):** Once $H \lesssim m_{\rm eff}$, the field rolls into the minimum and oscillates rapidly. Taylor expanding $1 - \cos\theta \approx \theta^2/2$ shows that $V(\phi) \propto \phi^{2n} = \phi^6$.

Applying the Turner (1983) virial theorem across one complete cycle:
$$\langle \dot\phi^2 \rangle = 2n \langle V(\phi) \rangle \implies \langle w_\phi \rangle = \frac{n-1}{n+1}.$$
For $n=3$, $\langle w_\phi \rangle = 1/2$. Energy conservation $\dot\rho_\phi + 3H(1 + \langle w_\phi \rangle)\rho_\phi = 0$ yields the rapid asymptotic dilution law:
$$\rho_\phi(a) \propto a^{-6n/(n+1)} = a^{-4.5}.$$
The field dilutes faster than radiation ($\propto a^{-4}$), leaving late-time cosmological evolution virtually undisturbed.

---

## 4. Background Cosmology & Expansion Kinematics
The modified Friedmann equation in flat FLRW is:
$$E^2(a) \equiv \left(\frac{H(a)}{H_0}\right)^2 = \Omega_{r,0}a^{-4} + \Omega_{m,0}a^{-3} + \Omega_{\Lambda,0} + \frac{\rho_\phi(a)}{\rho_{\mathrm{crit},0}}.$$

The expansion perturbation is defined as:
$$\Delta_H(z) \equiv \frac{H_{\mathrm{EDE}}(z) - H_{\Lambda\mathrm{CDM}}(z)}{H_{\Lambda\mathrm{CDM}}(z)}.$$

---

## 5. Linear Perturbation Theory & Gauge Equivalence
On subhorizon scales in conformal Newtonian gauge, pressureless cold dark matter perturbations obey:
$$\ddot\delta + 2H\dot\delta - 4\pi G\bar\rho_m\delta = 0.$$

### Gauge Invariance
Boltzmann solvers evaluate perturbations in synchronous gauge ($h, \eta$). The coordinate transformation to Newtonian gauge satisfies:
$$\delta_m^{(\mathrm{syn})} = \delta_m^{(\mathrm{Newt})} + \mathcal O\left(\frac{a^2H^2}{k^2}\right).$$
On halo-forming scales ($k \gtrsim 0.05\,h\,\mathrm{Mpc}^{-1}$), the gauge difference is $< 10^{-5}$, establishing physical equivalence.

### Scalar Field Jeans Scale
Scalar perturbations possess an effective sound speed $c_s^2(k,a)$ and comoving Jeans wavenumber:
$$k_J(a) \sim a\sqrt{m_{\rm eff} H}.$$
For $k \gg k_J$, pressure gradient forces prevent scalar clustering ($c_s^2 \to 1$). Early Dark Energy influences halo collapse strictly through the modified expansion rate $H(a)$ and background gravitational potential.

---

## 6. Linear Growth Factor & Scale-Dependence Diagnostics
Transforming time derivatives to scale factor $a$ yields Heath's (1977) growth ODE:
$$D''(a) + \left[ \frac{3}{a} + \frac{\dd\ln H}{\dd a} \right]D'(a) - \frac{3}{2}\frac{\Omega_{m,0}H_0^2}{a^5 H^2(a)}D(a) = 0.$$
The logarithmic growth rate is $f(a) \equiv \dd\ln D/\dd\ln a = aD'/D$.

To audit scale-dependence in real Boltzmann power grids, we define the effective growth factor:
$$D_{\rm eff}(k,z) \equiv \sqrt{\frac{P_m(k,z)}{P_m(k,0)}},$$
and the scale-dependence diagnostic:
$$\epsilon_D(k,z) \equiv \frac{D_{\rm eff}(k,z)}{D_{\rm eff}(k_{\rm ref},z)} - 1 \qquad (k_{\rm ref} = 0.01\,h\,\mathrm{Mpc}^{-1}).$$
This diagnostic should be evaluated from each retained solver output; no universal numerical bound follows from the equations alone.

---

## 7. From Linear Power Spectra to Halo Variance
The comoving Lagrangian radius $R$ enclosing mass $M$ is:
$$R(M) = \left(\frac{3M}{4\pi\bar\rho_{m,0}}\right)^{1/3}, \qquad \bar\rho_{m,0} = \Omega_{m,0}\rho_{\mathrm{crit},0}.$$

The Fourier-space top-hat filter is:
$$W(x) = 3\,\frac{\sin x - x\cos x}{x^3}, \qquad x \equiv kR.$$
Smoothed mass variance is computed by logarithmic Simpson quadrature:
$$\sigma^2(M,z) = \int_{-\infty}^\infty \Delta^2(k,z) W^2(k R(M)) \dd\ln k, \qquad \Delta^2(k) \equiv \frac{k^3P(k)}{2\pi^2}.$$

### Exact Analytical Logarithmic Jacobian
Rather than relying on noisy numerical splines, DMRA evaluates the exact analytical Jacobian:
$$J(M) \equiv \frac{\dd\ln\sigma^{-1}}{\dd\ln M} = -\frac{1}{3\sigma^2}\int_{-\infty}^\infty \dd\ln k\,\Delta^2(k)\,W(kR)\left[ kR\,W'(kR) \right],$$
where $W'(x) = \frac{3}{x}\left[\frac{\sin x}{x} - W(x)\right]$. This guarantees floating-point precision across all multiplicity models.

---

## 8. Multiplicity Models & Non-Universality
The halo mass function is given by:
$$\frac{\dd n}{\dd\ln M} = \frac{\bar\rho_{m,0}}{M}\,f(\nu)\,J(M), \qquad \nu \equiv \frac{\delta_c}{\sigma(M,z)}.$$

- **Press--Schechter (1974):** Spherical collapse with Gaussian thresholding:
  $$f_{\rm PS}(\nu) = \sqrt{\frac{2}{\pi}}\nu e^{-\nu^2/2}.$$
- **Sheth--Tormen (1999):** Ellipsoidal collapse with moving barrier:
  $$f_{\rm ST}(\nu) = A\sqrt{\frac{2a}{\pi}}\left[1 + (a\nu^2)^{-p}\right]\nu e^{-a\nu^2/2} \qquad (a=0.707, p=0.3, A \simeq 0.32218).$$
- **Tinker et al. (2008):** Calibrated simulation fit across $z=0\text{--}2.5$.
- **Evolution Mapping III (Fiorilli et al. 2026):** State-of-the-art parameterization capturing non-universality through growth history and local spectral slope $n_{\rm eff}(M) \equiv \dd\ln P/\dd\ln k|_{k=1/R(M)}$.

### Rare-Peak Exponential Sensitivity
Differentiating Press--Schechter multiplicity with respect to $\sigma$ yields:
$$\frac{\partial\ln f_{\rm PS}}{\partial\ln\sigma} = \nu^2 - 1 \approx \nu^2 \qquad (\nu \gg 1).$$
At $z \sim 8$, massive halos correspond to $\nu \sim 4\text{--}5$, yielding $\nu^2 \sim 16\text{--}25$. A $2\%$ increase in linear variance produces an exponential $30\%\text{--}50\%$ surge in halo abundance, demonstrating why high-$z$ halo statistics act as a powerful test of early expansion physics.

---

## 9. Matched-Cosmology Causal Framework
To isolate dynamical growth memory from instantaneous linear field shape, we formulate the matched-cosmology framework. We construct pairs $(A, B)$ minimizing power spectrum distance:
$$d_P(A, B) \equiv \int_{k_{\mathrm{min}}}^{k_{\mathrm{max}}} w_k \left[ \ln P_A(k, z_\star) - \ln P_B(k, z_\star) \right]^2 \dd\ln k \to 0,$$
while maximizing growth history divergence:
$$d_G(A, B) \equiv \int_0^{z_\star} w_z \left[ f_A(z) - f_B(z) \right]^2 \dd z \gg 0.$$
N-body simulations across matched cosmologies definitively isolate whether halo collapse retains memory of non-standard growth histories.
