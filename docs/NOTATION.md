# Notation and units

- `M`: Lagrangian mass in `M_sun / h` unless a file explicitly states another definition.
- `R`: comoving top-hat radius in `Mpc / h`.
- `k`: comoving wavenumber in `h / Mpc`.
- `P(k)`: linear matter power spectrum in `(Mpc / h)^3`.
- `Delta^2(k) = k^3 P(k) / (2 pi^2)`: dimensionless power per logarithmic interval.
- `rho_m0`: present-day mean comoving matter density in `(M_sun / h) / (Mpc / h)^3`.
- `sigma(R)` / `sigma(M)`: square root of top-hat-smoothed linear variance.
- `delta_c`: spherical-collapse threshold. The foundation uses the documented reference constant; later nonstandard cosmologies must test any alteration.
- `nu = delta_c / sigma`: the project-wide peak-height convention. Some papers use `nu^2`; convert explicitly.
- `f(nu)`: multiplicity per logarithmic interval in `nu` for models written in this convention.
- `dn/dM`, `dn/dlnM`: differential comoving halo number density. A halo mass definition (for example, SO overdensity and reference density) is part of the quantity, not metadata.

Never compare a quantity across code paths until its units, redshift convention, transfer/growth normalization, and halo mass definition agree.
