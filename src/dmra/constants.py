"""Numerical constants and frozen project conventions."""

from __future__ import annotations

# Critical density today in the h-scaled unit system used throughout the package:
#   (M_sun / h) / (Mpc / h)^3.
# This is the standard 2.77536627e11 h^2 M_sun Mpc^-3 rewritten consistently
# in h-scaled mass and distance units.
RHO_CRIT_0_H_UNITS: float = 2.77536627e11

# Einstein-de Sitter spherical-collapse threshold used by the analytic
# mass-function baselines. Later non-standard-cosmology work must not assume
# that a constant threshold remains adequate without testing it.
DEFAULT_DELTA_C: float = 1.68647
