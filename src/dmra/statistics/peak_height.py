"""Peak-height convention used throughout the project."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from dmra._typing import FloatArray
from dmra.constants import DEFAULT_DELTA_C


def peak_height(sigma: ArrayLike, *, delta_c: float = DEFAULT_DELTA_C) -> FloatArray:
    r"""Return :math:`\nu = \delta_c / \sigma`.

    Some literature uses ``nu`` for ``delta_c^2 / sigma^2``. This project does
    not. The convention is frozen here and documented in ``docs/CONVENTIONS.md``.
    """
    sigma_array = np.asarray(sigma, dtype=np.float64)
    if np.any(~np.isfinite(sigma_array)) or np.any(sigma_array <= 0.0):
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(delta_c) or delta_c <= 0.0:
        raise ValueError("delta_c must be finite and positive")
    return delta_c / sigma_array
