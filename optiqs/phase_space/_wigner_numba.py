"""Opt-in Numba implementation of the Wigner Clenshaw inner recurrence.

This module is deliberately imported only when accelerator='numba'. Normal
OptiQS installations never need Numba or incur a JIT warm-up cost.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
from numpy.typing import NDArray

# Numba is optional; resolve it only when this module is explicitly loaded.
# A direct `from numba import njit` fails static checking in environments
# where the optional JIT extra is not installed.
njit = import_module("numba").njit


@njit(cache=True)
def clenshaw_laguerre(
    coefficients: NDArray[np.complex128],
    offset: int,
    argument: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Evaluate all phase-space points without intermediate vector temporaries."""
    count = len(coefficients)
    output = np.empty(len(argument), dtype=np.complex128)
    denominators = np.empty(count, dtype=np.float64)
    back_factors = np.empty(count, dtype=np.float64)
    for n in range(count):
        denominator = np.sqrt((n + 1) * (n + offset + 1))
        denominators[n] = denominator
        back_factors[n] = denominator / np.sqrt((n + 2) * (n + offset + 2))
    for j in range(len(argument)):
        ahead = 0j
        two_ahead = 0j
        for n in range(count - 1, -1, -1):
            current = (
                coefficients[n]
                + ((argument[j] - (2 * n + offset + 1)) / denominators[n]) * ahead
                - back_factors[n] * two_ahead
            )
            two_ahead = ahead
            ahead = current
        output[j] = ahead
    return output
