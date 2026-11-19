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


@njit(cache=True)
def fused_wigner_clenshaw(
    coefficients: NDArray[np.complex128],
    starts: NDArray[np.int64],
    lengths: NDArray[np.int64],
    offsets: NDArray[np.int64],
    inverse_denominators: NDArray[np.float64],
    back_factors: NDArray[np.float64],
    amplitudes: NDArray[np.complex128],
    arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate all superdiagonals, offset powers and the Gaussian in one loop.

    This avoids allocating one complex vector per superdiagonal and per
    missing power, which was a sizable overhead with many Fock levels.
    """
    output = np.empty(len(arguments), dtype=np.float64)
    for j in range(len(arguments)):
        accumulator = 0j
        previous = -1
        for k in range(len(offsets)):
            offset = offsets[k]
            if previous >= 0:
                for power in range(previous, offset, -1):
                    accumulator *= amplitudes[j] / np.sqrt(power)
            ahead = 0j
            two_ahead = 0j
            start = starts[k]
            for n in range(lengths[k] - 1, -1, -1):
                idx = start + n
                current = (
                    coefficients[idx]
                    + (arguments[j] - (2 * n + offset + 1))
                    * inverse_denominators[idx]
                    * ahead
                    - back_factors[idx] * two_ahead
                )
                two_ahead = ahead
                ahead = current
            accumulator += ahead
            previous = offset
        if previous >= 0:
            for power in range(previous, 0, -1):
                accumulator *= amplitudes[j] / np.sqrt(power)
        output[j] = np.exp(-0.5 * arguments[j]) * accumulator.real
    return output
