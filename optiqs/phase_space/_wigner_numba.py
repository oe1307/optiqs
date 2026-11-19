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
_numba = import_module("numba")
njit = _numba.njit
prange = _numba.prange


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


@njit(cache=True, inline="always")
def _wigner_point(
    coefficients: NDArray[np.complex128],
    starts: NDArray[np.int64],
    lengths: NDArray[np.int64],
    offsets: NDArray[np.int64],
    inverse_denominators: NDArray[np.float64],
    back_factors: NDArray[np.float64],
    amplitudes: NDArray[np.complex128],
    arguments: NDArray[np.float64],
    j: int,
) -> float:
    """Pointwise Clenshaw recurrence shared by sequential/parallel kernels."""
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
    return np.exp(-0.5 * arguments[j]) * accumulator.real


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
    """Sequential fused Clenshaw recurrence for small grids."""
    output = np.empty(len(arguments), dtype=np.float64)
    for j in range(len(arguments)):
        output[j] = _wigner_point(
            coefficients,
            starts,
            lengths,
            offsets,
            inverse_denominators,
            back_factors,
            amplitudes,
            arguments,
            j,
        )
    return output


@njit(cache=True, parallel=True)
def parallel_fused_wigner_clenshaw(
    coefficients: NDArray[np.complex128],
    starts: NDArray[np.int64],
    lengths: NDArray[np.int64],
    offsets: NDArray[np.int64],
    inverse_denominators: NDArray[np.float64],
    back_factors: NDArray[np.float64],
    amplitudes: NDArray[np.complex128],
    arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distribute independent grid points over Numba's configured threads."""
    output = np.empty(len(arguments), dtype=np.float64)
    for j in prange(len(arguments)):
        output[j] = _wigner_point(
            coefficients,
            starts,
            lengths,
            offsets,
            inverse_denominators,
            back_factors,
            amplitudes,
            arguments,
            j,
        )
    return output
