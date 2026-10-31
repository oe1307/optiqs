import math

import numpy as np

from optiqs.utils.typing import Qobj, Qtype


def wigner(state: Qobj, xaxis, yaxis, scale=math.sqrt(2)) -> np.ndarray:
    """Wigner function of a quantum state."""
    if state.qtype == Qtype.ket:
        state = state.dm()

    xgrid, ygrid = np.meshgrid(xaxis, yaxis)
    position = scale * (xgrid + 1.0j * ygrid)
    distance = np.abs(position) ** 2

    # Clenshaw algorithm
    w0 = (2 * state[0, -1]) * np.ones_like(position)
    step = state.dims[0][0] - 1
    state = 2 * state - np.diag(state.diag())
    while step > 0:
        step -= 1
        w0 = _laguerre(step, distance, np.diag(state.data, step)) + (
            w0 * position / math.sqrt(step + 1)
        )
    return w0.real * np.exp(-distance * 0.5) * scale**2 / (2 * math.pi)


def _laguerre(step, distance, coeff):
    """Generalized Laguerre polynomial"""
    if len(coeff) == 1:
        y0 = coeff[0]
        y1 = 0
    elif len(coeff) == 2:
        y0 = coeff[0]
        y1 = coeff[1]
    else:
        k = len(coeff)
        y0 = coeff[-2]
        y1 = coeff[-1]
        for i in range(3, len(coeff) + 1):
            k -= 1
            y0, y1 = (
                coeff[-i]
                - y1 * (float((k - 1) * (step + k - 1)) / ((step + k) * k)) ** 0.5,
                y0 - y1 * ((step + 2 * k - 1) - distance) * ((step + k) * k) ** -0.5,
            )
    return y0 - y1 * ((step + 1) - distance) * (step + 1) ** -0.5
