import math

import numpy as np
from scipy.integrate import simpson
from scipy.special import eval_hermite

from optiqs.utils import Qobj


def gkp(dim: int, sq_level: float):
    q_max = 10
    q = np.linspace(-q_max, q_max, 2**12)
    n_range = np.arange(
        -int(q_max / math.sqrt(math.pi)) - 1, int(q_max / math.sqrt(math.pi)) + 1
    )

    sq_param = sq_level * math.log(10) / 20
    delta = math.exp(-sq_param) / math.sqrt(2)

    def fock_q(n):
        return (
            np.exp(-(q**2) / 2)
            * eval_hermite(n, q)
            / np.sqrt((2**n) * math.factorial(n) * math.sqrt(math.pi))
        )

    _logical_0 = np.zeros_like(q)
    for n in n_range:
        center = (2 * n) * math.sqrt(math.pi)
        _logical_0 += np.exp(-((q - center) ** 2) / (2 * delta**2))
    _logical_0 *= np.exp(-(q**2) / (2 * (1 / delta) ** 2))
    _logical_0 /= np.sqrt(simpson(np.abs(_logical_0) ** 2, q))
    logical_0 = np.zeros(dim, dtype=complex)
    for n in range(dim):
        logical_0[n] = simpson(fock_q(n) * _logical_0, q)
    logical_0 /= np.linalg.norm(logical_0)
    logical_0 = Qobj(logical_0)

    _logical_1 = np.zeros_like(q)
    for n in n_range:
        center = (2 * n + 1) * math.sqrt(math.pi)
        _logical_1 += np.exp(-((q - center) ** 2) / (2 * delta**2))
    _logical_1 *= np.exp(-(q**2) / (2 * (1 / delta) ** 2))
    _logical_1 /= np.sqrt(simpson(np.abs(_logical_1) ** 2, q))
    logical_1 = np.zeros(dim, dtype=complex)
    for n in range(dim):
        logical_1[n] = simpson(fock_q(n) * _logical_1, q)
    logical_1 /= np.linalg.norm(logical_1)
    logical_1 = Qobj(logical_1)

    return logical_0, logical_1
