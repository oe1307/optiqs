import numpy as np
import scipy as sp

from optiqs.utils import Qobj


def cat(
    dim: int, alpha: float | complex, odd: bool = False, sparse: bool = False
) -> Qobj:
    n = np.arange(1 if odd else 0, dim, 2, dtype=int)
    log_sqrt_fact = 0.5 * sp.special.gammaln(n + 1)
    data = np.power(alpha, n).astype(complex) * np.exp(-log_sqrt_fact)

    if sparse:
        rows = n
        cols = np.zeros_like(n)
        state_sp = sp.sparse.csr_matrix(
            (data, (rows, cols)), shape=(dim, 1), dtype=complex
        )
        return Qobj(state_sp).unit()
    else:
        state = np.zeros(dim, dtype=complex)
        state[n] = data
        return Qobj(state).unit()
