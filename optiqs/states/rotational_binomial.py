"""Rotation-symmetric binomial bosonic code states."""

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def rotational_binomial(
    dim: int,
    cutoff: int,
    order: int,
    sector: int = 0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct a normalized rotation-symmetric binomial codeword.

    For cutoff K, rotation order N, and remainder r = sector % N,
    the Fock levels r + k*N (k = 0, ..., K) have coefficients

        sqrt(comb(K, k) / 2**K) * (-1)**(logical*k),

    where logical = (sector // N) % 2. Sector 0 and sector N
    give the two logical codewords, orthogonal for K >= 1.

    Parameters
    ----------
    dim : int
        Fock-space dimension, greater than r + K*N.
    cutoff : int
        Binomial cutoff K (nonnegative), not the maximum photon number.
        K >= 1 is required for an orthogonal logical pair.
    order : int
        Positive rotation-symmetry order N.
    sector : int, optional
        Nonnegative integer selecting the Fock-number remainder and
        logical phase. Default is 0.
    dtype : DTypeLike, optional
        Coefficient data type (default: numpy.complex128).
    tol : float, optional
        Sparse coefficient-pruning threshold satisfying 0 <= tol < 1.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Single-mode ket, normalized before sparsity pruning.

    Notes
    -----
    Logarithmic binomial coefficients avoid factorial overflow.
    For K=0, the state is a single Fock state.

    For order=2 and cutoff=2, sector=0 gives
    (|0> + sqrt(2)|2> + |4>)/2, while sector=2 gives
    (|0> - sqrt(2)|2> + |4>)/2.
    """
    for name, value in (
        ("dim", dim),
        ("cutoff", cutoff),
        ("order", order),
        ("sector", sector),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    dim, cutoff, order, sector = map(int, (dim, cutoff, order, sector))
    if dim < 1:
        raise ValueError("dim must be positive")
    if cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    if order < 1:
        raise ValueError("order must be at least 1")
    if sector < 0:
        raise ValueError("sector must be nonnegative")
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")

    logical, remainder = divmod(sector, order)
    if remainder + cutoff * order >= dim:
        raise ValueError("dim must exceed remainder + cutoff * order")

    k = np.arange(cutoff + 1, dtype=np.int64)
    n = remainder + k * order

    log_comb = gammaln(cutoff + 1) - gammaln(k + 1) - gammaln(cutoff - k + 1)
    coeff = np.exp(0.5 * (log_comb - log_comb.max()))
    coeff /= np.linalg.norm(coeff)

    if logical % 2:
        coeff[1::2] *= -1

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=np.int64))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


rs_binomial = rotational_binomial
