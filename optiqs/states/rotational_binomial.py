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

    ``order=N`` specifies an N-fold rotation symmetry with angle ``2*pi/N``
    (180, 120, and 90 degrees for N=2, 3, 4). The codeword uses Fock
    levels ``r + k*N`` for ``k=0, ..., K``, with coefficients

        sqrt(comb(K, k) / 2**K) * (-1)**(logical*k),

    where ``r = sector % order``, ``logical = (sector // order) % 2``,
    and ``K = floor((cutoff - r) / order)``. In particular,
    ``sector=0`` generates the all-positive state and ``sector=order``
    generates the alternating-sign state. They are orthogonal for K >= 1.
    Both have N-fold rotational symmetry; a rotation by ``pi/N``
    interchanges them (up to a global phase).

    Parameters
    ----------
    dim : int
        Fock-space simulation dimension. Photon numbers from 0 to
        ``dim - 1`` are representable; ``dim`` must exceed ``cutoff``.
    cutoff : int
        Maximum allowed photon number (inclusive). A nontrivial logical
        pair with ``sector=0`` and ``sector=order`` requires
        ``cutoff >= order``.
    order : int
        Positive rotation-symmetry order N. The angular period is
        ``360 / order`` degrees when K >= 1 and both parities occur.
    sector : int, optional
        Selects the Fock-number remainder ``r = sector % order`` and
        the logical phase ``(sector // order) % 2``. Must satisfy
        ``0 <= sector <= cutoff``. Default is 0.
    dtype : DTypeLike, optional
        Coefficient data type (default: numpy.complex128).
    tol : float, optional
        Sparse coefficient-pruning threshold satisfying 0 <= tol < 1.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized single-mode ket, stored as a CSR sparse column vector.

    Notes
    -----
    Logarithmic binomial coefficients avoid factorial overflow. Each
    codeword is normalized before sparsity pruning; large ``tol`` values
    may reduce its final norm. For ``K=0``, a single Fock state is returned
    (its rotation symmetry is not uniquely N-fold).

    For ``order=2, cutoff=4``, ``sector=0`` generates
    ``(|0> + sqrt(2)|2> + |4>)/2``, while ``sector=2`` generates
    ``(|0> - sqrt(2)|2> + |4>)/2``.
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
            raise ValueError(f"{name} must be an integer")

    if dim < 1:
        raise ValueError("dim must be positive")
    if cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    if order < 1:
        raise ValueError("order must be at least 1")
    if not 0 <= sector <= cutoff:
        raise ValueError("sector must satisfy 0 <= sector <= cutoff")
    if cutoff >= dim:
        raise ValueError("dim must be at least cutoff + 1")
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")

    logical, remainder = divmod(sector, order)
    degree = (cutoff - remainder) // order
    k = np.arange(degree + 1, dtype=np.int64)
    n = remainder + k * order

    log_comb = gammaln(degree + 1) - gammaln(k + 1) - gammaln(degree - k + 1)
    log_amp = 0.5 * log_comb
    coeff = np.exp(log_amp - log_amp.max())
    coeff /= np.linalg.norm(coeff)

    if logical % 2:
        # Alternating signs give the orthogonal logical codeword.
        coeff[1::2] *= -1

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=np.int64))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


rs_binomial = rotational_binomial
