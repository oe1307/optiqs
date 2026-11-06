"""Rotation-symmetric binomial bosonic code states."""

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def rotational_binomial(
    dim: int,
    order: int,
    binomial_order: int,
    logical: int = 0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct a normalized rotation-symmetric binomial codeword.

    The logical basis states are

        |b_L> = 1/sqrt(2**(K-1)) * sum_{p=0..K, p%2=b}
                sqrt(comb(K, p)) |p*N>,

    where N=order, K=binomial_order, and b=logical (0 or 1).
    Both states are invariant under a rotation by 2*pi/N. A rotation by
    pi/N implements logical Z (eigenvalue +1 for logical=0, -1 for 1).

    Parameters
    ----------
    dim : int
        Fock-space dimension; must be at least order*binomial_order + 1
        so that the complete logical code space fits without truncation.
    order : int
        Positive rotation-symmetry order N (Fock spacing).
        The two logical sectors have photon numbers 0 and N modulo 2*N.
    binomial_order : int
        Positive binomial degree K, determining the maximum photon number
        K*N in the code space.
    logical : {0, 1}, optional
        Which computational-basis codeword to construct. Default is 0.
    dtype : DTypeLike, optional
        Coefficient data type (default: numpy.complex128).
    tol : float, optional
        Sparse coefficient-pruning threshold, satisfying 0 <= tol < 1.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized single-mode ket, stored as a CSR sparse column vector.

    Notes
    -----
    Logarithmic binomial coefficients (gammaln) prevent factorial overflow.
    Coefficients are normalized before the sparsity threshold is applied;
    very aggressive pruning may slightly reduce the returned norm.

    This is the computational (even/odd) logical basis, not the conjugate
    |+_L> and |-_L> basis. For order=2 and binomial_order=2, the states
    are (|0>+|4>)/sqrt(2) and |2>, respectively.
    """
    for name, value in (
        ("dim", dim),
        ("order", order),
        ("binomial_order", binomial_order),
        ("logical", logical),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer")

    if dim < 1:
        raise ValueError("dim must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    if binomial_order < 1:
        raise ValueError("binomial_order must be at least 1")
    if logical not in (0, 1):
        raise ValueError("logical must be 0 or 1")
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")
    if dim <= order * binomial_order:
        raise ValueError("dim must be at least order*binomial_order + 1")

    p = np.arange(logical, binomial_order + 1, 2, dtype=np.int64)
    n = p * order

    # Remove a common logarithmic scale, then normalize within the parity
    # sector. This avoids overflow even for large binomial orders.
    log_comb = (
        gammaln(binomial_order + 1) - gammaln(p + 1) - gammaln(binomial_order - p + 1)
    )
    log_amp = 0.5 * log_comb
    coeff = np.exp(log_amp - log_amp.max())
    coeff /= np.linalg.norm(coeff)

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=np.int64))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


rs_binomial = rotational_binomial
