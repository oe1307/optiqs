import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def rotational_cat(
    dim: int,
    alpha: complex,
    order: int,
    sector: int = 0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct a normalized, rotation-symmetric cat state in the Fock basis.

    The order-M cat state in rotation sector s is proportional to

        sum_{j=0}^{M-1} exp(-2*pi*i*s*j/M)
            |alpha * exp(2*pi*i*j/M)>.

    This is equivalently the coherent-state Fock expansion projected onto
    photon numbers n satisfying n % M == s. Thus only those coefficients
    need to be computed, and no coherent-state sums or operator matrix
    exponentials are required.

    Parameters
    ----------
    dim : int
        Truncation dimension of the single-mode Fock Hilbert space.
    alpha : complex
        Coherent-state amplitude (must be finite).
    order : int
        Rotational symmetry order M.
        Must be an integer >= 1.
        For order=1, the state reduces to a coherent state.
    sector : int, optional
        Rotation-eigenvalue sector s in [0, order). The resulting ket is
        an eigenstate of exp(2*pi*i*n_hat/order) with eigenvalue
        exp(2*pi*i*sector/order). Default is 0.
    dtype : DTypeLike, optional
        Output data type. Default is numpy.complex128.
    tol : float, optional
        Sparse coefficient pruning threshold, with 0 <= tol < 1.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized sparse ket with Fock support n = sector (mod order).

    Raises
    ------
    ValueError
        For invalid parameters, no Fock basis vector in the requested
        sector, or alpha == 0 with sector != 0 (the unnormalized
        superposition vanishes at alpha == 0).

    Notes
    -----
    The coefficients are normalized *within* the truncated Fock space.
    Logarithmic magnitudes avoid factorial overflow and numerical
    underflow during intermediate computations. For alpha == 0 and
    sector == 0, the state is the vacuum.

    With K = ceil((dim - sector)/order) selected levels, evaluating
    amplitudes costs O(K) time and space. The CSR matrix's row pointers
    additionally require O(dim) storage. Pruning small coefficients may
    slightly change the returned norm, as for other OptiQS states.

    For a conventional two-component cat, use order=2, sector=0 (even)
    or sector=1 (odd). For a rotation-symmetric bosonic code of degree N,
    the commonly used logical pair is obtained using order=2*N and
    sectors 0 and N, respectively.
    """
    for name, value in (("dim", dim), ("order", order), ("sector", sector)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    if dim < 1:
        raise ValueError("dim must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    if not 0 <= sector < order:
        raise ValueError("sector must satisfy 0 <= sector < order")
    if sector >= dim:
        raise ValueError("dim is too small for the requested sector")
    if not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if alpha == 0 and sector != 0:
        raise ValueError("Nonzero rotation sectors are undefined at alpha=0")

    n = np.arange(sector, dim, order, dtype=np.int64)
    if alpha == 0:
        n = n[:1]
        coeff = np.array([1.0 + 0j])
    else:
        log_amp = n * np.log(abs(alpha)) - 0.5 * gammaln(n + 1)
        coeff = np.exp(log_amp - log_amp.max() + 1j * n * np.angle(alpha))
        coeff /= np.linalg.norm(coeff)

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=np.int64))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


rs_cat = rotational_cat
