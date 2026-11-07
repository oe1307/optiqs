import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj


def gkp(
    dim: int,
    logical: int = 0,
    delta: float = 0.3,
    kappa: float | None = None,
    cutoff: int = 4,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """
    Construct a normalized approximate GKP state in the Fock basis.

    The state is represented as a Gaussian-weighted superposition
    of displaced squeezed vacuum states on a square GKP lattice.

    Fock coefficients are computed using a recurrence relation,
    avoiding explicit displacement and squeezing operators.

    Parameters
    ----------
    dim : int
        Truncation dimension of the Fock Hilbert space.
    logical : {0, 1}, optional
        Logical GKP basis state. Default is 0.
    delta : float, optional
        Width of the Gaussian peaks in position space.
        Must satisfy 0 < delta <= 1. Default is 0.3.
    kappa : float or None, optional
        Inverse width of the Gaussian envelope.
        If None, kappa is set to delta.
    cutoff : int, optional
        Number of positive peak pairs included in the
        Gaussian superposition. Default is 4.
    dtype : dtype, optional
        Data type of the state coefficients.
        Default is numpy.complex128.
    tol : float, optional
        Absolute threshold for removing small coefficients
        from the sparse representation. Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized approximate GKP-state ket represented
        as a sparse Qobj.

    Raises
    ------
    ValueError
        If dim, logical, delta, kappa, or cutoff is invalid,
        or the state cannot be normalized numerically.

    Notes
    -----
    Both logical basis states are symmetric under position
    inversion and contain only even Fock components.

    The state is normalized within the truncated Fock space
    before applying the sparsity tolerance. Removing small
    coefficients may slightly reduce the final norm.

    The computational complexity is O(dim * cutoff).
    No matrix exponentiation or dense operators are required.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    if logical not in (0, 1):
        raise ValueError("logical must be 0 or 1")
    if not np.isfinite(delta) or not 0 < delta <= 1:
        raise ValueError("delta must satisfy 0 < delta <= 1")

    if kappa is None:
        kappa = delta
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if cutoff < 1:
        raise ValueError("cutoff must be positive")

    # Non-negative lattice positions
    m = np.arange(logical, 2 * cutoff + 1, 2)
    q = m * np.sqrt(np.pi)
    alpha = q / np.sqrt(2)

    # Squeezing parameters
    t = (1 - delta**2) / (1 + delta**2)
    beta = alpha * (1 + t)

    # Vacuum amplitudes of displaced squeezed states
    c0 = np.sqrt(2 * delta / (1 + delta**2))
    c0 *= np.exp(-(alpha**2) / (1 + delta**2))

    # Gaussian envelope and symmetric peak multiplicities
    weights = np.exp(-0.5 * (kappa * q) ** 2)
    weights *= np.where(m == 0, 1.0, 2.0)

    # Only even Fock components are required
    coeff = np.empty((dim + 1) // 2, dtype=np.float64)
    c_prev = np.zeros_like(c0)
    c_curr = c0

    for k in range(dim):
        if k % 2 == 0:
            coeff[k // 2] = np.dot(weights, c_curr)
        if k + 1 < dim:
            c_next = (beta * c_curr - t * np.sqrt(k) * c_prev) / np.sqrt(k + 1)
            c_prev, c_curr = c_curr, c_next

    # Normalize
    norm = np.linalg.norm(coeff)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Failed to normalize the GKP state")
    coeff /= norm

    n = np.arange(0, dim, 2)
    mask = np.abs(coeff) >= tol

    if not np.any(mask):
        raise ValueError("No coefficients remain after applying tol")

    data = csr_array(
        (coeff[mask], (n[mask], np.zeros(np.count_nonzero(mask), dtype=int))),
        shape=(dim, 1),
    )

    return Qobj(data, dtype=dtype, tol=tol)
