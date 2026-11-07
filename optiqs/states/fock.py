import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj


def fock(
    dim: int,
    n: int,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """
    Construct an n-photon Fock state in the Fock basis.

    The Fock state is defined as:

        |n> = sum_k delta_{k,n} |k>

    Only the n-th Fock component is nonzero, with unit
    amplitude. The state is stored as a sparse column vector.

    Parameters
    ----------
    dim : int
        Truncation dimension of the Fock Hilbert space.
    n : int
        Number of photons.
    dtype : dtype, optional
        Data type of the state coefficients.
        Default is numpy.complex128.
    tol : float, optional
        Absolute threshold for removing small coefficients from
        the sparse representation. Coefficients with magnitudes
        smaller than tol are discarded.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized n-photon Fock-state ket represented
        as a sparse Qobj.

    Raises
    ------
    ValueError
        If dim is not positive or n is outside the
        truncated Fock space.

    Notes
    -----
    The state contains exactly one nonzero coefficient
    and is already normalized.

    No additional normalization is required.
    """
    if dim < 1:
        raise ValueError("dim must be positive")
    if not 0 <= n < dim:
        raise ValueError("n must satisfy 0 <= n < dim")

    data = csr_array(([1.0 + 0j], ([n], [0])), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


def vacuum(dim: int, dtype: DTypeLike = np.complex128, tol: float = 1e-17) -> Qobj:
    """
    Construct a vacuum state in the Fock basis.

    The vacuum state is the zero-photon Fock state:

        |0> = (1, 0, 0, ..., 0)^T

    Only the zeroth Fock component is nonzero, with unit
    amplitude. The state is stored as a sparse column vector.

    Parameters
    ----------
    dim : int
        Truncation dimension of the Fock Hilbert space.
    dtype : dtype, optional
        Data type of the state coefficients.
        Default is numpy.complex128.
    tol : float, optional
        Absolute threshold for removing small coefficients from
        the sparse representation. Coefficients with magnitudes
        smaller than tol are discarded.
        Default is 1e-17.

    Returns
    -------
    Qobj
        Normalized vacuum-state ket represented as a sparse Qobj.

    Raises
    ------
    ValueError
        If dim is not positive.

    Notes
    -----
    The vacuum state is equivalent to the Fock state |n=0>.

    The state contains exactly one nonzero coefficient
    and is already normalized.

    No additional normalization is required.
    """
    return fock(dim, 0, dtype=dtype, tol=tol)
