import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def cat(
    dim: int,
    alpha: complex,
    odd: bool = False,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """
    Construct a normalized Schrödinger cat state in the Fock basis.

    The cat state is a superposition of two coherent states
    with opposite amplitudes:

        |C_±(alpha)> ∝ |alpha> ± |-alpha>

    Only Fock components matching the specified parity are
    computed and stored in a sparse column vector.

    Parameters
    ----------
    dim : int
        Truncation dimension of the Fock Hilbert space.
    alpha : complex
        Coherent-state amplitude.
    odd : bool, optional
        If True, construct an odd cat state.
        If False, construct an even cat state.
        Default is False.
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
        Normalized cat-state ket represented as a sparse Qobj.

    Raises
    ------
    ValueError
        If dim is not positive, the odd cat state is requested
        with alpha=0, or the truncated Hilbert space does not
        contain any Fock states of the requested parity.

    Notes
    -----
    The state is normalized within the truncated Fock space
    before applying the sparsity tolerance. Removing small
    coefficients may slightly reduce the final norm.

    Only even or odd Fock components are generated, reducing
    the number of computed amplitudes by approximately half.

    The computational complexity is O(dim), avoiding the
    matrix exponential required by displacement-operator
    based constructions.
    """
    if not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")

    if dim < 1:
        raise ValueError("dim must be positive")
    if odd and alpha == 0:
        raise ValueError("Odd cat is undefined at alpha=0")

    n = np.arange(int(odd), dim, 2)
    if n.size == 0:
        raise ValueError("dim is too small for the requested parity")
    if alpha == 0:
        coeff = np.array([1.0 + 0j])
    else:
        log_amp = n * np.log(abs(alpha)) - 0.5 * gammaln(n + 1)
        coeff = np.exp(log_amp - log_amp.max() + 1j * n * np.angle(alpha))
        coeff /= np.linalg.norm(coeff)

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=int))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)
