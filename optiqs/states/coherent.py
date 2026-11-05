import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def coherent(
    dim: int,
    alpha: complex,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """
    Construct a normalized coherent state in the Fock basis.

    The coherent state is defined as:

        |alpha> = exp(-|alpha|^2 / 2)
                  * sum_n alpha^n / sqrt(n!) |n>

    Fock coefficients are computed directly without evaluating
    the matrix exponential of the displacement operator.

    Parameters
    ----------
    dim : int
        Truncation dimension of the Fock Hilbert space.
    alpha : complex
        Coherent-state amplitude.
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
        Normalized coherent-state ket represented as a sparse Qobj.

    Raises
    ------
    ValueError
        If dim is not positive.

    Notes
    -----
    The state is normalized within the truncated Fock space
    before applying the sparsity tolerance. Removing small
    coefficients may slightly reduce the final norm.

    Logarithmic coefficients are used to improve numerical
    stability when evaluating factorial terms.

    The computational complexity is O(dim), avoiding the
    matrix exponential required by displacement-operator
    based constructions.
    """
    if not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")

    if dim < 1:
        raise ValueError("dim must be positive")

    n = np.arange(dim)

    if alpha == 0:
        coeff = np.array([1.0 + 0j])
        n = n[:1]
    else:
        log_amp = n * np.log(abs(alpha)) - 0.5 * gammaln(n + 1)
        coeff = np.exp(log_amp - log_amp.max() + 1j * n * np.angle(alpha))
        coeff /= np.linalg.norm(coeff)

    data = csr_array(
        (coeff, (n, np.zeros(n.size, dtype=int))),
        shape=(dim, 1),
    )
    return Qobj(data, dtype=dtype, tol=tol)
