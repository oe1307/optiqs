import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import diags_array

from optiqs.core.q_obj import Qobj, Qtype


def annihilation(
    dim: int,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct the annihilation operator in a truncated Fock basis.

    The operator acts on Fock states as
    a|n> = sqrt(n)|n-1>, with a|0> = 0.

    Parameters
    ----------
    dim : int
        Number of Fock basis states, including the vacuum.
    dtype : numpy.dtype or dtype-like, optional
        Floating-point or complex output dtype.
    tol : float, optional
        Zero-removal tolerance passed to Qobj.

    Returns
    -------
    Qobj
        An operator stored as a CSR sparse matrix.
    """
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        raise ValueError("dim must be a positive integer")
    coefficients = np.sqrt(np.arange(1, dim))
    matrix = diags_array(
        coefficients,
        offsets=1,
        shape=(dim, dim),
        format="csr",
        dtype=np.complex128,
    )
    return Qobj(matrix, dims=((dim,), (dim,)), qtype=Qtype.oper)


destroy = annihilation
