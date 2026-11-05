import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import diags_array

from optiqs.core.q_obj import Qobj, Qtype


def identity(
    dim: int,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct the identity operator in a finite-dimensional Hilbert space.

    The operator acts as ``I |n> = |n>`` for every basis state.

    Parameters
    ----------
    dim : int
        Dimension of the Hilbert space. Must be positive.
    dtype : numpy.dtype or dtype-like, optional
        Floating-point or complex output dtype.
    tol : float, optional
        Zero-removal tolerance passed to ``Qobj``.

    Returns
    -------
    Qobj
        The identity operator stored as a CSR sparse matrix.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")

    np_dtype = np.dtype(dtype)
    if np_dtype.kind not in "fc":
        raise TypeError("dtype must be a floating-point or complex NumPy dtype")

    values = np.asarray(np.ones(dim, dtype=np.float64), dtype=dtype)
    data = diags_array(values, offsets=0, shape=(dim, dim), format="csr")

    return Qobj(
        data,
        dims=((dim,), (dim,)),
        qtype=Qtype.oper,
        tol=tol,
    )


qeye = identity
