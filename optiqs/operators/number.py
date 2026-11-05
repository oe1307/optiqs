import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import diags_array

from optiqs.core.q_obj import Qobj, Qtype


def number(
    dim: int,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct the number operator ``a.dag() @ a`` in a Fock basis.

    The operator acts on Fock states as ``n |k> = k |k>``. Its diagonal
    entries are ``0, 1, ..., dim - 1``. The matrix is stored in SciPy
    CSR format and requires O(dim) memory.

    Parameters
    ----------
    dim : int
        Number of Fock basis states, including the vacuum. Must be positive.
    dtype : numpy.dtype or dtype-like, optional
        Floating-point or complex output dtype. Defaults to ``np.complex128``.
    tol : float, optional
        Zero-removal tolerance passed to ``Qobj``.

    Returns
    -------
    Qobj
        An operator of shape ``(dim, dim)`` stored as a CSR sparse matrix.

    Raises
    ------
    TypeError
        If ``dim`` is not an integer or ``dtype`` is not floating-point
        or complex.
    ValueError
        If ``dim`` is not positive, or ``tol`` is invalid.

    Notes
    -----
    The number operator is diagonal and Hermitian even in a truncated
    Fock basis. It is equivalent to ``annihilation(dim).dag() @
    annihilation(dim)`` up to floating-point rounding.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")

    np_dtype = np.dtype(dtype)
    if np_dtype.kind not in "fc":
        raise TypeError("dtype must be a floating-point or complex NumPy dtype")

    values = np.asarray(np.arange(dim, dtype=np.float64), dtype=dtype)
    data = diags_array(values, offsets=0, shape=(dim, dim), format="csr")
    return Qobj(
        data,
        dims=((dim,), (dim,)),
        qtype=Qtype.oper,
        tol=tol,
    )


num = number
