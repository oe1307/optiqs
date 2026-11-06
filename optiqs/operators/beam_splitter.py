"""Two-mode, photon-number-conserving beam-splitter operator.

The default is a lazy exponential of a sparse generator: applying it to a
state does not require storing the full two-mode unitary matrix.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array

from optiqs.core.expm import Expm
from optiqs.core.q_obj import Qobj, Qtype


def beam_splitter(
    dim: int | tuple[int, int],
    theta: float = np.pi / 4,
    phi: float = 0.0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
    *,
    lazy: bool = True,
) -> Expm | Qobj:
    """Construct a lossless beam splitter acting on two Fock modes.

    The convention is

    ``U(theta, phi) = exp[theta * (exp(i*phi) a^dag b
                                    - exp(-i*phi) a b^dag)]``.

    Thus, in a basis containing at least two levels per mode,
    ``U |1,0> = cos(theta)|1,0> - exp(-i*phi)sin(theta)|0,1>``.
    ``theta = pi/4`` gives a 50:50 beam splitter.

    The two-mode basis is ordered ``|n_a, n_b>`` with flattened index
    ``n_a * dim_b + n_b``. The generator only couples adjacent states
    with the same total photon number, so it has at most
    ``2 * (dim_a - 1) * (dim_b - 1)`` nonzero entries. It is constructed
    directly in CSR form, without Kronecker products or dense matrices.

    Parameters
    ----------
    dim : int or tuple[int, int]
        Local Fock cutoffs; a single integer uses the same cutoff for both
        modes. Each local dimension must be positive.
    theta : float, optional
        Mixing angle in radians (``pi/4`` for 50:50).
    phi : float, optional
        Relative phase in radians. Real dtypes require a real phase factor
        (e.g. ``phi=0``); use a complex dtype otherwise.
    dtype : numpy dtype-like, optional
        Floating-point or complex dtype for the sparse generator.
    tol : float, optional
        Non-negative, finite pruning tolerance used by ``Qobj``.
    lazy : bool, optional
        If true (default), return an ``Expm`` that applies the unitary by
        ``scipy.sparse.linalg.expm_multiply`` without materializing it.
        If false, return a dense ``Qobj`` for the full unitary, which takes
        O((dim_a * dim_b)**2) memory.

    Returns
    -------
    Expm or Qobj
        Operator with subsystem dimensions
        ``((dim_a, dim_b), (dim_a, dim_b))``.

    Notes
    -----
    The finite-dimensional generator is anti-Hermitian, so its exponential
    is unitary even at the truncation boundary. However, when photons
    would leave the retained local Fock spaces, the finite-dimensional
    evolution differs from the untruncated optical transformation. Choose
    sufficiently large local cutoffs for your input states.
    """
    if isinstance(dim, tuple):
        if len(dim) != 2:
            raise ValueError("dim must contain exactly two local dimensions")
        dim_a, dim_b = dim
    else:
        dim_a = dim_b = dim

    for local_dim in (dim_a, dim_b):
        if isinstance(local_dim, (bool, np.bool_)) or not isinstance(
            local_dim, (int, np.integer)
        ):
            raise TypeError("local dimensions must be positive integers")
        if local_dim < 1:
            raise ValueError("local dimensions must be positive")
    dim_a, dim_b = int(dim_a), int(dim_b)

    for label, angle in (("theta", theta), ("phi", phi)):
        if isinstance(angle, (bool, np.bool_)) or not isinstance(angle, Real):
            raise TypeError(f"{label} must be a real number")
        if not math.isfinite(angle):
            raise ValueError(f"{label} must be finite")

    if not isinstance(lazy, (bool, np.bool_)):
        raise TypeError("lazy must be a boolean")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, Real)
        or not math.isfinite(tol)
        or tol < 0
    ):
        raise ValueError("tol must be a finite non-negative number")

    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "dtype must be a floating-point or complex NumPy dtype"
        ) from exc
    if np_dtype.kind not in "fc":
        raise TypeError("dtype must be a floating-point or complex NumPy dtype")
    if np_dtype == np.dtype(np.float16):
        raise TypeError("float16 is not supported by SciPy sparse arrays")

    # The kind check above guarantees a floating/complex dtype. Narrowing it
    # here keeps NumPy arithmetic typed as numeric instead of float64 | void.
    numeric_dtype = cast("np.dtype[np.inexact[Any]]", np_dtype)

    if np_dtype.kind == "f":
        if abs(math.sin(phi)) > 16 * np.finfo(np.float64).eps:
            raise ValueError("a nonzero complex phase requires a complex dtype")
        phase: float | complex = math.cos(phi)
    else:
        phase = np.exp(1j * phi)

    # Each pair (n_a, n_b) with n_a < dim_a-1 and n_b > 0 contributes
    # G[|n_a+1,n_b-1>, |n_a,n_b>] = exp(i*phi)*sqrt((n_a+1)*n_b),
    # and the negative conjugate at the transposed entry.
    n_a, n_b = np.meshgrid(
        np.arange(dim_a - 1, dtype=np.intp),
        np.arange(1, dim_b, dtype=np.intp),
        indexing="ij",
    )
    source = (n_a * dim_b + n_b).ravel()
    target = ((n_a + 1) * dim_b + n_b - 1).ravel()
    coefficient = (np.sqrt((n_a + 1) * n_b).ravel() * phase).astype(
        numeric_dtype, copy=False
    )

    rows = np.concatenate((target, source))
    cols = np.concatenate((source, target))
    data = np.concatenate((coefficient, -coefficient.conj()))
    size = dim_a * dim_b
    # scipy-stubs does not reliably recognize the valid COO-style tuple
    # constructor for CSR arrays; isolate the cast at this API boundary.
    generator = csr_array(
        cast(Any, (data, (rows, cols))), shape=(size, size), dtype=numeric_dtype
    )
    op = Expm(
        Qobj(
            generator, dims=((dim_a, dim_b), (dim_a, dim_b)), qtype=Qtype.oper, tol=tol
        ),
        scale=theta,
    )
    return op if lazy else op.to_qobj()


bs = beam_splitter
