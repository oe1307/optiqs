"""Single-mode displacement with a sparse, lazily exponentiated generator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Number, Real
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array, diags_array

from optiqs.core.expm import Expm
from optiqs.core.q_obj import Qobj, Qtype


def displace(
    dim: int,
    alpha: complex = 0.0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
    *,
    lazy: bool = True,
) -> Expm | Qobj:
    """Construct a single-mode displacement operator in a truncated Fock basis.

    The convention is ``D(alpha) = exp(alpha * a.dag() - conj(alpha) * a)``.
    In particular, ``D(alpha) |0>`` approaches the coherent state
    ``|alpha>`` as the Fock cutoff is increased.

    The anti-Hermitian generator is tridiagonal. It is built directly in
    CSR sparse format using O(dim) memory, without forming products of
    ladder operators. By default the result is a lazy
    :class:`~optiqs.core.expm.Expm`: applying it to a state uses
    ``expm_multiply`` rather than materializing a dense displacement matrix.

    Parameters
    ----------
    dim : int
        Positive number of retained Fock levels (including the vacuum).
    alpha : complex, optional
        Finite complex displacement amplitude (default 0).
    dtype : numpy.dtype-like, optional
        Floating-point or complex generator dtype (default complex128).
        Real dtypes are allowed only for real-valued ``alpha``.
    tol : float, optional
        Sparse pruning tolerance, with ``0 <= tol < 1``.
    lazy : bool, optional
        Return a lazy ``Expm`` (default). When false, materialize the
        full dense unitary as a ``Qobj`` (O(dim**2) memory).

    Returns
    -------
    Expm or Qobj
        Displacement operator with subsystem dims ``((dim,), (dim,))``.

    Notes
    -----
    The exponential of the truncated anti-Hermitian generator is exactly
    unitary (up to numerical precision). It is *not* exactly the projection
    of the infinite-dimensional displacement operator onto this space.
    Check results for convergence with increasing ``dim``, especially
    when ``abs(alpha)`` is large relative to the Fock cutoff.

    ``disp`` is an alias for ``displace``.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")

    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Number):
        raise TypeError("alpha must be a real or complex number")
    parameter = complex(alpha)
    if not (math.isfinite(parameter.real) and math.isfinite(parameter.imag)):
        raise ValueError("alpha must be finite")

    if not isinstance(lazy, (bool, np.bool_)):
        raise TypeError("lazy must be a boolean")
    if (
        isinstance(tol, (bool, np.bool_))
        or not isinstance(tol, Real)
        or not math.isfinite(tol)
        or not 0 <= tol < 1
    ):
        raise ValueError("tol must satisfy 0 <= tol < 1 and be finite")

    try:
        np_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "dtype must be a floating-point or complex NumPy dtype"
        ) from exc
    if np_dtype.kind not in "fc" or np_dtype == np.dtype(np.float16):
        raise TypeError("dtype must be a supported floating-point or complex dtype")

    # Normalize the parameter before building the sparse generator, so
    # pruning does not erase small, nonzero displacements. The magnitude
    # is applied separately by Expm.
    strength = abs(parameter)
    if not math.isfinite(strength):
        raise ValueError("alpha magnitude must be finite")
    # Even a finite alpha can overflow the largest generator entry at the
    # Fock cutoff (sqrt(dim - 1) * |alpha|). Avoid silently creating infs.
    if dim > 1 and strength > np.finfo(np_dtype.name).max / math.sqrt(dim - 1):
        raise ValueError("alpha is too large for the requested dim and dtype")
    phase: complex | float = parameter / strength if strength else 1.0
    if np_dtype.kind == "f":
        if parameter.imag != 0:
            raise ValueError("a complex displacement requires a complex dtype")
        phase = 1.0 if parameter.real >= 0 else -1.0

    if dim == 1 or strength == 0:
        generator = csr_array((dim, dim), dtype=np_dtype)
    else:
        coefficients = np.sqrt(np.arange(1, dim, dtype=np.float64))
        # G[n+1, n] = (alpha / |alpha|) * sqrt(n+1)
        # G[n, n+1] = -conj(alpha / |alpha|) * sqrt(n+1).
        diagonals = cast(
            Sequence[Sequence[complex]],
            [phase * coefficients, -np.conjugate(phase) * coefficients],
        )
        numeric_dtype = cast(np.dtype[np.number[Any]], np_dtype)
        generator = diags_array(
            diagonals,
            offsets=[-1, 1],
            shape=(dim, dim),
            format="csr",
            dtype=numeric_dtype,
        )

    op = Expm(
        Qobj(generator, dims=((dim,), (dim,)), qtype=Qtype.oper, tol=tol),
        scale=strength,
    )
    return op if lazy else op.to_qobj()


disp = displace
