"""Memory-efficient, single-mode phase-shift (Fock rotation) operator."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import diags_array

from optiqs.core.q_obj import Qobj, Qtype


def phase_shift(
    dim: int,
    phi: float = 0.0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
) -> Qobj:
    """Construct a phase shifter directly in the truncated Fock basis.

    The convention is ``U(phi) = exp(1j * phi * a.dag() @ a)``, so
    ``U(phi) |n> = exp(1j * n * phi) |n>``. No matrix exponential or dense
    matrix is constructed: the diagonal is computed directly and stored
    as a CSR sparse array. Construction time and memory are both O(dim).

    Parameters
    ----------
    dim : int
        Number of retained Fock states, including the vacuum. Must be
        a positive integer.
    phi : float, optional
        Phase rotation in radians; must be finite. Defaults to zero.
    dtype : numpy.dtype-like, optional
        Complex floating-point output dtype (default ``np.complex128``).
        Real floating-point dtypes are permitted only when the phase
        shift is real (integer multiples of pi).
    tol : float, optional
        Sparse zero-pruning tolerance, in the range ``0 <= tol < 1``.
        This restriction ensures that unit-modulus diagonal entries
        are never pruned away.

    Returns
    -------
    Qobj
        Unitary operator of shape ``(dim, dim)`` with subsystem dimensions
        ``((dim,), (dim,))``, stored in CSR sparse format.

    Notes
    -----
    ``rot`` is an alias for ``phase_shift``.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")

    if isinstance(phi, (bool, np.bool_)) or not isinstance(phi, Real):
        raise TypeError("phi must be a real number")
    if not math.isfinite(phi):
        raise ValueError("phi must be finite")

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

    if np_dtype.kind == "f":
        # A real-valued phase shifter must have exp(i*phi) = +/-1.
        # Computing alternating signs directly avoids round-off imaginary
        # parts in the diagonal at phi = pi.
        if abs(math.sin(phi)) > 16 * np.finfo(np.float64).eps:
            raise ValueError("a non-real phase shift requires a complex dtype")
        diagonal = np.ones(dim, dtype=np_dtype)
        if math.cos(phi) < 0:
            diagonal[1::2] = -1
    else:
        # Reducing the angle limits argument growth for large photon numbers.
        reduced_phi = math.remainder(float(phi), 2 * math.pi)
        photon_numbers = np.arange(dim, dtype=np.float64)
        diagonal = np.exp(1j * (reduced_phi * photon_numbers)).astype(
            np_dtype, copy=False
        )

    # The scalar dtype is validated above, but Pyright cannot infer it from
    # the user-supplied DTypeLike. Narrow only the SciPy stub boundary.
    matrix = diags_array(
        cast("NDArray[Any]", diagonal),
        offsets=0,
        shape=(dim, dim),
        format="csr",
    )
    return Qobj(matrix, dims=((dim,), (dim,)), qtype=Qtype.oper, tol=tol)


rot = phase_shift
