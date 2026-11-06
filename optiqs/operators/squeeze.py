"""Single-mode squeezing with a sparse, lazily exponentiated generator."""

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


def squeeze(
    dim: int,
    r: float = 0.0,
    phi: float = 0.0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
    *,
    lazy: bool = True,
    z: complex | None = None,
) -> Expm | Qobj:
    """Construct the single-mode squeezing operator in a truncated Fock basis.

    The convention is ``S(z) = exp[(conj(z) * a**2 - z * (a.dag())**2)/2]``,
    with ``z = r * exp(1j * phi)``.  For ``phi=0``, squeezing the vacuum
    produces a *negative* two-photon amplitude.  A complex ``z`` can also
    be given directly (instead of ``r`` and ``phi``).

    The finite-dimensional anti-Hermitian generator has only the second
    upper and lower diagonals, coupling ``|n>`` to ``|n+2>``.  It requires
    O(dim) space and is constructed without multiplying annihilation
    operators.  By default the returned :class:`~optiqs.core.expm.Expm`
    applies the exponential directly to states with ``expm_multiply``;
    no dense squeezing matrix is built.

    Parameters
    ----------
    dim : int
        Positive number of retained Fock levels (including the vacuum).
    r : float, optional
        Real squeezing strength, which may be negative. Default is 0.
    phi : float, optional
        Squeezing angle in radians. Default is 0.
    dtype : numpy.dtype-like, optional
        Floating or complex dtype for the generator (default complex128).
        Real dtypes work only if the squeezing parameter is real.
    tol : float, optional
        Sparse pruning tolerance, with ``0 <= tol < 1``.
    lazy : bool, optional
        Return a lazy ``Expm`` (default). If false, materialize the full
        dense unitary as a ``Qobj``; this requires O(dim**2) memory.
    z : complex, optional
        Alternative direct complex squeezing parameter. Cannot be combined
        with nonzero ``r`` or ``phi``.

    Returns
    -------
    Expm or Qobj
        Squeezing operator with subsystem dimensions ``((dim,), (dim,))``.

    Notes
    -----
    The exponential of the *truncated generator* is exactly unitary,
    including at the cutoff. However, its action differs from projecting
    the infinite-dimensional squeezing operator onto a finite Fock space.
    For quantitative results, check convergence as ``dim`` increases.

    ``sq`` is an alias for ``squeeze``.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
        raise TypeError("dim must be a positive integer")
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive")

    for label, value in (("r", r), ("phi", phi)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{label} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")

    if z is not None:
        if r != 0 or phi != 0:
            raise ValueError("z cannot be combined with nonzero r or phi")
        if isinstance(z, (bool, np.bool_)) or not isinstance(z, Number):
            raise TypeError("z must be a real or complex number")
        parameter = complex(z)
    else:
        # Reduce the phase before evaluating it, to avoid precision loss for
        # large angles. Keep a real r separate from the complex phase.
        angle = math.remainder(float(phi), 2 * math.pi)
        parameter = float(r) * complex(math.cos(angle), math.sin(angle))

    if not (math.isfinite(parameter.real) and math.isfinite(parameter.imag)):
        raise ValueError("z must be finite")
    strength = abs(parameter)
    if not math.isfinite(strength):
        raise ValueError("squeezing magnitude must be finite")

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

    # z/|z| is a pure phase. For real dtypes, only phases of +/-1
    # generate real matrices (and a zero squeezing strength is always real).
    phase: complex | float = parameter / strength if strength else 1.0
    if np_dtype.kind == "f":
        # sin(pi) is not exactly zero in floating-point arithmetic.
        if abs(phase.imag) > 16 * np.finfo(np.float64).eps:
            raise ValueError("a complex squeezing phase requires a complex dtype")
        phase = 1.0 if phase.real >= 0 else -1.0

    if dim < 3 or strength == 0:
        generator = csr_array((dim, dim), dtype=np_dtype)
    else:
        n = np.arange(1, dim - 1, dtype=np.float64)
        coefficients = 0.5 * np.sqrt(n * (n + 1))
        # G[n+2, n] = -exp(i phi) * sqrt((n+1)(n+2))/2;
        # G[n, n+2] = conj(exp(i phi)) * sqrt((n+1)(n+2))/2.
        # scipy-stubs does not type NumPy arrays as valid diagonal sequences,
        # although SciPy accepts them. The dtype has already been checked to
        # be floating/complex, but Pyright cannot narrow it using `.kind`.
        # These casts do not allocate or convert the O(dim) coefficient arrays.
        diagonals = cast(
            Sequence[Sequence[complex]],
            [-phase * coefficients, np.conjugate(phase) * coefficients],
        )
        numeric_dtype = cast(np.dtype[np.number[Any]], np_dtype)
        generator = diags_array(
            diagonals,
            offsets=[-2, 2],
            shape=(dim, dim),
            format="csr",
            dtype=numeric_dtype,
        )

    op = Expm(
        Qobj(generator, dims=((dim,), (dim,)), qtype=Qtype.oper, tol=tol),
        scale=strength,
    )
    return op if lazy else op.to_qobj()


sq = squeeze
