"""Two-mode beam splitter with lazy, in-memory, and disk caching."""

from __future__ import annotations

import json
import math
import os
import tempfile
from functools import lru_cache
from hashlib import sha256
from numbers import Real
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array, eye_array, load_npz, save_npz

from optiqs.core.lazy_cached_operator import LazyCachedOperator
from optiqs.core.q_obj import Qobj, Qtype


def beam_splitter(
    dim: int | tuple[int, int],
    theta: float = np.pi / 4,
    phi: float = 0.0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
    *,
    lazy: bool = True,
    precompute: bool = True,
    cache_dir: str | PathLike[str] | None = "~/.cache/optiqs/beam_splitter",
    max_cache_mb: float = 256.0,
) -> LazyCachedOperator | Qobj:
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
        If true (default), return a ``LazyCachedOperator`` that processes
        only the total-photon-number blocks touched by an input state and
        caches frequently used block exponentials in memory. If false,
        return a dense ``Qobj`` for the full unitary.
    precompute : bool, optional
        Construct and cache a block-sparse ``Qobj`` eagerly instead of a
        lazy operator. Results are shared and read-only; call ``.copy()``
        before modifying them. Incompatible with ``lazy=False``.
    cache_dir : str or None, optional
        Directory for persisting precomputed operators. Used only with
        ``precompute=True``; None disables disk caching.
    max_cache_mb : float, optional
        Resident block-exponential cache limit for the lazy operator (MiB).
        With ``precompute=True``, limits estimated final CSR storage instead.
        Peak temporary memory is not included.

    Returns
    -------
    LazyCachedOperator or Qobj
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
    if not isinstance(precompute, (bool, np.bool_)):
        raise TypeError("precompute must be a boolean")
    if precompute and not lazy:
        raise ValueError("precompute=True is incompatible with lazy=False")
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

    if np_dtype.kind == "f" and abs(math.sin(phi)) > 16 * np.finfo(np.float64).eps:
        raise ValueError("a nonzero complex phase requires a complex dtype")

    if precompute:
        return _precomputed_beam_splitter(
            dim_a, dim_b, theta, phi, np_dtype, tol, cache_dir, max_cache_mb
        )

    op = _lazy_beam_splitter(dim_a, dim_b, theta, phi, np_dtype, tol, max_cache_mb)
    return op if lazy else op.to_qobj()


# The public alias is kept alongside the primary factory.
bs = beam_splitter


def _lazy_beam_splitter(
    dim_a: int,
    dim_b: int,
    theta: float,
    phi: float,
    dtype: np.dtype[Any],
    tol: float,
    max_cache_mb: float,
) -> LazyCachedOperator:
    """Build the shared sparse generator and its invariant photon blocks.

    This private factory is also used by the eager/disk-cached path so the
    optical convention and block partition have one implementation.
    """
    numeric_dtype = cast("np.dtype[np.inexact[Any]]", dtype)
    phase: float | complex = math.cos(phi) if dtype.kind == "f" else np.exp(1j * phi)

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
    # Enumerate blocks in O(dim_a * dim_b), without repeatedly scanning the
    # entire basis for each photon-number label.
    blocks = []
    for total in range(dim_a + dim_b - 1):
        a = np.arange(
            max(0, total - dim_b + 1),
            min(total, dim_a - 1) + 1,
            dtype=np.intp,
        )
        blocks.append(a * dim_b + (total - a))

    return LazyCachedOperator(
        Qobj(
            generator,
            dims=((dim_a, dim_b), (dim_a, dim_b)),
            qtype=Qtype.oper,
            tol=tol,
        ),
        scale=theta,
        blocks=blocks,
        max_cache_mb=max_cache_mb,
    )


# Precomputed path: the optical generator and block partition above are the
# sole source of truth for both lazy and eagerly materialized operators.

_VERSION = 1


class _SharedReadOnlyQobj(Qobj):
    """Prevent accidental mutation of an operator shared by the LRU cache.

    ``copy()`` yields a regular, writable Qobj when a caller needs one.
    """

    def __setitem__(self, key: object, value: object) -> None:
        raise TypeError(
            "A precomputed beam splitter is shared and read-only; "
            "call .copy() before editing its elements"
        )


def _memory_estimate(
    dim_a: int, dim_b: int, dtype: np.dtype, *, identity: bool = False
) -> int:
    """Approximate final CSR bytes, before pruning tiny coefficients."""
    # Block sizes are 1, 2, ..., s-1, s (repeated l-s+1 times),
    # s-1, ..., 2, 1 for s=min(dim_a, dim_b), l=max(...).
    s, l = sorted((dim_a, dim_b))
    entries = (
        dim_a * dim_b
        if identity
        else (l - s + 1) * s * s + (s - 1) * s * (2 * s - 1) // 3
    )
    return (
        entries * (dtype.itemsize + np.dtype(np.intp).itemsize)
        + (dim_a * dim_b + 1) * np.dtype(np.intp).itemsize
    )


@lru_cache(maxsize=2)
def _cached_build(
    dim_a: int,
    dim_b: int,
    theta: float,
    phi: float,
    dtype_str: str,
    tol: float,
    cache_dir: str | None,
) -> Qobj:
    """Reuse the precomputed CSR operator, loading or building as necessary."""
    dtype = np.dtype(dtype_str)
    dims = ((dim_a, dim_b), (dim_a, dim_b))
    path: Path | None = None
    if cache_dir is not None:
        key = json.dumps(
            [_VERSION, dim_a, dim_b, theta, phi, dtype_str, tol],
            separators=(",", ":"),
        )
        digest = sha256(key.encode("ascii")).hexdigest()[:24]
        path = Path(cache_dir).expanduser() / f"bs_{dim_a}x{dim_b}_{digest}.npz"
        if path.is_file():
            try:
                loaded = load_npz(path)
                if (
                    loaded.shape == (dim_a * dim_b, dim_a * dim_b)
                    and loaded.dtype == dtype
                ):
                    # load_npz can produce an old-style sparse matrix. Qobj
                    # accepts concrete CSR *arrays*, so normalize explicitly.
                    cached = csr_array(cast(Any, loaded))
                    return _SharedReadOnlyQobj(
                        cached, dims=dims, qtype=Qtype.oper, tol=tol
                    )
            except (OSError, ValueError, TypeError):
                # Corrupted or incompatible data: rebuild.
                pass

    if theta == 0 or min(dim_a, dim_b) == 1:
        # Nothing mixes: don't construct or exponentiate a zero generator.
        sparse_matrix = eye_array(dim_a * dim_b, dtype=dtype, format="csr")
    else:
        # Materialize invariant blocks directly, avoiding a dense identity
        # multiplication and unnecessary construction of the full matrix.
        # The eager path historically applies `tol` to the *result* only.
        # Keep the generator unpruned and prune in _SharedReadOnlyQobj below.
        op = _lazy_beam_splitter(dim_a, dim_b, theta, phi, dtype, 0.0, 0.0)
        sparse_matrix = csr_array(cast(Any, op.to_sparse_qobj()._data), dtype=dtype)
    obj = _SharedReadOnlyQobj(sparse_matrix, dims=dims, qtype=Qtype.oper, tol=tol)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement avoids leaving a partial cache after interruption.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".bs_", suffix=".npz", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
            saved_data = obj._data
            if not isinstance(saved_data, csr_array):
                raise TypeError("cached beam splitter must use SciPy CSR storage")
            save_npz(temporary_path, saved_data)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return obj


def _precomputed_beam_splitter(
    dim_a: int,
    dim_b: int,
    theta: float,
    phi: float,
    dtype: np.dtype,
    tol: float,
    cache_dir: str | os.PathLike[str] | None,
    max_cache_mb: float,
) -> Qobj:
    """Return a shared cached CSR Qobj, subject to the final-storage budget.

    Call ``copy()`` before item assignment. The limit excludes temporary
    memory used to build or load the operator.
    """
    if not np.isfinite(max_cache_mb) or max_cache_mb <= 0:
        raise ValueError("max_cache_mb must be a finite positive number")
    estimated = _memory_estimate(
        dim_a, dim_b, dtype, identity=theta == 0 or min(dim_a, dim_b) == 1
    )
    if estimated > max_cache_mb * 1024 * 1024:
        raise MemoryError(
            f"Precomputed beam splitter estimated at {estimated / (1024**2):.1f} MiB "
            f"(limit {max_cache_mb:.1f} MiB). Use default lazy=True, "
            "or raise max_cache_mb explicitly."
        )
    return _cached_build(
        dim_a,
        dim_b,
        float(theta),
        float(phi),
        dtype.str,
        float(tol),
        os.fspath(cache_dir) if cache_dir is not None else None,
    )
