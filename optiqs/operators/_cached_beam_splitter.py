"""Optional sparse, photon-number-block cached beam-splitter implementation.

A two-mode passive beam splitter conserves the total photon number.  Rather
than exponentiating a (dim_a * dim_b) ** 2 matrix, we exponentiate independent
blocks, whose largest size is min(dim_a, dim_b), and embed them in a CSR array.

This path trades memory and one-time construction cost for very fast repeated
applications.  The existing lazy Expm implementation remains the default.
"""

from __future__ import annotations

import json
import os
import tempfile
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_array, load_npz, save_npz

from optiqs.core.q_obj import Qobj, Qtype

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


def _memory_estimate(dim_a: int, dim_b: int, dtype: np.dtype) -> int:
    """Approximate final CSR bytes, before pruning tiny coefficients."""
    entries = 0
    for number in range(dim_a + dim_b - 1):
        count = min(number, dim_a - 1) - max(0, number - dim_b + 1) + 1
        entries += count * count
    return (
        entries * (dtype.itemsize + np.dtype(np.intp).itemsize)
        + (dim_a * dim_b + 1) * np.dtype(np.intp).itemsize
    )


def _photon_block_sparse(
    dim_a: int, dim_b: int, theta: float, phi: float, dtype: np.dtype
) -> csr_array:
    """Construct the unitary by exponentiating total-photon-number blocks."""
    size = dim_a * dim_b
    a_all = np.repeat(np.arange(dim_a), dim_b)
    b_all = np.tile(np.arange(dim_b), dim_a)
    n_all = a_all + b_all
    row_counts = np.minimum(n_all, dim_a - 1) - np.maximum(0, n_all - dim_b + 1) + 1
    row_ptr = np.zeros(size + 1, dtype=np.intp)
    row_ptr[1:] = np.cumsum(row_counts)
    cols = np.empty(int(row_ptr[-1]), dtype=np.intp)
    values = np.empty(int(row_ptr[-1]), dtype=dtype)
    phase = np.exp(1j * phi) if dtype.kind == "c" else np.cos(phi)

    for number in range(dim_a + dim_b - 1):
        first = max(0, number - dim_b + 1)
        last = min(number, dim_a - 1)
        a = np.arange(first, last + 1)
        b = number - a
        indices = a * dim_b + b
        m = len(indices)
        if m == 1:
            block = np.ones((1, 1), dtype=dtype)
        else:
            coeff = np.sqrt((a[:-1] + 1) * b[:-1]) * phase
            generator = np.zeros((m, m), dtype=dtype)
            k = np.arange(m - 1)
            generator[k + 1, k] = theta * coeff
            generator[k, k + 1] = -theta * np.conjugate(coeff)
            block = expm(generator)
        for local_row, global_row in enumerate(indices):
            start, end = int(row_ptr[global_row]), int(row_ptr[global_row + 1])
            cols[start:end] = indices
            values[start:end] = block[local_row]
    return csr_array((values, cols, row_ptr), shape=(size, size))


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
    """Memoize the *operator*, not just the generator (bounded LRU)."""
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

    sparse_matrix = _photon_block_sparse(dim_a, dim_b, theta, phi, dtype)
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


def cached_beam_splitter(
    dim_a: int,
    dim_b: int,
    theta: float,
    phi: float,
    dtype: np.dtype,
    tol: float,
    cache_dir: str | os.PathLike[str] | None,
    max_cache_mb: float,
) -> Qobj:
    """Return a shared read-only-in-practice cached CSR Qobj.

    The returned Qobj disallows item assignment to protect the shared cache.
    Call ``copy()`` to modify it. The memory cap estimates the final CSR size,
    not temporary peak memory needed to build or load it.
    """
    if not np.isfinite(max_cache_mb) or max_cache_mb <= 0:
        raise ValueError("max_cache_mb must be a finite positive number")
    estimated = _memory_estimate(dim_a, dim_b, dtype)
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
