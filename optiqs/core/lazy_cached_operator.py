"""Adaptive, block-aware lazy caching for matrix-exponential operators.

``LazyCachedOperator`` represents ``exp(A)`` without eagerly materializing it.
When invariant subspaces (blocks) of ``A`` are known, it processes only those
blocks touched by the input. The first applications use ``expm_multiply``;
frequently used blocks are replaced by cached dense exponentials under an
LRU memory budget.

The class is operator-agnostic: beam splitters, single-mode squeezers and any
other block-diagonal generator can use it. No conserved quantity is guessed:
callers must supply a valid partition or use the single-block fallback.
"""

from __future__ import annotations

import math
from _thread import RLock as _RLockType
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from threading import RLock
from types import NotImplementedType
from typing import Any, TypeAlias, cast

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from optiqs.core.expm import Expm
from optiqs.core.q_obj import Qobj, Qtype

# Use a concrete two-dimensional CSR type for SciPy stub compatibility.
_MatrixData: TypeAlias = np.ndarray | sparse.csr_array[Any, tuple[int, int]]


@dataclass(frozen=True)
class LazyCacheInfo:
    """Snapshot of per-instance cache statistics and memory usage."""

    hits: int
    misses: int
    evictions: int
    cached_blocks: int
    cached_bytes: int
    max_cache_bytes: int


class LazyCachedOperator:
    """Compute ``exp(A)`` lazily and cache frequently used invariant blocks.

    Parameters
    ----------
    generator : Qobj or Expm
        A square NumPy/SciPy-backed generator Qobj, or an existing Expm.
        Passing Expm preserves its already-applied scale. The input is copied
        to prevent caller mutation from invalidating cached exponentials.
    scale : number
        Additional exponential scale for Qobj generators. Must be 1 when
        wrapping an Expm, since Expm has already incorporated its scale.
    blocks : sequence of 1-D integer index sequences, optional
        A complete, disjoint partition of the generator's basis into
        invariant subspaces. Empty blocks, missing/duplicate indices, or
        cross-block couplings are rejected. Ordering inside each block is
        arbitrary. Without blocks/labels, the entire space is one block.
    block_labels : 1-D integer sequence, optional
        Alternative to ``blocks``: equal labels indicate the same invariant
        block. Example for squeezing: ``np.arange(dim) % 2``. Labels are
        checked for cross-block generator couplings.
    cache_after : int, default 2
        Cache the matrix exponential on this block's Nth active use, provided
        its size fits the limits. Earlier uses call ``expm_multiply``.
    max_cache_mb : float, default 256
        Hard cap on resident cached NumPy matrix buffers (MiB). LRU eviction
        is applied across blocks. Does not include generator/input arrays or
        temporary workspace. Zero disables caching, retaining lazy behavior.
    max_block_dim : int, default 512
        Never materialize an exponential larger than this dimension, even
        when the byte limit would allow it; oversized blocks stay lazy.

    Notes
    -----
    * The class caches *operators*, not outputs tied to a specific state.
    * All outputs match the exponential of the **finite-dimensional** input
      generator. It does not assert unitarity or exploit any symmetry beyond
      a caller-supplied validated block partition.
    * Memory limits apply to persistent cache buffers only; ``expm`` can use
      additional temporary memory when building a new block.
    * Cache is instance-local and memory-only. Disk persistence and automatic
      generation of block labels are deliberately left to operator factories.
    * Right multiplication and adjoints are supported. NumPy/SciPy Qobjs only.
    """

    def __init__(
        self,
        generator: Qobj | Expm,
        *,
        scale: complex = 1.0,
        blocks: Sequence[Sequence[int] | np.ndarray] | None = None,
        block_labels: Sequence[int] | np.ndarray | None = None,
        cache_after: int = 2,
        max_cache_mb: float = 256.0,
        max_block_dim: int = 512,
    ) -> None:
        if isinstance(generator, Qobj):
            base = Expm(generator, scale=scale)
        elif isinstance(generator, Expm):
            # An Expm already contains its scale; do not apply it twice.
            if scale != 1:
                raise ValueError("scale must be 1 when wrapping an existing Expm")
            base = generator
        else:
            raise TypeError("generator must be a Qobj or an Expm")

        if isinstance(cache_after, (bool, np.bool_)) or not isinstance(
            cache_after, Integral
        ):
            raise TypeError("cache_after must be a positive integer")
        if cache_after < 1:
            raise ValueError("cache_after must be at least 1")
        if isinstance(max_block_dim, (bool, np.bool_)) or not isinstance(
            max_block_dim, Integral
        ):
            raise TypeError("max_block_dim must be a positive integer")
        if max_block_dim < 1:
            raise ValueError("max_block_dim must be at least 1")
        if isinstance(max_cache_mb, (bool, np.bool_)) or not isinstance(
            max_cache_mb, Real
        ):
            raise TypeError("max_cache_mb must be a nonnegative finite number")
        if not math.isfinite(max_cache_mb) or max_cache_mb < 0:
            raise ValueError("max_cache_mb must be nonnegative and finite")
        if blocks is not None and block_labels is not None:
            raise ValueError("pass either blocks or block_labels, not both")

        self._data: _MatrixData = base._data.copy()
        self._dims: tuple[tuple[int, ...], tuple[int, ...]] = base.dims
        self._tol: float = base._tol
        self.cache_after: int = int(cache_after)
        self.max_block_dim: int = int(max_block_dim)
        self.max_cache_bytes: int = int(max_cache_mb * 1024 * 1024)
        self._blocks: tuple[np.ndarray, ...] = self._make_blocks(
            self.shape[0], blocks, block_labels
        )
        # Direct look-up avoids visiting every inactive block on each call.
        self._labels: np.ndarray = np.empty(self.shape[0], dtype=np.intp)
        for i, ix in enumerate(self._blocks):
            self._labels[ix] = i
        self._validate_invariant_blocks()
        # The generator's submatrices are sliced only upon first use, not
        # when the wrapper is constructed.
        self._block_data: dict[int, _MatrixData] = {}

        # Cache contains independent read-only NumPy arrays. OrderedDict
        # provides true LRU eviction instead of a FIFO policy.
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._use_count: np.ndarray = np.zeros(len(self._blocks), dtype=np.int64)
        self._cached_bytes: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._lock: _RLockType = RLock()

    @property
    def dims(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._dims

    @property
    def shape(self) -> tuple[int, int]:
        shape = self._data.shape
        return shape[0], shape[1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def qtype(self) -> Qtype:
        return Qtype.oper

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    def is_sparse(self) -> bool:
        """Whether the generator is sparse (not the cached exponentials)."""
        return isinstance(self._data, sparse.csr_array)

    @staticmethod
    def _make_blocks(
        size: int,
        blocks: Sequence[Sequence[int] | np.ndarray] | None,
        block_labels: Sequence[int] | np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        pieces: list[np.ndarray]
        if block_labels is not None:
            labels = np.asarray(block_labels)
            if labels.shape != (size,) or labels.dtype.kind not in "iu":
                raise ValueError("block_labels must be a length-N vector of integers")
            _, inverse = np.unique(labels, return_inverse=True)
            pieces = [
                np.flatnonzero(inverse == k) for k in range(int(inverse.max()) + 1)
            ]
        elif blocks is None:
            pieces = [np.arange(size, dtype=np.intp)]
        else:
            pieces = []
            for part in blocks:
                raw = np.asarray(part)
                if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
                    raise ValueError(
                        "each block must be a nonempty 1-D integer sequence"
                    )
                pieces.append(raw.astype(np.intp, copy=True))

        if not pieces:
            raise ValueError("blocks must not be empty")
        flat = np.concatenate(pieces)
        if len(flat) != size or np.any(flat < 0) or np.any(flat >= size):
            raise ValueError("blocks must cover each basis index exactly once")
        if np.unique(flat).size != size:
            raise ValueError("blocks must not contain duplicate basis indices")
        result = tuple(np.array(part, dtype=np.intp, copy=True) for part in pieces)
        for part in result:
            part.flags.writeable = False
        return result

    def _validate_invariant_blocks(self) -> None:
        """Reject partitions that would silently discard generator couplings."""
        labels = self._labels
        if isinstance(self._data, np.ndarray):
            # Dense generators already occupy O(N**2) memory. Compare only
            # the off-block entries, without allocating a dense N-by-N mask.
            for i, ix in enumerate(self._blocks):
                outside = np.flatnonzero(labels != i)
                if outside.size and np.any(self._data[np.ix_(ix, outside)] != 0):
                    raise ValueError("generator couples different blocks")
        else:
            coo = self._data.tocoo(copy=False)
            if np.any((labels[coo.row] != labels[coo.col]) & (coo.data != 0)):
                raise ValueError("generator couples different blocks")

    def _slice_generator(self, ix: np.ndarray) -> np.ndarray | sparse.csr_array:
        if isinstance(self._data, np.ndarray):
            return self._data[np.ix_(ix, ix)]
        return sparse.csr_array(self._data[ix, :][:, ix])

    def cache_info(self) -> LazyCacheInfo:
        """Return cache hit/miss counts and the current resident matrix bytes."""
        with self._lock:
            return LazyCacheInfo(
                self._hits,
                self._misses,
                self._evictions,
                len(self._cache),
                self._cached_bytes,
                self.max_cache_bytes,
            )

    def clear_cache(self, *, reset_stats: bool = True) -> None:
        """Drop cached exponentials. Optionally reset access counters as well."""
        with self._lock:
            self._cache.clear()
            self._block_data.clear()
            self._cached_bytes = 0
            if reset_stats:
                self._use_count.fill(0)
                self._hits = 0
                self._misses = 0
                self._evictions = 0

    def _block_action(
        self,
        block_id: int,
        rhs: np.ndarray | sparse.csr_array,
        *,
        transpose: bool,
    ) -> np.ndarray | sparse.csr_array:
        with self._lock:
            generator = self._block_data.get(block_id)
            if generator is None:
                generator = self._slice_generator(self._blocks[block_id])
                self._block_data[block_id] = generator
        m = generator.shape[0]
        # Guard persistent memory AND cubic construction costs. This only
        # estimates the final dense output; temporary memory can be larger.
        estimated = m * m * np.dtype(np.result_type(self.dtype, np.float64)).itemsize
        eligible = (
            m <= self.max_block_dim
            and estimated <= self.max_cache_bytes
            and self.max_cache_bytes > 0
        )
        with self._lock:
            matrix = self._cache.get(block_id)
            if matrix is not None:
                self._hits += 1
                self._cache.move_to_end(block_id)
            else:
                self._misses += 1
                self._use_count[block_id] += 1
                if eligible and self._use_count[block_id] >= self.cache_after:
                    # Narrow on ndarray first: SciPy's sparse superclass is
                    # incompletely modelled by some static type checkers.
                    if isinstance(generator, np.ndarray):
                        dense = generator
                    else:
                        dense = generator.toarray()
                    matrix = np.asarray(expm(dense))
                    matrix.flags.writeable = False
                    nbytes = matrix.nbytes
                    if nbytes <= self.max_cache_bytes:
                        while (
                            self._cache
                            and self._cached_bytes + nbytes > self.max_cache_bytes
                        ):
                            _, old = self._cache.popitem(last=False)
                            self._cached_bytes -= old.nbytes
                            self._evictions += 1
                        self._cache[block_id] = matrix
                        self._cached_bytes += nbytes

        # SciPy's sparse-RHS expm_multiply can be much slower than its dense
        # implementation, especially for a sparse ket with only one column.
        # Densify ONLY this small active block; never the full Hilbert space
        # or a wide sparse operator/density matrix.
        if isinstance(rhs, sparse.csr_array) and (
            rhs.shape[0]
            * rhs.shape[1]
            * np.dtype(np.result_type(self.dtype, rhs.dtype, np.float64)).itemsize
            <= 2**20
        ):
            operand: np.ndarray | sparse.csr_array = rhs.toarray()
        else:
            operand = rhs

        if matrix is None:
            action_gen = generator.T if transpose else generator
            # Avoid SciPy's sparse trace estimator; the exact trace is cheap.
            trace = (
                np.sum(generator.diagonal())
                if isinstance(generator, sparse.csr_array)
                else None
            )
            result = expm_multiply(action_gen, operand, traceA=trace)
        else:
            applied = matrix.T if transpose else matrix
            if isinstance(operand, sparse.csr_array):
                # Avoid densifying wide sparse operators/density matrices.
                result = sparse.csr_array(applied) @ operand
            else:
                result = applied @ operand
        if isinstance(result, sparse.sparray):
            return sparse.csr_array(cast(Any, result))
        return np.asarray(result)

    @staticmethod
    def _csr_from_coordinates(
        values: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
        shape: tuple[int, int],
        dtype: np.dtype[Any],
    ) -> sparse.csr_array[Any, tuple[int, int]]:
        """Build a 2-D CSR array, isolated from incomplete SciPy overload stubs."""
        # scipy-stubs currently does not fully describe the valid
        # (values, (row_indices, col_indices)) constructor overload.
        make_csr = cast(
            Callable[..., sparse.csr_array[Any, tuple[int, int]]], sparse.csr_array
        )
        return make_csr((values, (rows, columns)), shape=shape, dtype=dtype)

    def _apply_sparse_wide(
        self,
        rhs: sparse.csr_array[Any, tuple[int, int]],
        active_blocks: np.ndarray,
        out_dtype: np.dtype[Any],
        *,
        transpose: bool,
    ) -> sparse.csr_array[Any, tuple[int, int]]:
        """Preserve sparsity when converting the full RHS would be expensive."""
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for block_id in active_blocks:
            index = self._blocks[int(block_id)]
            part = sparse.csr_array(rhs[index, :])
            computed = self._block_action(int(block_id), part, transpose=transpose)
            coo = sparse.coo_array(computed)
            if coo.nnz:
                rows.append(index[coo.row])
                columns.append(coo.col)
                values.append(coo.data)
        if not values:
            empty = np.empty(0, dtype=out_dtype)
            indices = np.empty(0, dtype=np.intp)
            return self._csr_from_coordinates(
                empty, indices, indices, rhs.shape, out_dtype
            )
        return self._csr_from_coordinates(
            np.concatenate(values),
            np.concatenate(rows),
            np.concatenate(columns),
            rhs.shape,
            out_dtype,
        )

    def _apply_rows(
        self,
        rhs: _MatrixData,
        *,
        transpose: bool,
    ) -> _MatrixData:
        out_dtype = np.dtype(np.result_type(self.dtype, rhs.dtype, np.float64))
        if isinstance(rhs, sparse.csr_array):
            active_rows = np.flatnonzero(np.diff(rhs.indptr))
            active_blocks = np.unique(self._labels[active_rows])
            # A small sparse ket is faster to process as a temporary dense
            # array. Keep wide sparse operators sparse to avoid O(N^2) memory.
            if rhs.shape[0] * rhs.shape[1] * out_dtype.itemsize > 2**20:
                return self._apply_sparse_wide(
                    rhs, active_blocks, out_dtype, transpose=transpose
                )
            source = rhs.toarray()
            sparse_output = True
        else:
            active_rows = np.flatnonzero(np.any(rhs != 0, axis=1))
            active_blocks = np.unique(self._labels[active_rows])
            source = rhs
            sparse_output = False

        output = np.zeros(rhs.shape, dtype=out_dtype)
        for block_id in active_blocks:
            ix = self._blocks[int(block_id)]
            computed = self._block_action(
                int(block_id), source[ix, :], transpose=transpose
            )
            if isinstance(computed, np.ndarray):
                output[ix, :] = computed
            else:
                output[ix, :] = computed.toarray()
        if sparse_output:
            return sparse.csr_array(output)
        return output

    def _apply(self, other: Qobj, *, from_right: bool) -> Qobj:
        if not isinstance(other, Qobj):
            raise TypeError("LazyCachedOperator requires Qobj operands")
        data = other._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("LazyCachedOperator supports only NumPy/SciPy-backed Qobjs")
        if from_right:
            if other.dims[1] != self.dims[0]:
                raise ValueError("Qobj @ LazyCachedOperator dimensions do not match")
            turned = (
                np.asarray(data.T)
                if isinstance(data, np.ndarray)
                else sparse.csr_array(data.T)
            )
            result = self._apply_rows(turned, transpose=True)
            output = (
                result.T
                if isinstance(result, np.ndarray)
                else sparse.csr_array(result.T)
            )
            dims = (other.dims[0], self.dims[1])
        else:
            if self.dims[1] != other.dims[0]:
                raise ValueError("LazyCachedOperator @ Qobj dimensions do not match")
            output = self._apply_rows(data, transpose=False)
            dims = (self.dims[0], other.dims[1])
        return Qobj(
            output, dims=dims, qtype=other.qtype, tol=max(self._tol, other._tol)
        )

    def __matmul__(self, other: object) -> Qobj:
        if not isinstance(other, Qobj):
            raise TypeError("LazyCachedOperator @ requires a Qobj")
        return self._apply(other, from_right=False)

    def __rmatmul__(self, other: object) -> Qobj | NotImplementedType:
        if not isinstance(other, Qobj):
            return NotImplemented
        return self._apply(other, from_right=True)

    def __mul__(self, other: object) -> Qobj:
        return self.__matmul__(other)

    def __rmul__(self, other: object) -> Qobj | NotImplementedType:
        return self.__rmatmul__(other)

    def _derived(self, data: _MatrixData) -> LazyCachedOperator:
        # Avoid running through Qobj's tolerance-pruning constructor again.
        base = object.__new__(Expm)
        base._data = data.copy()
        base._dims = self._dims
        base._tol = self._tol
        return LazyCachedOperator(
            base,
            blocks=self._blocks,
            cache_after=self.cache_after,
            max_cache_mb=self.max_cache_bytes / 1024**2,
            max_block_dim=self.max_block_dim,
        )

    def dag(self) -> LazyCachedOperator:
        """Lazy adjoint, ``exp(A).dag() == exp(A.conj().T)``."""
        return self._derived(
            self._data.conj().T
            if isinstance(self._data, np.ndarray)
            else sparse.csr_array(self._data.conj().T)
        )

    def inverse(self) -> LazyCachedOperator:
        """Lazy inverse of the exponential, ``exp(-A)``."""
        return self._derived(-self._data)

    def _dense_block(self, block_id: int) -> np.ndarray:
        """Materialize one invariant block, reusing an existing cache entry.

        Explicit materialization does not populate the application cache:
        keeping every block *and* the assembled operator would duplicate
        potentially large amounts of memory.
        """
        with self._lock:
            cached = self._cache.get(block_id)
            generator = self._block_data.get(block_id)
        if cached is not None:
            return cached
        if generator is None:
            generator = self._slice_generator(self._blocks[block_id])
        dense = generator if isinstance(generator, np.ndarray) else generator.toarray()
        return np.asarray(expm(dense))

    def to_sparse_qobj(self) -> Qobj:
        """Materialize the exponential as a block-sparse CSR operator.

        Only the supplied invariant blocks are exponentiated. Storage is
        O(sum(block_size**2)), rather than O(full_dimension**2). This is an
        explicit, potentially expensive operation; unlike the lazy action,
        it constructs every block, including those not yet used by a state.
        """
        size = self.shape[0]
        row_counts = np.empty(size, dtype=np.intp)
        for ix in self._blocks:
            row_counts[ix] = ix.size
        indptr = np.empty(size + 1, dtype=np.intp)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])

        indices = np.empty(int(indptr[-1]), dtype=np.intp)
        values = np.empty(int(indptr[-1]), dtype=np.result_type(self.dtype, np.float32))
        for block_id, ix in enumerate(self._blocks):
            matrix = self._dense_block(block_id)
            for local_row, global_row in enumerate(ix):
                start = int(indptr[global_row])
                end = start + ix.size
                indices[start:end] = ix
                values[start:end] = matrix[local_row]

        result = sparse.csr_array((values, indices, indptr), shape=self.shape)
        result.sort_indices()
        return Qobj(result, dims=self.dims, qtype=Qtype.oper, tol=self._tol)

    def to_qobj(self) -> Qobj:
        """Materialize the dense exponential using its invariant blocks."""
        dense = np.zeros(self.shape, dtype=np.result_type(self.dtype, np.float32))
        for block_id, ix in enumerate(self._blocks):
            dense[np.ix_(ix, ix)] = self._dense_block(block_id)
        return Qobj(dense, dims=self.dims, qtype=Qtype.oper, tol=self._tol)

    def __repr__(self) -> str:
        info = self.cache_info()
        return (
            f"LazyCachedOperator(dims={self.dims!r}, blocks={self.block_count}, "
            f"cached={info.cached_blocks}, cache_bytes={info.cached_bytes})"
        )
