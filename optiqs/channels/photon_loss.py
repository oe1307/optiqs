"""Exact bosonic photon-loss (pure-loss) channel in the Fock basis.

The channel acts directly on a ``Qobj`` density matrix or on the Kraus
images of a ket/bra, without forming an environment, beam-splitter unitary,
or a Liouville-space superoperator. Sparse inputs are never densified.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from functools import lru_cache
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.low_rank_density import LowRankDensity
from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import torch


class _BalancedCsrSum:
    """Accumulate bounded sparse blocks in a binary reduction tree.

    Repeated ``output += partial`` scans an ever-growing CSR matrix on each
    batch. A binary carry scheme merges each block at most O(log batches)
    times and holds at most one partial at each level. Duplicate COO indices
    are still summed by the CSR constructor before insertion.
    """

    def __init__(self, shape: tuple[int, int], dtype: np.dtype | type) -> None:
        self._shape = shape
        self._dtype = dtype
        self._levels: list[csr_array | None] = []

    def add(self, part: csr_array) -> None:
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            other = self._levels[level]
            assert other is not None
            part = other + part
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(part)
        else:
            self._levels[level] = part

    def finish(self) -> csr_array:
        result = csr_array(self._shape, dtype=self._dtype)
        for part in reversed(self._levels):
            if part is not None:
                result = result + part
        return result


def _jump_amplitudes(
    dim: int, lost: int, eta: float, log_factorials: np.ndarray
) -> np.ndarray:
    """Return <n-lost|K_lost|n> for n = lost, ..., dim - 1.

    Log-binomial probabilities are stable near eta=0/1 and avoid overflow
    from explicit binomial coefficients/factorials.
    """
    if eta == 0.0:
        result = np.zeros(dim - lost, dtype=np.float64)
        result[0] = 1.0
        return result
    if eta == 1.0:
        return np.ones(dim - lost) if lost == 0 else np.zeros(dim - lost)

    n = np.arange(lost, dim)
    log_probability = (
        log_factorials[n]
        - log_factorials[lost]
        - log_factorials[n - lost]
        + (n - lost) * math.log(eta)
        + lost * math.log1p(-eta)
    )
    return np.exp(0.5 * log_probability)


def _apply_zero_dense_numpy(
    data: np.ndarray, dim: int, left: int, right: int
) -> np.ndarray:
    """At eta=0, replace the selected mode with vacuum after tracing it out."""
    if data.dtype.kind not in "fc":
        data = data.astype(np.float64)
    elif data.dtype == np.float16:
        data = data.astype(np.float32)

    source = data.reshape(left, dim, right, left, dim, right)
    output = np.zeros_like(data)
    # trace over the two selected-mode indices, retaining the full original
    # operator shape and mode ordering. No dim-term Kraus loop is necessary.
    output.reshape(left, dim, right, left, dim, right)[:, 0, :, :, 0, :] = np.trace(
        source, axis1=1, axis2=4
    )
    return output


def _apply_dense_numpy(
    data: np.ndarray, dim: int, left: int, right: int, eta: float
) -> np.ndarray:
    if data.dtype.kind not in "fc":
        data = data.astype(np.float64)
    elif data.dtype == np.float16:
        data = data.astype(np.float32)

    output = np.zeros_like(data)
    source = data.reshape(left, dim, right, left, dim, right)
    target = output.reshape(left, dim, right, left, dim, right)
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    weight_dtype = (
        np.float32 if data.dtype in (np.float32, np.complex64) else np.float64
    )

    for lost in range(dim):
        # The selected dtype is always float32 or float64; make that explicit
        # so NumPy's typed multiplication overload accepts the arrays.
        weights = cast(
            "npt.NDArray[np.floating[Any]]",
            _jump_amplitudes(dim, lost, eta, log_factorials).astype(
                weight_dtype, copy=False
            ),
        )
        weight_matrix = (weights[:, None] * weights[None, :])[
            None, :, None, None, :, None
        ]
        target[:, : dim - lost, :, :, : dim - lost, :] += (
            source[:, lost:, :, :, lost:, :] * weight_matrix
        )
    return output


# Build a bounded set of COO transitions at a time. This is a cap on the
# *generated* transitions, rather than on the number of original CSR entries.
_SPARSE_LOSS_BATCH_MAX_ENTRIES = 250_000
_SPARSE_LOSS_BATCH_TERMS = 32
# Cache at most eight 1-MiB Numba amplitude tables. Larger or very sparse
# problems keep the original log-weight kernel to avoid the setup cost.
_NUMBA_TABLE_MAX_BYTES = 1 << 20


@lru_cache(maxsize=8)
def _cached_numba_loss_amplitudes(levels: int, eta: float) -> np.ndarray:
    """Read-only Kraus amplitudes indexed by [lost, input photons]."""
    log_factorials = gammaln(np.arange(levels, dtype=np.float64) + 1)
    result = np.zeros((levels, levels), dtype=np.float64)
    for lost in range(levels):
        result[lost, lost:] = _jump_amplitudes(levels, lost, eta, log_factorials)
    result.flags.writeable = False
    return result


def _apply_zero_sparse(data: csr_array, dim: int, stride: int) -> csr_array:
    """Pure loss at eta=0 by summing only matching Fock-number entries."""
    coo = data.tocoo(copy=False)
    dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    if coo.nnz == 0:
        return csr_array(data.shape, dtype=dtype)

    row_numbers = (coo.row // stride) % dim
    col_numbers = (coo.col // stride) % dim
    same_number = row_numbers == col_numbers
    offset = row_numbers[same_number] * stride
    return csr_array(
        (
            coo.data[same_number].astype(dtype, copy=False),
            (coo.row[same_number] - offset, coo.col[same_number] - offset),
        ),
        shape=data.shape,
    )


def _apply_sparse_by_lost(
    data: csr_array, dim: int, stride: int, eta: float
) -> csr_array:
    """Shift occupied entries and combine bounded groups of Kraus terms."""
    matrix = data.tocoo()
    rows, cols, values = matrix.row, matrix.col, matrix.data
    size = data.shape[0]
    output_dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    accumulator = _BalancedCsrSum((size, size), output_dtype)
    if matrix.nnz == 0:
        return accumulator.finish()

    row_photons = (rows // stride) % dim
    col_photons = (cols // stride) % dim
    max_lost = min(int(row_photons.max()), int(col_photons.max()))
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    batch_rows: list[np.ndarray] = []
    batch_cols: list[np.ndarray] = []
    batch_values: list[np.ndarray] = []
    batch_entries = 0

    def flush_batch() -> None:
        nonlocal batch_entries
        if not batch_rows:
            return
        partial = csr_array(
            (
                np.concatenate(batch_values),
                (np.concatenate(batch_rows), np.concatenate(batch_cols)),
            ),
            shape=(size, size),
        )
        accumulator.add(partial)
        batch_rows.clear()
        batch_cols.clear()
        batch_values.clear()
        batch_entries = 0

    for lost in range(max_lost + 1):
        eligible = (row_photons >= lost) & (col_photons >= lost)
        if not np.any(eligible):
            continue
        weights = _jump_amplitudes(dim, lost, eta, log_factorials)
        coefficients = (
            values[eligible]
            * weights[row_photons[eligible] - lost]
            * weights[col_photons[eligible] - lost]
        )
        # Skip identically zero weights (notably when eta == 0).
        nonzero = coefficients != 0
        if not np.any(nonzero):
            continue
        selected_rows = rows[eligible][nonzero] - lost * stride
        selected_cols = cols[eligible][nonzero] - lost * stride
        # Some scipy-stubs versions do not accept `dtype` for the
        # COO-style CSR constructor. Convert the values beforehand.
        selected_values = coefficients[nonzero].astype(output_dtype, copy=False)
        batch_rows.append(selected_rows)
        batch_cols.append(selected_cols)
        batch_values.append(selected_values)
        batch_entries += selected_values.size
        # Flush at a term count or entry limit; an individual term can
        # exceed the limit, but is never split into additional copies.
        if (
            len(batch_rows) >= _SPARSE_LOSS_BATCH_TERMS
            or batch_entries >= _SPARSE_LOSS_BATCH_MAX_ENTRIES
        ):
            flush_batch()
    flush_batch()
    return accumulator.finish()


def _apply_sparse(
    data: csr_array, dim: int, stride: int, eta: float, *, accelerator: str = "numpy"
) -> csr_array:
    """Expand occupied (row, col) entries into Kraus transitions once.

    Each entry (n, m) contributes only for lost=0..min(n, m). Generating
    these transitions directly avoids rescanning *every* nonzero for each
    global loss term. COO duplicates are combined by the CSR constructor.
    A prefix-sum partition bounds working memory for large sparse states.
    """
    # With many occupied matrix entries and a small Fock cutoff, the
    # pre-existing per-Kraus-term kernel is cheaper than constructing and
    # exponentiating all individual COO transitions. Preserve that path.
    if accelerator == "numpy" and dim <= 24 and data.nnz >= 4096:
        return _apply_sparse_by_lost(data, dim, stride, eta)

    sparse_loss_kernel = None
    table_kernel = None
    if accelerator == "numba":
        try:
            from optiqs.channels._photon_loss_numba import (
                sparse_loss_transitions,
                sparse_loss_transitions_table,
            )
        except ImportError as exc:
            raise ImportError(
                "The Numba photon-loss accelerator requires 'pip install OptiQS[jit]'"
            ) from exc
        sparse_loss_kernel = sparse_loss_transitions
        table_kernel = sparse_loss_transitions_table

    coo = data.tocoo(copy=False)
    rows, cols, values = coo.row, coo.col, coo.data
    dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    accumulator = _BalancedCsrSum(data.shape, dtype)
    if coo.nnz == 0:
        return accumulator.finish()

    row_photons = (rows // stride) % dim
    col_photons = (cols // stride) % dim
    counts = (np.minimum(row_photons, col_photons) + 1).astype(np.int64)
    cumulative = np.cumsum(counts, dtype=np.int64)
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    log_eta, log_loss = math.log(eta), math.log1p(-eta)
    # A small table replaces per-transition exp/gammaln arithmetic in the
    # JIT kernel. Build it only when the generated transition count is large
    # enough to amortize setup and the table fits the bounded cache.
    amplitudes = None
    if table_kernel is not None:
        levels = int(max(row_photons.max(), col_photons.max())) + 1
        if levels * levels * np.dtype(
            np.float64
        ).itemsize <= _NUMBA_TABLE_MAX_BYTES and int(cumulative[-1]) >= max(
            128, 4 * levels * levels
        ):
            amplitudes = _cached_numba_loss_amplitudes(levels, eta)

    start = 0
    while start < coo.nnz:
        previous = int(cumulative[start - 1]) if start else 0
        # Avoid splitting any one input entry. A single entry may exceed the
        # cap for a very large local Fock cutoff, but all other batches fit.
        stop = max(
            start + 1,
            int(
                np.searchsorted(
                    cumulative,
                    previous + _SPARSE_LOSS_BATCH_MAX_ENTRIES,
                    side="right",
                )
            ),
        )
        if amplitudes is not None and table_kernel is not None:
            out_rows, out_cols, out_values = table_kernel(
                rows[start:stop],
                cols[start:stop],
                values[start:stop].astype(dtype, copy=False),
                row_photons[start:stop],
                col_photons[start:stop],
                counts[start:stop],
                amplitudes,
                stride,
            )
        elif sparse_loss_kernel is not None:
            out_rows, out_cols, out_values = sparse_loss_kernel(
                rows[start:stop],
                cols[start:stop],
                values[start:stop].astype(dtype, copy=False),
                row_photons[start:stop],
                col_photons[start:stop],
                counts[start:stop],
                log_factorials,
                log_eta,
                log_loss,
                stride,
            )
        else:
            sizes = counts[start:stop]
            # Map each Kraus transition back to its occupied source entry.
            parents = np.repeat(np.arange(start, stop, dtype=np.intp), sizes)
            starts = np.repeat(np.cumsum(sizes, dtype=np.int64) - sizes, sizes)
            lost = np.arange(parents.size, dtype=np.int64) - starts
            nr = row_photons[parents]
            nc = col_photons[parents]
            log_weight = 0.5 * (
                log_factorials[nr]
                + log_factorials[nc]
                - 2 * log_factorials[lost]
                - log_factorials[nr - lost]
                - log_factorials[nc - lost]
                + (nr + nc - 2 * lost) * log_eta
                + 2 * lost * log_loss
            )
            contributions = values[parents] * np.exp(log_weight)
            nonzero = contributions != 0
            shift = lost[nonzero] * stride
            out_rows = rows[parents[nonzero]] - shift
            out_cols = cols[parents[nonzero]] - shift
            out_values = contributions[nonzero].astype(dtype, copy=False)
        if out_values.size:
            partial = csr_array(
                (out_values, (out_rows, out_cols)),
                shape=data.shape,
            )
            accumulator.add(partial)
        start = stop
    return accumulator.finish()


def _sparse_diagonal_support(
    data: csr_array, dim: int, stride: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Select a cheap occupied-level path, excluding any coherences.

    Unlike the dense-diagonal kernel, this path never creates a vector of
    length ``data.shape[0]``. Bound the number of generated transitions to
    avoid materializing a very wide distribution in sparse COO buffers.
    """
    if data.nnz == 0 or data.nnz > min(256, max(16, data.shape[0] // 8)):
        return None
    entries = data.tocoo(copy=False)
    active = entries.data != 0
    if np.any((entries.row != entries.col) & active):
        return None
    indices = entries.row[active]
    if int(np.sum((indices // stride) % dim + 1, dtype=np.int64)) > 250_000:
        return None
    return indices, entries.data[active]


def _apply_occupied_diagonal_sparse(
    data: csr_array,
    dim: int,
    stride: int,
    eta: float,
    indices: np.ndarray,
    populations: np.ndarray,
) -> csr_array:
    """Compute binomial loss transitions from only occupied Fock levels."""
    dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    log_eta = math.log(eta)
    log_loss = math.log1p(-eta)
    output_indices: list[np.ndarray] = []
    output_values: list[np.ndarray] = []
    for index, population in zip(indices, populations):
        photons = int((index // stride) % dim)
        lost = np.arange(photons + 1)
        log_probs = (
            log_factorials[photons]
            - log_factorials[lost]
            - log_factorials[photons - lost]
            + (photons - lost) * log_eta
            + lost * log_loss
        )
        output_indices.append(index - lost * stride)
        output_values.append(population * np.exp(log_probs))
    if not output_indices:
        return csr_array(data.shape, dtype=dtype)
    rows = np.concatenate(output_indices)
    values = np.concatenate(output_values).astype(dtype, copy=False)
    # The CSR constructor combines transitions from different parent levels
    # that end at the same photon number; no repeated Kraus CSR addition.
    return csr_array((values, (rows, rows)), shape=data.shape)


def _apply_diagonal_sparse(
    data: csr_array, dim: int, left: int, right: int, eta: float
) -> csr_array:
    """Thin a diagonal density operator without constructing Kraus terms.

    Diagonal operators are classical photon-number distributions.  The
    binomial kernel transforms only the selected Fock axis, so the working
    storage is O(total Hilbert-space dimension), not O(D**2).
    """
    # Imported on this path only: regular photon-loss calls should not pay
    # the import cost of the measurements package.
    from optiqs.measurements.joint_photon_number import _loss_kernel

    dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    populations = np.asarray(data.diagonal()).reshape(left, dim, right)
    # _loss_kernel takes the probability of *losing* a photon, whereas this
    # channel accepts transmissivity (survival probability).
    kernel = _loss_kernel(dim, 1.0 - eta)
    thinned = np.tensordot(kernel, populations, axes=(1, 1))
    flattened = thinned.transpose(1, 0, 2).reshape(-1)
    occupied = np.flatnonzero(flattened)
    return csr_array(
        (
            flattened[occupied].astype(dtype, copy=False),
            (occupied, occupied),
        ),
        shape=data.shape,
    )


def _use_diagonal_fast_path(data: csr_array, dim: int) -> bool:
    """Avoid allocating a large dense diagonal for very sparse inputs.

    Only CSR diagonal entries are inspected.  General inputs remain on the
    coherence-preserving Kraus path; even a single off-diagonal entry makes
    the specialized path inapplicable.
    """
    dimension = data.shape[0]
    if (
        data.nnz == 0
        or data.nnz < max(1, dimension // 8)
        or dim > 384
        or dimension * np.dtype(data.dtype).itemsize > 64 * 1024 * 1024
    ):
        return False
    coo = data.tocoo(copy=False)
    return bool(np.all((coo.row == coo.col) | (coo.data == 0)))


def _apply_zero_dense_torch(
    data: torch.Tensor, dim: int, left: int, right: int
) -> torch.Tensor:
    """Trace out the lost mode on device, preserving autograd."""
    import torch

    if not (data.is_floating_point() or data.is_complex()):
        data = data.to(
            dtype=torch.float32 if data.device.type == "mps" else torch.float64
        )
    elif data.dtype in (torch.float16, torch.bfloat16):
        data = data.to(torch.float32)

    source = data.reshape(left, dim, right, left, dim, right)
    reduced = source.diagonal(dim1=1, dim2=4).sum(dim=-1)
    output = torch.zeros_like(data)
    # CopySlices maintains the gradient connection to reduced (even when
    # the input requires_grad), and never transfers data to the CPU.
    output.reshape(left, dim, right, left, dim, right)[:, 0, :, :, 0, :] = reduced
    return output


def _apply_dense_torch(
    data: torch.Tensor, dim: int, left: int, right: int, eta: float
) -> torch.Tensor:
    import torch

    if not (data.is_floating_point() or data.is_complex()):
        data = data.to(
            dtype=torch.float32 if data.device.type == "mps" else torch.float64
        )
    elif data.dtype in (torch.float16, torch.bfloat16):
        data = data.to(torch.float32)

    source = data.reshape(left, dim, right, left, dim, right)
    result = torch.zeros_like(source)
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    real_dtype = (
        torch.float64
        if data.dtype in (torch.float64, torch.complex128)
        else torch.float32
    )

    # A single host -> device transfer avoids one transfer per Kraus term.
    weights = np.zeros((dim, dim), dtype=np.float64)
    for lost in range(dim):
        weights[lost, lost:] = _jump_amplitudes(dim, lost, eta, log_factorials)
    weights_device = torch.as_tensor(weights, device=data.device, dtype=real_dtype)

    if data.requires_grad:
        # Functional updates preserve autograd; inplace writes into a tensor
        # that becomes a grad-tracked leaf can fail on current PyTorch.
        from torch.nn.functional import pad

        for lost in range(dim):
            row = weights_device[lost, lost:]
            scale = torch.outer(row, row)[None, :, None, None, :, None]
            contribution = source[:, lost:, :, :, lost:, :] * scale
            result = result + pad(contribution, (0, 0, 0, lost, 0, 0, 0, 0, 0, lost))
    else:
        for lost in range(dim):
            row = weights_device[lost, lost:]
            scale = torch.outer(row, row)[None, :, None, None, :, None]
            result[:, : dim - lost, :, :, : dim - lost, :] += (
                source[:, lost:, :, :, lost:, :] * scale
            )
    return result.reshape(data.shape)


def _apply_vector_dense_numpy(
    data: np.ndarray, dim: int, left: int, right: int, eta: float, *, bra: bool
) -> np.ndarray:
    """Pure-state Kraus columns, avoiding the intermediate D-by-D density."""
    ket = data.conj().T if bra else data
    if ket.dtype.kind not in "fc":
        ket = ket.astype(np.float64)
    elif ket.dtype == np.float16:
        ket = ket.astype(np.float32)

    norm = np.linalg.norm(ket)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize an object with zero or invalid norm")
    psi = (ket / norm).reshape(left, dim, right)
    total = left * dim * right

    if eta == 0.0:
        # The lost mode is vacuum in the output. Trace it out directly from
        # the normalized ket, without allocating a Kraus matrix or its dm().
        environment = psi.transpose(1, 0, 2).reshape(dim, left * right)
        reduced = environment.T @ environment.conj()
        result = np.zeros((total, total), dtype=psi.dtype)
        result.reshape(left, dim, right, left, dim, right)[:, 0, :, :, 0, :] = (
            reduced.reshape(left, right, left, right)
        )
        return result

    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    weight_dtype = np.float32 if psi.dtype in (np.float32, np.complex64) else np.float64
    # Contiguous Kraus rows make each slice assignment a view; the last
    # operation is a BLAS matrix multiplication on a (dim, D) buffer.
    columns = np.zeros((dim, total), dtype=psi.dtype)
    for lost in range(dim):
        weights = _jump_amplitudes(dim, lost, eta, log_factorials).astype(
            weight_dtype, copy=False
        )
        columns[lost].reshape(left, dim, right)[:, : dim - lost, :] = (
            psi[:, lost:, :] * weights[None, :, None]
        )
    return columns.T @ columns.conj()


def _apply_vector_sparse(
    data: csr_array, dim: int, stride: int, eta: float, *, bra: bool
) -> csr_array:
    """Use a sparse D-by-dim Kraus matrix, not a sparse D-by-D ket dm().

    Transition coordinates are generated in bounded batches. Accumulating
    the (still sparse) Kraus columns limits temporary COO allocations for
    vectors with large Fock-number support.
    """
    entries = data.tocoo(copy=False)
    positions = entries.col if bra else entries.row
    amplitudes = entries.data.conj() if bra else entries.data
    if amplitudes.dtype.kind not in "fc":
        amplitudes = amplitudes.astype(np.float64)
    elif amplitudes.dtype == np.float16:
        amplitudes = amplitudes.astype(np.float32)

    norm = np.linalg.norm(amplitudes)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Cannot normalize an object with zero or invalid norm")
    amplitudes = amplitudes / norm
    photons = (positions // stride) % dim
    total = max(data.shape)
    output_dtype = amplitudes.dtype

    if eta == 0.0:
        # Each coefficient loses all photons in the selected mode.
        columns = csr_array(
            (amplitudes, (positions - photons * stride, photons)),
            shape=(total, dim),
        )
    else:
        counts = np.asarray(photons + 1, dtype=np.int64)
        cumulative = np.cumsum(counts, dtype=np.int64)
        log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
        log_eta, log_loss = math.log(eta), math.log1p(-eta)
        weight_dtype = (
            np.float32 if output_dtype in (np.float32, np.complex64) else np.float64
        )
        columns = csr_array((total, dim), dtype=output_dtype)
        start = 0
        while start < positions.size:
            previous = int(cumulative[start - 1]) if start else 0
            stop = max(
                start + 1,
                int(
                    np.searchsorted(
                        cumulative,
                        previous + _SPARSE_LOSS_BATCH_MAX_ENTRIES,
                        side="right",
                    )
                ),
            )
            sizes = counts[start:stop]
            parents = np.repeat(np.arange(start, stop, dtype=np.intp), sizes)
            offsets = np.repeat(np.cumsum(sizes, dtype=np.int64) - sizes, sizes)
            lost = np.arange(parents.size, dtype=np.int64) - offsets
            n = photons[parents]
            log_prob = (
                log_factorials[n]
                - log_factorials[lost]
                - log_factorials[n - lost]
                + (n - lost) * log_eta
                + lost * log_loss
            )
            weights = np.exp(0.5 * log_prob).astype(weight_dtype, copy=False)
            values = amplitudes[parents] * weights
            nonzero = values != 0
            if np.any(nonzero):
                partial = csr_array(
                    (
                        values[nonzero],
                        (
                            positions[parents[nonzero]] - lost[nonzero] * stride,
                            lost[nonzero],
                        ),
                    ),
                    shape=(total, dim),
                )
                columns = columns + partial
            start = stop

    return csr_array(columns @ columns.conj().T)


def _apply_vector_dense_torch(
    data: torch.Tensor, dim: int, left: int, right: int, eta: float, *, bra: bool
) -> torch.Tensor:
    """Compute ket/bra Kraus columns on device while retaining autograd."""
    import torch

    ket = data.conj().T if bra else data
    if not (ket.is_floating_point() or ket.is_complex()):
        ket = ket.to(dtype=torch.get_default_dtype())
    elif ket.dtype in (torch.float16, torch.bfloat16):
        ket = ket.to(dtype=torch.float32)

    norm = torch.linalg.vector_norm(ket)
    if not torch.isfinite(norm).item() or norm.abs().item() == 0:
        raise ValueError("Cannot normalize an object with zero or invalid norm")
    psi = (ket / norm).reshape(left, dim, right)
    total = left * dim * right

    if eta == 0.0:
        environment = psi.permute(1, 0, 2).reshape(dim, left * right)
        reduced = environment.T @ environment.conj()
        result = torch.zeros((total, total), dtype=psi.dtype, device=psi.device)
        result.reshape(left, dim, right, left, dim, right)[:, 0, :, :, 0, :] = (
            reduced.reshape(left, right, left, right)
        )
        return result

    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    weights = np.zeros((dim, dim), dtype=np.float64)
    for lost in range(dim):
        weights[lost, lost:] = _jump_amplitudes(dim, lost, eta, log_factorials)
    real_dtype = (
        torch.float64
        if psi.dtype in (torch.float64, torch.complex128)
        else torch.float32
    )
    weights_device = torch.as_tensor(weights, dtype=real_dtype, device=psi.device)
    columns = torch.zeros((dim, total), dtype=psi.dtype, device=psi.device)
    for lost in range(dim):
        columns[lost].reshape(left, dim, right)[:, : dim - lost, :] = (
            psi[:, lost:, :] * weights_device[lost, lost:][None, :, None]
        )
    return columns.T @ columns.conj()


def photon_loss(
    state: Qobj, eta: float, mode: int = 0, *, accelerator: str = "numpy"
) -> Qobj:
    """Apply the exact independent-environment pure-loss channel to one mode.

    ``eta`` is the *transmissivity* (photon survival probability); the loss
    probability is ``1 - eta``. For each number of lost photons ``k``, the
    channel uses the Kraus map

    ``K_k |n> = sqrt(comb(n,k) * eta**(n-k) * (1-eta)**k) |n-k>``.

    The same loss event acts on both indices of the density operator, so
    the map preserves coherences and entanglement with untouched modes.

    Parameters
    ----------
    state : Qobj
        A ket, bra, or square density operator. Ket/bra inputs are normalized
        and processed directly via Kraus columns instead of forming ``dm()``.
    eta : float
        Power transmissivity in [0, 1]; 1 = no loss, 0 = complete loss.
    mode : int, optional
        Zero-based subsystem index of the lossy mode (default: 0).
    accelerator : {'numpy', 'numba'}, optional
        Select the sparse density-matrix transition engine. The default
        ``'numpy'`` preserves the existing behavior. ``'numba'`` compiles a
        bounded COO kernel on first use and requires the optional ``jit``
        extra. Dense states, kets, and diagonal-only sparse states still
        follow their existing specialized paths.

    Returns
    -------
    Qobj
        The *density operator*, even if ``state`` is a ket/bra or eta=1.
        A SciPy CSR input gives a CSR result; NumPy and PyTorch inputs remain
        on their respective backend (PyTorch retains device and autograd).

    Notes
    -----
    Operates directly on Fock-index slices/shifts, without a large
    beam-splitter or superoperator. The work for a dense D x D matrix is
    O(dim * D**2), using O(D**2) output memory; pure states first create
    a D-by-dim Kraus buffer instead of a D-by-D input density matrix. For
    CSR, only occupied entries are processed and no dense D x D matrix is
    allocated.

    Requires a valid physical input density operator. As usual, the Fock
    cutoff must already be high enough to represent the initial state.
    """
    if not isinstance(state, Qobj):
        raise TypeError("state must be an optiqs.Qobj")
    if accelerator not in ("numpy", "numba"):
        raise ValueError("accelerator must be 'numpy' or 'numba'")
    if isinstance(eta, (bool, np.bool_)) or not isinstance(eta, Real):
        raise TypeError("eta must be a real number in [0, 1]")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite and in [0, 1]")
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
        raise TypeError("mode must be an integer subsystem index")

    dims = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    if not 0 <= mode < len(dims):
        raise IndexError("mode is outside the available subsystems")
    if state.qtype == Qtype.oper and state.dims[0] != state.dims[1]:
        raise ValueError("density operator must have matching subsystem dimensions")

    if eta == 1:
        # Keep the exact original dtype/normalization semantics for no loss.
        return (state.dm() if state.is_vector() else state).copy()

    dim = dims[mode]
    left = math.prod(dims[:mode])
    right = math.prod(dims[mode + 1 :])
    if state.is_vector():
        vector_data = state._data
        is_bra = state.qtype == Qtype.bra
        if isinstance(vector_data, csr_array):
            result = _apply_vector_sparse(
                vector_data, dim, right, float(eta), bra=is_bra
            )
        elif isinstance(vector_data, np.ndarray):
            result = _apply_vector_dense_numpy(
                vector_data, dim, left, right, float(eta), bra=is_bra
            )
        else:
            result = _apply_vector_dense_torch(
                vector_data, dim, left, right, float(eta), bra=is_bra
            )
        return Qobj._from_owned_data(
            result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol
        )

    density = state
    data = density._data
    if isinstance(data, csr_array):
        if eta == 0:
            result = _apply_zero_sparse(data, dim, right)
        else:
            support = _sparse_diagonal_support(data, dim, right)
            if support is not None:
                result = _apply_occupied_diagonal_sparse(
                    data, dim, right, float(eta), *support
                )
            elif _use_diagonal_fast_path(data, dim):
                result = _apply_diagonal_sparse(data, dim, left, right, float(eta))
            else:
                result = _apply_sparse(
                    data, dim, right, float(eta), accelerator=accelerator
                )
    elif isinstance(data, np.ndarray):
        result = (
            _apply_zero_dense_numpy(data, dim, left, right)
            if eta == 0
            else _apply_dense_numpy(data, dim, left, right, float(eta))
        )
    else:
        result = (
            _apply_zero_dense_torch(data, dim, left, right)
            if eta == 0
            else _apply_dense_torch(data, dim, left, right, float(eta))
        )
    return Qobj._from_owned_data(
        result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol
    )


def photon_loss_sequence(
    state: Qobj | LowRankDensity,
    losses: Iterable[tuple[float, int]],
    *,
    accelerator: str = "numpy",
) -> Qobj | LowRankDensity:
    """Apply pure-loss channels, combining all transmissivities by mode.

    ``losses`` contains ``(eta, mode)`` pairs where eta is the photon
    *survival* probability. Since pure-loss channels commute on different
    modes and satisfy ``L(a) L(b) = L(a*b)`` on the same mode, each mode is
    processed only once. This is exact for a sequence containing *only*
    loss channels; do not fuse across intervening noncommuting gates.

    ``Qobj`` inputs return a ``Qobj`` density matrix, and ``LowRankDensity``
    inputs retain the low-rank representation. An empty sequence returns a
    copy of the original state. The accelerator affects Qobj CSR inputs only.
    """
    if not isinstance(state, (Qobj, LowRankDensity)):
        raise TypeError("state must be a Qobj or LowRankDensity")
    if accelerator not in ("numpy", "numba"):
        raise ValueError("accelerator must be 'numpy' or 'numba'")
    dims = (
        state.dims
        if isinstance(state, LowRankDensity)
        else state.dims[1]
        if state.qtype == Qtype.bra
        else state.dims[0]
    )
    combined: dict[int, float] = {}
    for eta, mode in losses:
        if isinstance(eta, (bool, np.bool_)) or not isinstance(eta, Real):
            raise TypeError("eta must be a real transmissivity")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
            raise TypeError("mode must be an integer subsystem index")
        if not 0 <= mode < len(dims):
            raise IndexError("mode is outside the available subsystems")
        index = int(mode)
        combined[index] = combined.get(index, 1.0) * float(eta)

    if not combined:
        if isinstance(state, LowRankDensity):
            return LowRankDensity(state.factors, state.dims)
        return state.copy()

    current: Qobj | LowRankDensity = state
    for mode, eta in combined.items():
        current = (
            current.photon_loss(eta, mode)
            if isinstance(current, LowRankDensity)
            else photon_loss(current, eta, mode, accelerator=accelerator)
        )
    return current
