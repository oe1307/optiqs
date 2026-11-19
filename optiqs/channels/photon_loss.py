"""Exact bosonic photon-loss (pure-loss) channel in the Fock basis.

The channel acts directly on a ``Qobj`` density matrix, without forming
an environment, beam-splitter unitary, or a Liouville-space superoperator.
For a sparse input it never densifies the matrix.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import torch


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


# Avoid rebuilding the output CSR for every Kraus term while keeping
# buffered COO entries bounded for states with many occupied Fock levels.
_SPARSE_LOSS_BATCH_TERMS = 5
_SPARSE_LOSS_BATCH_MAX_ENTRIES = 250_000


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
    # COO construction combines entries from different original photon
    # numbers that map onto the same vacuum output coordinate.
    return csr_array(
        (
            coo.data[same_number].astype(dtype, copy=False),
            (coo.row[same_number] - offset, coo.col[same_number] - offset),
        ),
        shape=data.shape,
    )


def _apply_sparse(data: csr_array, dim: int, stride: int, eta: float) -> csr_array:
    """Shift occupied entries and combine bounded groups of Kraus terms."""
    matrix = data.tocoo()
    rows, cols, values = matrix.row, matrix.col, matrix.data
    size = data.shape[0]
    output_dtype = data.dtype if data.dtype.kind in "fc" else np.float64
    output = csr_array((size, size), dtype=output_dtype)
    if matrix.nnz == 0:
        return output

    row_photons = (rows // stride) % dim
    col_photons = (cols // stride) % dim
    max_lost = min(int(row_photons.max()), int(col_photons.max()))
    log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
    batch_rows: list[np.ndarray] = []
    batch_cols: list[np.ndarray] = []
    batch_values: list[np.ndarray] = []
    batch_entries = 0

    def flush_batch() -> None:
        nonlocal output, batch_entries
        if not batch_rows:
            return
        partial = csr_array(
            (
                np.concatenate(batch_values),
                (np.concatenate(batch_rows), np.concatenate(batch_cols)),
            ),
            shape=(size, size),
        )
        output = output + partial
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
    return output


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


def photon_loss(state: Qobj, eta: float, mode: int = 0) -> Qobj:
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
        A ket, bra, or square density operator. For a ket/bra, ``dm()`` is
        used first (which normalizes the input, as elsewhere in OptiQS).
    eta : float
        Power transmissivity in [0, 1]; 1 = no loss, 0 = complete loss.
    mode : int, optional
        Zero-based subsystem index of the lossy mode (default: 0).

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
    O(dim * D**2), using O(D**2) output memory; for CSR, only occupied
    matrix entries are processed and no dense D x D matrix is allocated.

    Requires a valid physical input density operator. As usual, the Fock
    cutoff must already be high enough to represent the initial state.
    """
    if not isinstance(state, Qobj):
        raise TypeError("state must be an optiqs.Qobj")
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

    density = state.dm() if state.is_vector() else state
    if eta == 1:
        return density.copy()

    dim = dims[mode]
    left = math.prod(dims[:mode])
    right = math.prod(dims[mode + 1 :])
    data = density._data
    if isinstance(data, csr_array):
        result = (
            _apply_zero_sparse(data, dim, right)
            if eta == 0
            else _apply_sparse(data, dim, right, float(eta))
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
    return Qobj(result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol)
