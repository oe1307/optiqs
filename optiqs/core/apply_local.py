"""Apply a local quantum gate without constructing a full-space operator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse

from optiqs.core.expm import Expm
from optiqs.core.lazy_cached_operator import LazyCachedOperator
from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    from torch import Tensor

LocalGate = Qobj | Expm | LazyCachedOperator


def _selected_modes(modes: int | Sequence[int], count: int) -> tuple[int, ...]:
    """Validate the caller's mode order without silently sorting it."""
    if isinstance(modes, (bool, np.bool_)):
        raise TypeError("modes must contain integer indices, not booleans")
    if isinstance(modes, Integral):
        selected: tuple[int, ...] = (int(modes),)
    elif isinstance(modes, Sequence) and not isinstance(modes, (str, bytes)):
        selected = tuple(modes)
    else:
        raise TypeError("modes must be an integer or a sequence of integers")

    if not selected:
        raise ValueError("modes must contain at least one index")
    if any(
        isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in selected
    ):
        raise TypeError("modes must contain integer indices, not booleans")
    indices = tuple(int(i) for i in selected)
    if len(indices) != len(set(indices)):
        raise ValueError("modes must not contain duplicate indices")
    if any(i < 0 or i >= count for i in indices):
        raise IndexError("mode index out of range")
    return indices


def _apply_dense_rows(
    state: Qobj,
    gate: LocalGate,
    selected: tuple[int, ...],
) -> Qobj:
    """Move selected row axes to the front and apply a local gate."""
    row_dims = state.dims[0]
    remaining = tuple(i for i in range(len(row_dims)) if i not in selected)
    permutation = selected + remaining + (len(row_dims),)
    inverse = tuple(permutation.index(i) for i in range(len(permutation)))
    local_dims = tuple(row_dims[i] for i in selected)
    remaining_dims = tuple(row_dims[i] for i in remaining)
    local_size = math.prod(local_dims)
    columns = state._data.shape[1]
    batch_size = math.prod(remaining_dims) * columns

    data = state._data
    packed_data: np.ndarray | Tensor
    if isinstance(data, np.ndarray):
        packed_data = data.reshape(row_dims + (columns,)).transpose(permutation)
        packed_data = packed_data.reshape(local_size, batch_size)
    else:
        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError("Expected a NumPy array or PyTorch tensor")
        packed_data = data.reshape(row_dims + (columns,)).permute(permutation)
        packed_data = packed_data.reshape(local_size, batch_size)

    packed = Qobj(
        packed_data,
        dims=(local_dims, (batch_size,)),
        qtype=Qtype.oper,
        tol=state._tol,
    )
    applied = gate @ packed
    result_data = applied._data
    output: np.ndarray | Tensor
    if isinstance(result_data, np.ndarray):
        output = (
            result_data.reshape(local_dims + remaining_dims + (columns,))
            .transpose(inverse)
            .reshape(data.shape)
        )
    else:
        import torch

        if not isinstance(result_data, torch.Tensor):
            raise TypeError("A dense state requires a dense local gate result")
        output = (
            result_data.reshape(local_dims + remaining_dims + (columns,))
            .permute(inverse)
            .reshape(data.shape)
        )
    return Qobj(
        output,
        dims=state.dims,
        qtype=state.qtype,
        tol=max(state._tol, gate._tol),
    )


def _extract_flat_index(
    global_indices: np.ndarray,
    global_dims: tuple[int, ...],
    axes: tuple[int, ...],
) -> np.ndarray:
    """Map flat indices to a flat index for the ordered subset of axes."""
    flat = np.zeros_like(global_indices)
    stride = math.prod(global_dims)
    strides: list[int] = []
    for dim in global_dims:
        stride //= dim
        strides.append(stride)
    for axis in axes:
        flat = flat * global_dims[axis] + (
            (global_indices // strides[axis]) % global_dims[axis]
        )
    return flat


def _restore_flat_index(
    local: np.ndarray,
    remaining: np.ndarray,
    dims: tuple[int, ...],
    selected: tuple[int, ...],
    other_axes: tuple[int, ...],
) -> np.ndarray:
    """Restore global flat row indices from selected/rest flattened indices."""
    output = np.zeros_like(local)
    stride = math.prod(dims)
    strides: list[int] = []
    for dim in dims:
        stride //= dim
        strides.append(stride)
    for axis in reversed(selected):
        output += (local % dims[axis]) * strides[axis]
        local = local // dims[axis]
    for axis in reversed(other_axes):
        output += (remaining % dims[axis]) * strides[axis]
        remaining = remaining // dims[axis]
    return output


def _apply_sparse_rows(
    state: Qobj,
    gate: LocalGate,
    selected: tuple[int, ...],
) -> Qobj:
    """Group sparse entries by local index; never build a global gate."""
    source = state._data
    if not isinstance(source, sparse.csr_array):
        raise TypeError("Expected a SciPy CSR array")
    row_dims = state.dims[0]
    remaining = tuple(i for i in range(len(row_dims)) if i not in selected)
    local_dims = tuple(row_dims[i] for i in selected)
    local_size = math.prod(local_dims)
    columns = source.shape[1]
    batch_size = math.prod(row_dims[i] for i in remaining) * columns

    coo = source.tocoo(copy=False)
    local_rows = _extract_flat_index(coo.row, row_dims, selected)
    rest_rows = _extract_flat_index(coo.row, row_dims, remaining)
    packed_cols = rest_rows * columns + coo.col
    # The CSR indptr has only local_size + 1 entries. Columns encode the
    # untouched modes and the original matrix columns without allocating
    # any full-space identity or a dense tensor of total dimension.
    packed_sparse = sparse.csr_array(
        cast(Any, (coo.data, (local_rows, packed_cols))),
        shape=(local_size, batch_size),
    )
    packed = Qobj(
        packed_sparse,
        dims=(local_dims, (batch_size,)),
        qtype=Qtype.oper,
        tol=state._tol,
    )
    # A dense *local* Qobj is cheap to convert to CSR relative to the full
    # Hilbert space. This prevents Qobj's mixed-backend matrix product from
    # converting the enormous packed sparse RHS into a dense array.
    if isinstance(gate, Qobj) and isinstance(gate._data, np.ndarray):
        gate = Qobj(
            sparse.csr_array(gate._data),
            dims=gate.dims,
            qtype=Qtype.oper,
            tol=gate._tol,
        )
    applied = gate @ packed
    result_data = applied._data
    if isinstance(result_data, np.ndarray):
        result_coo = sparse.coo_array(result_data)
    elif isinstance(result_data, sparse.csr_array):
        result_coo = result_data.tocoo(copy=False)
    else:
        raise TypeError("A sparse state requires a NumPy/SciPy-backed gate")

    restored_rows = _restore_flat_index(
        result_coo.row,
        result_coo.col // columns,
        row_dims,
        selected,
        remaining,
    )
    restored_cols = result_coo.col % columns
    restored = sparse.csr_array(
        cast(Any, (result_coo.data, (restored_rows, restored_cols))),
        shape=source.shape,
    )
    return Qobj(
        restored,
        dims=state.dims,
        qtype=state.qtype,
        tol=max(state._tol, gate._tol),
    )


def _apply_rows(state: Qobj, gate: LocalGate, selected: tuple[int, ...]) -> Qobj:
    if isinstance(state._data, sparse.csr_array):
        return _apply_sparse_rows(state, gate, selected)
    return _apply_dense_rows(state, gate, selected)


def apply_local(
    state: Qobj,
    gate: LocalGate,
    modes: int | Sequence[int],
) -> Qobj:
    """Apply a gate on specified modes without expanding it to all modes.

    ``modes`` is zero-based, and **its order matters**: a two-mode gate's
    first subsystem acts on ``modes[0]`` and its second on ``modes[1]``.
    ``tensor(a, b, c)`` stores mode 0 as the leftmost (slowest-varying) axis.

    Kets are transformed as ``U @ state``. Density operators are transformed
    as ``U @ state @ U.dag()``. Bras are transformed as ``state @ U.dag()``.
    Operators must have matching row/column subsystem dimensions; the
    function treats any square ``Qtype.oper`` as a density-like operator.
    The gate must be a square ``Qtype.oper`` with subsystem dimensions equal
    to the selected modes, in the supplied order. It is not checked for
    unitarity, so a nonunitary gate acts as ``K rho K.dag()`` on operators.

    Supports NumPy, SciPy CSR and PyTorch ``Qobj`` gates/states and the
    NumPy/SciPy-only lazy ``Expm`` and ``LazyCachedOperator`` gates. A
    NumPy/SciPy gate cannot be mixed with a PyTorch state. Pure NumPy and
    PyTorch states use reshape/transpose of *state data*; SciPy CSR states
    rearrange stored nonzero coordinates. None of these paths constructs
    a Kronecker product with full-space identity operators. Sparse output
    may have more nonzeros after applying a mixing gate, as expected.

    Parameters
    ----------
    state : Qobj
        Multi-mode ket, bra, or density operator.
    gate : Qobj, Expm, or LazyCachedOperator
        Local operator with ``gate.dims == (target_dims, target_dims)``.
    modes : int or sequence of int
        Target mode indices, in the gate's local subsystem order.

    Returns
    -------
    Qobj
        Transformed state with original subsystem dimensions and backend.

    Examples
    --------
    >>> psi = tensor(vacuum(8), fock(8, 1), vacuum(8))
    >>> out = apply_local(psi, displace(8, 0.2), modes=2)
    >>> mixed = apply_local(out, bs(8), modes=(0, 2))
    """
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")
    if not isinstance(gate, (Qobj, Expm, LazyCachedOperator)):
        raise TypeError("gate must be a Qobj, Expm or LazyCachedOperator")
    if gate.qtype != Qtype.oper or gate.dims[0] != gate.dims[1]:
        raise ValueError("gate must be a square operator with matching local dims")
    if state.qtype == Qtype.oper and state.dims[0] != state.dims[1]:
        raise ValueError("operator state must have matching row and column dims")

    subsystem_dims = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    selected = _selected_modes(modes, len(subsystem_dims))
    required = tuple(subsystem_dims[i] for i in selected)
    if gate.dims[0] != required:
        raise ValueError(
            f"gate subsystem dims {gate.dims[0]} do not match "
            f"selected mode dims {required}"
        )

    gate_data = gate._data
    state_data = state._data
    state_torch = not isinstance(state_data, (np.ndarray, sparse.csr_array))
    gate_torch = isinstance(gate, Qobj) and not isinstance(
        gate_data, (np.ndarray, sparse.csr_array)
    )
    if state_torch != gate_torch:
        raise TypeError(
            "Cannot mix PyTorch and NumPy/SciPy data in apply_local; "
            "convert the gate and state to the same backend"
        )

    if state.qtype == Qtype.bra:
        return _apply_rows(state.dag(), gate, selected).dag()
    if state.qtype == Qtype.ket:
        return _apply_rows(state, gate, selected)
    intermediate = _apply_rows(state, gate, selected)
    return _apply_rows(intermediate.dag(), gate, selected).dag()
