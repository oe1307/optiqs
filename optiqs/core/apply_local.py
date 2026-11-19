"""Apply a local quantum gate without constructing a full-space operator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np
from scipy import sparse

from optiqs.core.expm import Expm
from optiqs.core.lazy_cached_operator import LazyCachedOperator
from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    from torch import Tensor

LocalGate = Union[Qobj, Expm, LazyCachedOperator]


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
    return Qobj._from_owned_data(
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
    # Although Qobj now accepts mixed CSR/NumPy matmul, multiplying a dense
    # local gate by this sparse, very wide RHS yields a dense output. Convert
    # only the small local gate to CSR so the packed result remains sparse.
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
    return Qobj._from_owned_data(
        restored,
        dims=state.dims,
        qtype=state.qtype,
        tol=max(state._tol, gate._tol),
    )


def _diagonal_factors(gate: LocalGate) -> np.ndarray | None:
    """Return diagonal factors for NumPy/CSR Qobj gates, if strictly diagonal.

    Do not interpret a generic exponential as diagonal: checking its generator
    would not guarantee the exponential has the same stored representation.
    """
    if not isinstance(gate, Qobj):
        return None
    data = gate._data
    if isinstance(data, sparse.csr_array):
        entries = data.tocoo(copy=False)
        if np.any((entries.row != entries.col) & (entries.data != 0)):
            return None
        return np.asarray(data.diagonal())
    if isinstance(data, np.ndarray):
        diagonal = np.diagonal(data)
        if not np.array_equal(data, np.diag(diagonal)):
            return None
        return diagonal
    return None


def _apply_dense_diagonal(
    state: Qobj, gate: LocalGate, selected: tuple[int, ...], diagonal: np.ndarray
) -> Qobj | None:
    """Scale dense NumPy entries directly instead of packing/multiplying a gate.

    This also handles nonunitary diagonal Kraus operators (and density
    operators) and does not create a dense full-space diagonal operator.
    """
    data = state._data
    if not isinstance(data, np.ndarray):
        return None
    if (
        state.qtype != Qtype.oper
        and isinstance(gate, Qobj)
        and isinstance(gate._data, sparse.csr_array)
    ):
        # CSR diagonal gate @ dense ket/bra is already exceptionally cheap
        # in SciPy. Avoid allocating and mapping every state index there.
        return None
    output = np.array(data, dtype=np.result_type(data.dtype, diagonal.dtype), copy=True)
    if state.qtype == Qtype.bra:
        local_cols = _extract_flat_index(
            np.arange(data.shape[1], dtype=np.intp), state.dims[1], selected
        )
        output *= np.conjugate(diagonal[local_cols])[None, :]
    else:
        local_rows = _extract_flat_index(
            np.arange(data.shape[0], dtype=np.intp), state.dims[0], selected
        )
        output *= diagonal[local_rows, None]
        if state.qtype == Qtype.oper:
            local_cols = _extract_flat_index(
                np.arange(data.shape[1], dtype=np.intp), state.dims[1], selected
            )
            output *= np.conjugate(diagonal[local_cols])[None, :]
    return Qobj._from_owned_data(
        output,
        dims=state.dims,
        qtype=state.qtype,
        tol=max(state._tol, gate._tol),
    )


def _apply_sparse_diagonal(
    state: Qobj, gate: LocalGate, selected: tuple[int, ...], diagonal: np.ndarray
) -> Qobj | None:
    """Apply diagonal factors to stored CSR entries without rearranging axes.

    Kets receive K, bras receive K.dag(), and operators receive K rho K.dag().
    The gate may be backed by either CSR or a dense NumPy diagonal array.
    """
    data = state._data
    if not isinstance(data, sparse.csr_array):
        return None
    if state.qtype == Qtype.bra:
        indices = _extract_flat_index(data.indices, state.dims[1], selected)
        values = data.data * np.conjugate(diagonal[indices])
    else:
        # Construct row indices only for stored entries, using searchsorted
        # for very sparse states to avoid touching a huge array of empty rows.
        if data.nnz < max(1, data.shape[0] // 100):
            rows = np.searchsorted(
                data.indptr[1:],
                np.arange(data.nnz, dtype=data.indptr.dtype),
                side="right",
            )
        else:
            rows = np.repeat(
                np.arange(data.shape[0], dtype=data.indptr.dtype),
                np.diff(data.indptr),
            )
        row_indices = _extract_flat_index(rows, state.dims[0], selected)
        values = data.data * diagonal[row_indices]
        if state.qtype == Qtype.oper:
            col_indices = _extract_flat_index(data.indices, state.dims[1], selected)
            values *= np.conjugate(diagonal[col_indices])

    result = sparse.csr_array(
        (values, data.indices.copy(), data.indptr.copy()), shape=data.shape
    )
    return Qobj._from_owned_data(
        result,
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

    # Small CSR diagonal @ dense vectors is already a fast sparse/dense
    # product. Skip even the diagonal-inspection overhead on this route.
    if (
        isinstance(state_data, np.ndarray)
        and state.qtype != Qtype.oper
        and isinstance(gate, Qobj)
        and isinstance(gate_data, sparse.csr_array)
    ):
        diagonal = None
    else:
        diagonal = _diagonal_factors(gate)
    if diagonal is not None:
        if isinstance(state_data, np.ndarray):
            result = _apply_dense_diagonal(state, gate, selected, diagonal)
        else:
            result = _apply_sparse_diagonal(state, gate, selected, diagonal)
        if result is not None:
            return result

    if state.qtype == Qtype.bra:
        return _apply_rows(state.dag(), gate, selected).dag()
    if state.qtype == Qtype.ket:
        return _apply_rows(state, gate, selected)
    intermediate = _apply_rows(state, gate, selected)
    return _apply_rows(intermediate.dag(), gate, selected).dag()
