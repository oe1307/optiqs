"""Apply a local quantum gate without constructing a full-space operator."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
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


class SparseKetLocalPlan:
    """Compile repeated applications of one CSR gate to sparse multi-mode kets.

    Unlike a generic local contraction, this plan never rearranges the whole
    ket into an intermediate CSR matrix. It stores a CSC *snapshot* of the
    gate and the local-to-global basis offsets; one call scatters only the
    occupied input columns to their output rows. The snapshot means editing
    the original gate after compilation does not silently corrupt the plan.

    This specializes the ket case only. Density operators, bras, lazy gates
    and Torch states continue to use :func:`apply_local`. The plan takes
    O(local_dim + gate.nnz) extra persistent storage, so compile only gates
    whose local Hilbert space fits in memory.

    Parameters
    ----------
    gate : Qobj
        A SciPy CSR backed local operator.
    dims : sequence of int
        Global ket subsystem dimensions, leftmost mode first.
    modes : int or sequence of int
        Selected mode indices in the *gate's* subsystem order.

    Examples
    --------
    >>> plan = SparseKetLocalPlan(bs(8, precompute=True), (8, 8, 8), (0, 2))
    >>> result = plan(tensor(fock(8, 1), vacuum(8), vacuum(8)))
    """

    def __init__(
        self,
        gate: Qobj,
        dims: Sequence[int],
        modes: int | Sequence[int],
    ) -> None:
        if not isinstance(gate, Qobj) or not isinstance(gate._data, sparse.csr_array):
            raise TypeError("gate must be a SciPy CSR-backed Qobj")
        if gate.qtype != Qtype.oper or gate.dims[0] != gate.dims[1]:
            raise ValueError("gate must be a square operator")
        if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes)):
            raise TypeError("dims must be a sequence of positive integers")
        if not dims or any(
            isinstance(d, (bool, np.bool_)) or not isinstance(d, Integral) or d < 1
            for d in dims
        ):
            raise ValueError("dims must contain positive integers")

        self.dims = tuple(int(d) for d in dims)
        self.modes = _selected_modes(modes, len(self.dims))
        local_dims = tuple(self.dims[i] for i in self.modes)
        if gate.dims[0] != local_dims:
            raise ValueError(
                f"gate subsystem dims {gate.dims[0]} do not match "
                f"selected mode dims {local_dims}"
            )
        self._gate = gate._data.tocsc(copy=True)
        self._tol = gate._tol
        self._strides: list[int] = []
        remaining = math.prod(self.dims)
        for dim in self.dims:
            remaining //= dim
            self._strides.append(remaining)
        local_size = math.prod(local_dims)
        local_offsets = np.zeros(local_size, dtype=np.intp)
        for digit, axis in zip(
            np.unravel_index(np.arange(local_size, dtype=np.intp), local_dims),
            self.modes,
        ):
            local_offsets += digit * self._strides[axis]
        self._local_offsets = local_offsets
        # These arrays depend only on the gate, not the incoming sparse ket.
        # Reusing the plan avoids repeating these gathers/differences.
        self._column_counts = np.diff(self._gate.indptr)
        self._scatter_offsets = local_offsets[self._gate.indices]

    def __call__(self, state: Qobj) -> Qobj:
        """Apply the compiled gate; retain the sparse ket representation."""
        if not isinstance(state, Qobj):
            raise TypeError("state must be a Qobj")
        source = state._data
        if state.qtype != Qtype.ket or not isinstance(source, sparse.csr_array):
            raise TypeError("SparseKetLocalPlan requires a SciPy CSR ket")
        if state.dims[0] != self.dims or math.prod(state.dims[1]) != 1:
            raise ValueError("state subsystem dims do not match the compiled plan")

        if source.nnz * 16 <= source.shape[0]:
            rows = np.asarray(
                np.searchsorted(
                    source.indptr[1:],
                    np.arange(source.nnz, dtype=source.indptr.dtype),
                    side="right",
                ),
                dtype=np.intp,
            )
        else:
            rows = np.asarray(
                source.tocoo(copy=False).row,
                dtype=np.intp,
            )

        local = _extract_flat_index(rows, self.dims, self.modes)
        remainder = rows - self._local_offsets[local]
        indptr = self._gate.indptr
        counts = self._column_counts[local]
        parents = np.repeat(np.arange(rows.size, dtype=np.intp), counts)
        # All positions in the relevant CSC columns, without a Python loop
        # over occupied Fock states or a full-space tensor permutation.
        offsets = np.cumsum(counts, dtype=np.intp) - counts
        positions = np.repeat(indptr[local] - offsets, counts) + np.arange(
            parents.size, dtype=np.intp
        )
        output_rows = remainder[parents] + self._scatter_offsets[positions]
        values = source.data[parents] * self._gate.data[positions]
        result = sparse.csr_array(
            cast(Any, (values, (output_rows, np.zeros_like(output_rows)))),
            shape=source.shape,
            dtype=np.result_type(source.dtype, self._gate.dtype),
        )
        return Qobj._from_owned_data(
            result,
            dims=state.dims,
            qtype=Qtype.ket,
            tol=max(state._tol, self._tol),
        )


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

    # The direct CSC scatter wins on sparse kets with small local gates.
    # Bound the temporary output size; a strongly mixing gate or a nearly
    # dense state is better handled by the generic CSR contraction below.
    if (
        state.qtype == Qtype.ket
        and isinstance(state_data, sparse.csr_array)
        and isinstance(gate, Qobj)
        and isinstance(gate_data, sparse.csr_array)
        and state_data.nnz <= 4096
        and math.prod(required) <= 4096
        and gate_data.nnz <= 200_000
        # Bound the expected number of COO entries before allocating them.
        and state_data.nnz * max(1, gate_data.nnz // gate_data.shape[1]) <= 262_144
    ):
        return SparseKetLocalPlan(gate, subsystem_dims, selected)(state)

    if state.qtype == Qtype.bra:
        return _apply_rows(state.dag(), gate, selected).dag()
    if state.qtype == Qtype.ket:
        return _apply_rows(state, gate, selected)
    intermediate = _apply_rows(state, gate, selected)
    return _apply_rows(intermediate.dag(), gate, selected).dag()


def _apply_fused_diagonals(
    state: Qobj,
    operations: list[tuple[Qobj, tuple[int, ...], np.ndarray]],
) -> Qobj:
    """One state traversal for commuting phase gates on arbitrary modes."""
    data = state._data
    if not isinstance(data, (np.ndarray, sparse.csr_array)):
        # Preserve backend mismatch checks from apply_local, including Torch.
        result = state
        for gate, modes, _ in operations:
            result = apply_local(result, gate, modes)
        return result

    # Reduce repeated gates on the same ordered subset in the small local
    # Hilbert space, without ever constructing a full-space diagonal gate.
    groups: dict[tuple[int, ...], np.ndarray] = {}
    for _, modes, diagonal in operations:
        if modes in groups:
            groups[modes] *= diagonal
        else:
            groups[modes] = np.array(diagonal, copy=True)
    dtype = np.result_type(
        data.dtype, *(diagonal.dtype for diagonal in groups.values())
    )

    output: np.ndarray | sparse.csr_array
    if isinstance(data, np.ndarray):
        output = np.array(data, dtype=dtype, copy=True)
        if state.qtype == Qtype.bra:
            indices = np.arange(data.shape[1], dtype=np.intp)
            for modes, diagonal in groups.items():
                local = _extract_flat_index(indices, state.dims[1], modes)
                output *= np.conjugate(diagonal[local])[None, :]
        else:
            indices = np.arange(data.shape[0], dtype=np.intp)
            for modes, diagonal in groups.items():
                local = _extract_flat_index(indices, state.dims[0], modes)
                output *= diagonal[local, None]
            if state.qtype == Qtype.oper:
                cols = np.arange(data.shape[1], dtype=np.intp)
                for modes, diagonal in groups.items():
                    local = _extract_flat_index(cols, state.dims[1], modes)
                    output *= np.conjugate(diagonal[local])[None, :]
    else:
        values = np.array(data.data, dtype=dtype, copy=True)
        if state.qtype == Qtype.bra:
            for modes, diagonal in groups.items():
                local = _extract_flat_index(data.indices, state.dims[1], modes)
                values *= np.conjugate(diagonal[local])
        else:
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
            for modes, diagonal in groups.items():
                local = _extract_flat_index(rows, state.dims[0], modes)
                values *= diagonal[local]
            if state.qtype == Qtype.oper:
                for modes, diagonal in groups.items():
                    local = _extract_flat_index(data.indices, state.dims[1], modes)
                    values *= np.conjugate(diagonal[local])
        output = sparse.csr_array(
            (values, data.indices.copy(), data.indptr.copy()), shape=data.shape
        )

    return Qobj._from_owned_data(
        output,
        dims=state.dims,
        qtype=state.qtype,
        tol=max(state._tol, *(gate._tol for gate, _, _ in operations)),
    )


def apply_local_sequence(
    state: Qobj,
    operations: Iterable[tuple[LocalGate, int | Sequence[int]]],
    *,
    cache_sparse_plans: bool = True,
) -> Qobj:
    """Apply local operations in order, fusing consecutive diagonal phases.

    Consecutive unit-modulus NumPy/CSR diagonal operators commute even when
    they act on different modes. They are contracted in one pass over the
    state. Nonunitary diagonals and non-diagonal gates still follow the
    normal ``apply_local`` path. As with same-mode fusion, changing the order
    of floating-point products may produce tiny rounding differences.

    With ``cache_sparse_plans=True``, repeated non-diagonal CSR gates acting
    on sparse kets reuse one ``SparseKetLocalPlan`` per gate/mode combination
    for the duration of this call. No global cache is retained, and the
    compiled plan uses the gate values at its *first* application. Disable
    this option if you intentionally mutate private ``gate._data`` during
    iteration over ``operations``.

    Examples
    --------
    >>> result = apply_local_sequence(psi, [
    ...     (phase_shift(8, 0.2), 0),
    ...     (phase_shift(8, 0.3), 0),
    ... ])
    """
    if not isinstance(cache_sparse_plans, bool):
        raise TypeError("cache_sparse_plans must be a bool")
    pending: list[tuple[Qobj, tuple[int, ...], np.ndarray]] = []
    plans: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]],
        tuple[Qobj, SparseKetLocalPlan],
    ] = {}

    def apply(current: Qobj, gate: LocalGate, modes: int | Sequence[int]) -> Qobj:
        data = current._data
        if (
            cache_sparse_plans
            and current.qtype == Qtype.ket
            and isinstance(data, sparse.csr_array)
            and isinstance(gate, Qobj)
            and isinstance(gate._data, sparse.csr_array)
            and data.nnz <= 4096
            and gate._data.nnz <= 200_000
        ):
            selected = _selected_modes(modes, len(current.dims[0]))
            required = tuple(current.dims[0][i] for i in selected)
            local_size = math.prod(required)
            if (
                gate.qtype == Qtype.oper
                and gate.dims == (required, required)
                and local_size <= 4096
                and data.nnz * max(1, gate._data.nnz // gate._data.shape[1]) <= 262_144
            ):
                key = (id(gate), selected, current.dims[0])
                entry = plans.get(key)
                if entry is None or entry[0] is not gate:
                    entry = (
                        gate,
                        SparseKetLocalPlan(gate, current.dims[0], selected),
                    )
                    plans[key] = entry
                return entry[1](current)
        return apply_local(current, gate, modes)

    def flush(current: Qobj) -> Qobj:
        if not pending:
            return current
        if len(pending) == 1:
            gate, modes, _ = pending[0]
            result = apply(current, gate, modes)
        else:
            result = _apply_fused_diagonals(current, pending)
        pending.clear()
        return result

    current = state
    for gate, modes in operations:
        # Preserve apply_local's type, dimensions and mode validation on
        # operations that cannot safely join the pending fusion group.
        if not isinstance(gate, Qobj):
            current = flush(current)
            current = apply(current, gate, modes)
            continue
        dims = current.dims[1] if current.qtype == Qtype.bra else current.dims[0]
        selected = _selected_modes(modes, len(dims))
        required = tuple(dims[i] for i in selected)
        diagonal = _diagonal_factors(gate)
        if (
            gate.qtype != Qtype.oper
            or gate.dims != (required, required)
            or diagonal is None
            or not np.all(np.isfinite(diagonal))
            or not np.all(np.abs(np.abs(diagonal) - 1.0) <= 1e-14)
        ):
            current = flush(current)
            current = apply(current, gate, selected)
            continue
        pending.append((gate, selected, diagonal))
    return flush(current)
