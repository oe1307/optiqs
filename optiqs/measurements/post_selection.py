"""Deterministic photon-number post-selection of a chosen subsystem.

A fixed-number herald on an ancillary mode is useful for conditional optical
operations such as photon subtraction.  The measured mode can be removed
without forming a full projector or a partial-trace density matrix.
"""

from __future__ import annotations

from math import sqrt
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj, Qtype
from optiqs.measurements.photon_number import (
    _csr_ket_rows,
    _geometry,
    _normalized_weights,
    _project,
)

if TYPE_CHECKING:
    import torch


def _discard_dims(state: Qobj, mode: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Remove one subsystem, accommodating both vector dimension conventions."""
    left, right = state.dims
    if state.qtype == Qtype.oper:
        remaining = left[:mode] + left[mode + 1 :]
        return remaining, remaining

    if state.qtype == Qtype.ket:
        remaining = left[:mode] + left[mode + 1 :]
        # Kets can have either (1, ..., 1) or a single (1,) on the right.
        if len(right) == len(left):
            right = right[:mode] + right[mode + 1 :]
        return remaining, right

    remaining = right[:mode] + right[mode + 1 :]
    if len(left) == len(right):
        left = left[:mode] + left[mode + 1 :]
    return left, remaining


def _discard_number_mode(
    state: Qobj,
    mode: int,
    dim: int,
    stride: int,
    outcome: int,
    weight: float,
    sparse_ket_rows: np.ndarray | None = None,
) -> Qobj:
    """Contract a fixed Fock outcome, preserving pure states and backends."""
    data = state._data
    normalization = sqrt(weight) if state.is_vector() else weight
    # Shape, unlike .size, is available uniformly on NumPy, CSR, and torch.
    total_size = data.shape[1] if state.qtype == Qtype.bra else data.shape[0]
    lead = total_size // (dim * stride)
    output_size = lead * stride

    result: np.ndarray | csr_array | torch.Tensor
    if isinstance(data, csr_array):
        if state.qtype == Qtype.ket:
            rows = (
                sparse_ket_rows if sparse_ket_rows is not None else _csr_ket_rows(data)
            )
            keep = (rows // stride) % dim == outcome
            kept = rows[keep]
            new_rows = (kept // (dim * stride)) * stride + kept % stride
            result = csr_array(
                (data.data[keep] / normalization, (new_rows, np.zeros_like(new_rows))),
                shape=(output_size, 1),
            )
        elif state.qtype == Qtype.bra:
            keep = (data.indices // stride) % dim == outcome
            kept = data.indices[keep]
            new_cols = (kept // (dim * stride)) * stride + kept % stride
            result = csr_array(
                (data.data[keep] / normalization, (np.zeros_like(new_cols), new_cols)),
                shape=(1, output_size),
            )
        else:
            entries = data.tocoo(copy=False)
            keep = ((entries.row // stride) % dim == outcome) & (
                (entries.col // stride) % dim == outcome
            )
            rows, cols = entries.row[keep], entries.col[keep]
            reduced_rows = (rows // (dim * stride)) * stride + rows % stride
            reduced_cols = (cols // (dim * stride)) * stride + cols % stride
            result = csr_array(
                (entries.data[keep] / normalization, (reduced_rows, reduced_cols)),
                shape=(output_size, output_size),
            )
    elif isinstance(data, np.ndarray):
        if state.is_vector():
            selected = data.reshape(lead, dim, stride)[:, outcome, :].reshape(-1)
            result = (selected / normalization).reshape(
                (1, output_size) if state.qtype == Qtype.bra else (output_size, 1)
            )
        elif data.flags.c_contiguous:
            # Copy only the heralded block, not the full D x D matrix.
            selected = data.reshape(lead, dim, stride, lead, dim, stride)[
                :, outcome, :, :, outcome, :
            ]
            result = (selected / normalization).reshape(output_size, output_size)
        else:
            indices = (
                np.arange(lead, dtype=np.intp)[:, None] * (dim * stride)
                + outcome * stride
                + np.arange(stride, dtype=np.intp)[None, :]
            ).reshape(-1)
            result = data[np.ix_(indices, indices)] / normalization
    else:
        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError("unsupported Qobj data backend")
        # Integer tensors must be promoted before division, just as in _project.
        if not (data.is_floating_point() or data.is_complex()):
            data = data.to(
                torch.float32 if data.device.type == "mps" else torch.float64
            )
        if state.is_vector():
            torch_selected = data.reshape(lead, dim, stride)[:, outcome, :].reshape(-1)
            result = (torch_selected / normalization).reshape(
                (1, output_size) if state.qtype == Qtype.bra else (output_size, 1)
            )
        elif data.is_contiguous():
            torch_selected = data.reshape(lead, dim, stride, lead, dim, stride)[
                :, outcome, :, :, outcome, :
            ]
            result = (torch_selected / normalization).reshape(output_size, output_size)
        else:
            torch_indices = (
                torch.arange(lead, device=data.device)[:, None] * (dim * stride)
                + outcome * stride
                + torch.arange(stride, device=data.device)[None, :]
            ).reshape(-1)
            result = (
                data.index_select(0, torch_indices).index_select(1, torch_indices)
                / normalization
            )

    return Qobj(
        result, dims=_discard_dims(state, mode), qtype=state.qtype, tol=state._tol
    )


def post_select(
    state: Qobj,
    outcome: int,
    mode: int = 0,
    *,
    discard: bool = True,
) -> tuple[Qobj, float]:
    """Condition on exactly ``outcome`` photons being detected in ``mode``.

    Unlike :func:`measure_photon_number`, this function does *not* sample a
    random result.  It returns the normalized conditional state and the
    success probability (calculated relative to the input's norm/trace).
    The input is never modified.

    By default the measured mode is discarded, keeping only the unmeasured
    subsystems in their original order.  A heralded ket/bra stays pure and
    keeps its representation (NumPy, SciPy CSR, or PyTorch); density matrices
    stay density matrices.  With ``discard=False``, the normalized projected
    state retains the measured mode instead.  Discarding the sole mode is
    undefined because no quantum subsystem would remain.

    A zero-probability branch raises ``ValueError`` (there is no normalized
    conditional state).  Only integer, number-resolving Fock outcomes are
    supported; this is not a threshold-detector click model.

    Examples
    --------
    Heralding one photon in the tapped-off mode of a beam splitter implements
    photon subtraction on the signal (up to finite-transmissivity factors)::

        joint = opt.tensor(opt.fock(8, 2), opt.vacuum(8))
        tapped = opt.beam_splitter(8, theta=0.1) @ joint
        conditional, success_probability = opt.post_select(
            tapped, outcome=1, mode=1
        )
    """
    dim, stride = _geometry(state, mode)
    if isinstance(outcome, (bool, np.bool_)) or not isinstance(outcome, Integral):
        raise TypeError("outcome must be an integer photon number")
    if not 0 <= outcome < dim:
        raise ValueError(f"outcome must satisfy 0 <= outcome < {dim}")
    if not isinstance(discard, bool):
        raise TypeError("discard must be a bool")
    subsystems = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    if discard and len(subsystems) == 1:
        raise ValueError("cannot discard the only mode; pass discard=False")

    cached_rows: list[np.ndarray] | None = (
        [] if isinstance(state._data, csr_array) and state.qtype == Qtype.ket else None
    )
    probabilities, raw = _normalized_weights(
        state, dim, stride, sparse_ket_rows=cached_rows
    )
    weight = float(raw[outcome])
    if weight == 0:
        raise ValueError("post-selected outcome has zero probability")

    rows = cached_rows[0] if cached_rows else None
    conditional = (
        _discard_number_mode(state, mode, dim, stride, int(outcome), weight, rows)
        if discard
        else _project(state, dim, stride, int(outcome), weight, rows)
    )
    return conditional, float(probabilities[outcome])


# The explicit name is useful alongside measure_photon_number.
postselect_photon_number = post_select
