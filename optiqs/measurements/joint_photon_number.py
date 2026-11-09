"""RNG-free simultaneous photon-number measurements of two or more modes.

The subsystem order and tensor convention are those of :class:`optiqs.Qobj`.
Computations use basis indices, not Kronecker-product projectors or partial
traces.  All three Qobj backends (NumPy, SciPy CSR, PyTorch) are supported.

A deterministic *measurement* means either computing the joint Born
probabilities, conditioning on a user-specified result, enumerating every
possible nonzero branch, or applying the non-selective measurement channel.
It does NOT mean predicting a random single-shot photon count.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from math import prod, sqrt
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj, Qtype
from optiqs.measurements.photon_number import (
    _checked_diagonal_numpy,
    _checked_diagonal_torch,
    _csr_ket_rows,
    _geometry,
    _squared_magnitudes_numpy,
)

if TYPE_CHECKING:
    import torch

__all__ = [
    "joint_photon_number_branches",
    "joint_photon_number_probabilities",
    "measure_joint_nonselective",
    "measure_joint_photon_number",
    "post_select_joint",
]


def _modes_and_dims(
    state: Qobj, modes: Sequence[int] | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate selected modes and retain their specified order."""
    # Also validates Qobj type and density-operator subsystem geometry.
    _geometry(state, 0)
    dims = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    if modes is None:
        selected = tuple(range(len(dims)))
    else:
        if isinstance(modes, (str, bytes)):
            raise TypeError("modes must be a sequence of integer indices")
        try:
            selected = tuple(modes)
        except TypeError as exc:
            raise TypeError("modes must be a sequence of integer indices") from exc
    if len(selected) < 2:
        raise ValueError("at least two distinct modes must be measured")
    for mode in selected:
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
            raise TypeError("each mode must be an integer subsystem index")
        if not 0 <= mode < len(dims):
            raise IndexError("mode is outside the available subsystems")
    if len(set(selected)) != len(selected):
        raise ValueError("modes must not contain duplicates")
    return dims, tuple(int(mode) for mode in selected)


def _outcomes(
    outcomes: Sequence[int], modes: tuple[int, ...], dims: tuple[int, ...]
) -> tuple[int, ...]:
    if isinstance(outcomes, (str, bytes)):
        raise TypeError("outcomes must be a sequence of integer photon counts")
    try:
        values = tuple(outcomes)
    except TypeError as exc:
        raise TypeError("outcomes must be a sequence of integer photon counts") from exc
    if len(values) != len(modes):
        raise ValueError("outcomes must have one photon number per measured mode")
    for mode, number in zip(modes, values):
        if isinstance(number, (bool, np.bool_)) or not isinstance(number, Integral):
            raise TypeError("each outcome must be an integer photon number")
        if not 0 <= number < dims[mode]:
            raise ValueError(
                f"outcome for mode {mode} must satisfy 0 <= n < {dims[mode]}"
            )
    return tuple(int(number) for number in values)


def _strides(dims: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(dims)
    for pos in range(len(dims) - 2, -1, -1):
        strides[pos] = strides[pos + 1] * dims[pos + 1]
    return tuple(strides)


def _flat_outcomes(
    indices: np.ndarray, dims: tuple[int, ...], modes: tuple[int, ...]
) -> np.ndarray:
    """Pack selected Fock digits into mixed-radix integer labels."""
    strides = _strides(dims)
    if len(modes) == 2:
        # The most common measurement: avoid a general mixed-radix loop and
        # the intermediate zero-filled labels array. Both axes retain the
        # order requested by the caller (also for unsorted modes).
        first, second = modes
        return ((indices // strides[first]) % dims[first]) * dims[second] + (
            (indices // strides[second]) % dims[second]
        )
    labels = np.zeros(indices.shape, dtype=np.intp)
    for mode in modes:
        labels = labels * dims[mode] + (indices // strides[mode]) % dims[mode]
    return labels


def _reduce_numpy(
    weights: np.ndarray, dims: tuple[int, ...], modes: tuple[int, ...]
) -> np.ndarray:
    """Sum out unmeasured tensor axes without dense projectors."""
    if len(modes) == 2:
        # Dedicated 2-mode contraction. A single contiguous reduction is
        # preferable to several successive axis reductions for many layouts.
        # No full D x D projector or density matrix is constructed.
        first, second = modes
        if len(dims) == 2:
            return weights.reshape(dims).transpose(modes).astype(np.float64, copy=False)
        unmeasured = tuple(i for i in range(len(dims)) if i not in modes)
        arranged = weights.reshape(dims).transpose((first, second, *unmeasured))
        return arranged.reshape(dims[first], dims[second], -1).sum(
            axis=-1, dtype=np.float64
        )
    unmeasured = tuple(i for i in range(len(dims)) if i not in modes)
    reduced = weights.reshape(dims)
    if unmeasured:
        reduced = reduced.sum(axis=unmeasured, dtype=np.float64)
    sorted_modes = tuple(sorted(modes))
    if modes != sorted_modes:
        reduced = reduced.transpose(tuple(sorted_modes.index(i) for i in modes))
    return np.asarray(reduced, dtype=np.float64)


def _reduce_torch(
    weights: torch.Tensor, dims: tuple[int, ...], modes: tuple[int, ...]
) -> torch.Tensor:
    """Fuse all marginalizations in one PyTorch kernel on the original device."""
    reduced = weights.reshape(dims)
    unmeasured = tuple(mode for mode in range(len(dims)) if mode not in modes)
    if unmeasured:
        # torch.sum(dim=tuple) avoids O(number of unmeasured modes) separate
        # reductions and intermediate tensors. Especially relevant at 8-16 modes.
        reduced = reduced.sum(dim=unmeasured)
    sorted_modes = tuple(sorted(modes))
    if modes != sorted_modes:
        reduced = reduced.permute(tuple(sorted_modes.index(i) for i in modes))
    return reduced


def _sparse_basis_weights(state: Qobj) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero amplitudes or density diagonals, with their flat basis indices.

    A dense diagonal may be materialized for a high-occupancy sparse operator;
    its size is O(D), NEVER the full D x D density matrix.
    """
    data = state._data
    assert isinstance(data, csr_array)
    if state.qtype == Qtype.ket:
        return _csr_ket_rows(data), _numpy_vector_weights(data.data)
    if state.qtype == Qtype.bra:
        return data.indices, _numpy_vector_weights(data.data)
    if data.nnz >= 4 * data.shape[0]:
        diagonal = _checked_diagonal_numpy(np.asarray(data.diagonal()))
        return np.arange(data.shape[0], dtype=np.intp), np.asarray(
            diagonal, dtype=np.float64
        )
    entries = data.tocoo(copy=False)
    on_diagonal = entries.row == entries.col
    return (
        entries.row[on_diagonal],
        np.asarray(
            _checked_diagonal_numpy(entries.data[on_diagonal]), dtype=np.float64
        ),
    )


def _torch_coefficients(data: torch.Tensor) -> torch.Tensor:
    import torch

    if data.dtype in (torch.float16, torch.bfloat16):
        return data.to(torch.float32)
    if not (data.is_floating_point() or data.is_complex()):
        return data.to(torch.float32 if data.device.type == "mps" else torch.float64)
    return data


def _torch_vector_weights(data: torch.Tensor) -> torch.Tensor:
    import torch

    coeff = _torch_coefficients(data)
    if coeff.device.type != "mps" and coeff.dtype in (torch.float32, torch.complex64):
        # Promote BEFORE squaring, not only after seeing inf: tiny float32
        # squares can silently underflow to 0 or become inaccurate subnormals.
        # The probability API returns float64 in any case. MPS lacks float64.
        coeff = coeff.to(torch.complex128 if coeff.is_complex() else torch.float64)
    return coeff.reshape(-1).abs().square()


def _safe_numpy_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
    """Promote only when low-precision outer products risk overflow/underflow.

    Unlike normalized post-selection, a non-selective measurement *must*
    preserve the original squared norm. Small complex64 coefficients need a
    complex128 output to represent that trace (e.g. 1e-48).
    """
    if amplitudes.dtype.kind not in "fc":
        return amplitudes.astype(np.float64)
    if amplitudes.dtype == np.float16:
        return amplitudes.astype(np.float64)
    if amplitudes.dtype in (np.dtype("float32"), np.dtype("complex64")):
        magnitude = np.abs(amplitudes)
        if np.max(magnitude, initial=0) > 1e18 or np.any(
            (magnitude > 0) & (magnitude < 1e-18)
        ):
            return amplitudes.astype(
                np.complex128 if np.iscomplexobj(amplitudes) else np.float64
            )
    return amplitudes


def _safe_torch_amplitudes(amplitudes: torch.Tensor) -> torch.Tensor:
    """Use float64 outer products only for numerically sensitive vectors."""
    import torch

    amplitudes = _torch_coefficients(amplitudes)
    if amplitudes.device.type == "mps":
        # MPS does not support double precision. Normal values keep the device
        # and dtype, while out-of-range values cannot be safely represented.
        return amplitudes
    if amplitudes.dtype in (torch.float32, torch.complex64):
        magnitude = amplitudes.abs()
        if (magnitude.amax() > 1e18).item() or (
            ((magnitude > 0) & (magnitude < 1e-18)).any()
        ).item():
            return amplitudes.to(
                torch.complex128 if amplitudes.is_complex() else torch.float64
            )
    return amplitudes


def _numpy_vector_weights(values: np.ndarray) -> np.ndarray:
    """Compute complex64 Born weights without losing subnormal components."""
    if values.dtype == np.complex64:
        # np.abs(complex64) itself rounds to float32. Compute the magnitude
        # from complex128 instead, before the stable float64 squaring step.
        values = values.astype(np.complex128)
    return _squared_magnitudes_numpy(values)


def _safe_numpy_dividend(values: np.ndarray, denominator: float) -> np.ndarray:
    """Avoid NumPy rounding a tiny/large Python scalar to an unusable float32."""
    if values.dtype in (
        np.dtype("float16"),
        np.dtype("float32"),
        np.dtype("complex64"),
    ):
        # A dtype string avoids type-variable conflicts in NumPy's finfo stubs.
        real_dtype = "float16" if values.dtype == np.float16 else "float32"
        limits = np.finfo(real_dtype)
        if denominator < limits.tiny or denominator > limits.max:
            return values.astype(
                np.complex128 if np.iscomplexobj(values) else np.float64
            )
    return values


def _numpy_needs_precision_promotion(dtype: np.dtype, denominator: float) -> bool:
    """Cheap fast-path predicate for a normalizing real scalar."""
    if dtype == np.dtype("float16"):
        real_dtype = "float16"
    elif dtype in (np.dtype("float32"), np.dtype("complex64")):
        real_dtype = "float32"
    else:
        return False
    limits = np.finfo(real_dtype)
    return bool(denominator < float(limits.tiny) or denominator > float(limits.max))


def _numpy_needs_stable_division(
    dtype: np.dtype, denominator: float, scale: float
) -> bool:
    return (
        scale != 1.0
        or _numpy_needs_precision_promotion(dtype, denominator)
        or (dtype.kind == "c" and (denominator < 1e-150 or denominator > 1e150))
    )


def _numpy_normal_fast_path(dtype: np.dtype, denominator: float, scale: float) -> bool:
    """Avoid repeated type/precision probes on the common double-precision path."""
    # NumPy dtype comparisons may be typed as np.bool_; this API returns bool.
    return bool(
        scale == 1.0
        and (
            dtype == np.float64
            or (dtype == np.complex128 and 1e-150 <= denominator <= 1e150)
        )
    )


def _safe_torch_dividend(values: torch.Tensor, denominator: float) -> torch.Tensor:
    """Avoid low-precision division by a scalar rounded to zero or infinity."""
    import torch

    if values.device.type == "mps":
        return values
    if values.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        limits = torch.finfo(
            torch.float16 if values.dtype == torch.float16 else torch.float32
        )
        if denominator < limits.tiny or denominator > limits.max:
            return values.to(torch.complex128 if values.is_complex() else torch.float64)
    return values


def _divide_numpy_real(values: np.ndarray, denominator: float) -> np.ndarray:
    """Avoid intermediate reciprocal overflow in extreme complex division."""
    if np.iscomplexobj(values) and (denominator < 1e-150 or denominator > 1e150):
        # np.real/np.imag have ndarray-compatible typing in NumPy's stubs.
        return np.real(values) / denominator + 1j * (np.imag(values) / denominator)
    return values / denominator


def _divide_torch_real(values: torch.Tensor, denominator: float) -> torch.Tensor:
    import torch

    if values.is_complex() and (denominator < 1e-150 or denominator > 1e150):
        return torch.complex(values.real / denominator, values.imag / denominator)
    return values / denominator


def _torch_checked_weights(
    data: torch.Tensor, *, vector: bool
) -> torch.Tensor | np.ndarray:
    """Return vector Born weights or a checked density diagonal."""
    if vector:
        return _torch_vector_weights(data)
    diagonal = data.diagonal()
    import torch

    if data.device.type == "cpu" and data.dtype != torch.bfloat16:
        # No CPU<->GPU copy and no D x D density conversion.
        view = diagonal.detach().resolve_conj().resolve_neg().numpy()
        return _checked_diagonal_numpy(view)
    checked = _checked_diagonal_torch(diagonal)
    return (
        checked.to(torch.float32)
        if checked.dtype in (torch.float16, torch.bfloat16)
        else checked
    )


def _scaled_basis_weights(state: Qobj) -> tuple[np.ndarray, float]:
    """Stable, dimensionless weights for the *exceptional* overflow/underflow path.

    Normally we square amplitudes in float64 directly. If their physical norm
    cannot be represented in float64, all normalized measurement APIs can
    still work by dividing by the maximum amplitude/diagonal BEFORE squaring.
    Returned ``scale`` is in amplitude units (ket/bra) or density units (oper).
    This path transfers torch inputs to CPU only when a normal reduction failed.
    """
    data = state._data
    if isinstance(data, csr_array):
        values = data.data if state.is_vector() else _sparse_basis_weights(state)[1]
    elif isinstance(data, np.ndarray):
        values = (
            data.reshape(-1)
            if state.is_vector()
            else _checked_diagonal_numpy(np.diagonal(data))
        )
    else:
        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError("unsupported Qobj data backend")
        if state.is_vector():
            values = (
                data.detach().cpu().resolve_conj().resolve_neg().numpy().reshape(-1)
            )
        else:
            checked = _torch_checked_weights(data, vector=False)
            values = (
                checked
                if isinstance(checked, np.ndarray)
                else checked.detach().cpu().numpy()
            )

    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("state coefficients or density diagonal must be finite")
    if state.is_vector():
        values = values.astype(
            np.complex128 if np.iscomplexobj(values) else np.float64, copy=False
        )
        # Max of real and imaginary components remains representable even if
        # hypot(real, imag) of an extreme complex128 value would overflow.
        scale = max(
            float(np.max(np.abs(values.real), initial=0.0)),
            float(np.max(np.abs(values.imag), initial=0.0))
            if np.iscomplexobj(values)
            else 0.0,
        )
        if scale <= 0:
            raise ValueError("state must have positive finite norm or trace")
        reduced = _divide_numpy_real(values, scale)
        weights = np.square(np.abs(reduced), dtype=np.float64)
    else:
        weights = np.asarray(values, dtype=np.float64)
        if (weights < 0).any():
            raise ValueError("joint photon-number probabilities must be nonnegative")
        scale = float(np.max(weights, initial=0.0))
        if scale <= 0:
            raise ValueError("state must have positive finite norm or trace")
        weights = weights / scale
    return weights, scale


def _scaled_joint_weights(
    state: Qobj, dims: tuple[int, ...], modes: tuple[int, ...]
) -> tuple[np.ndarray, float]:
    weights, scale = _scaled_basis_weights(state)
    if isinstance(state._data, csr_array):
        data = state._data
        if state.qtype == Qtype.ket:
            indices = _csr_ket_rows(data)
        elif state.qtype == Qtype.bra:
            indices = data.indices
        elif data.nnz >= 4 * data.shape[0]:
            indices = np.arange(data.shape[0], dtype=np.intp)
        else:
            entries = data.tocoo(copy=False)
            indices = entries.row[entries.row == entries.col]
        counts = np.bincount(
            _flat_outcomes(indices, dims, modes),
            weights=weights,
            minlength=prod(dims[mode] for mode in modes),
        )
        return counts.reshape(tuple(dims[mode] for mode in modes)), scale
    return _reduce_numpy(weights, dims, modes), scale


def _raw_joint_weights(
    state: Qobj, dims: tuple[int, ...], modes: tuple[int, ...]
) -> np.ndarray:
    data = state._data
    output_shape = tuple(dims[mode] for mode in modes)
    if isinstance(data, csr_array):
        indices, weights = _sparse_basis_weights(state)
        bins = _flat_outcomes(indices, dims, modes)
        return np.bincount(
            bins,
            weights=np.asarray(weights, dtype=np.float64),
            minlength=prod(output_shape),
        ).reshape(output_shape)
    if isinstance(data, np.ndarray):
        weights = (
            _numpy_vector_weights(data.reshape(-1))
            if state.is_vector()
            else _checked_diagonal_numpy(np.diagonal(data))
        )
        return _reduce_numpy(weights, dims, modes)

    import torch

    if not isinstance(data, torch.Tensor):
        raise TypeError("unsupported Qobj data backend")
    torch_weights = _torch_checked_weights(data, vector=state.is_vector())
    if isinstance(torch_weights, np.ndarray):
        return _reduce_numpy(torch_weights, dims, modes)
    return (
        _reduce_torch(torch_weights, dims, modes)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )


def _checked_total(weights: np.ndarray) -> float:
    if not np.isfinite(weights).all():
        raise ValueError("joint photon-number probabilities must be finite")
    if np.any(weights < 0):
        raise ValueError("joint photon-number probabilities must be nonnegative")
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("state must have positive finite norm or trace")
    return total


def _normalized_joint_weights(
    state: Qobj, dims: tuple[int, ...], modes: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, float]:
    raw = _raw_joint_weights(state, dims, modes)
    scale = 1.0
    try:
        total = _checked_total(raw)
    except ValueError:
        raw, scale = _scaled_joint_weights(state, dims, modes)
        total = _checked_total(raw)
    return raw / total, raw, scale


def joint_photon_number_probabilities(
    state: Qobj, modes: Sequence[int] | None = None
) -> np.ndarray:
    """Exact Born distribution of *two or more* photon-count modes.

    ``modes=None`` measures all modes. Otherwise output axes follow precisely
    the order in ``modes`` (even if unsorted); unselected modes are marginalized.
    Kets, bras and density operators work with NumPy, SciPy CSR and PyTorch.
    Unnormalized states are accepted, invalid diagonals/zero norms rejected.
    Returns a NumPy array. No RNG is used.
    """
    dims, selected = _modes_and_dims(state, modes)
    normalized, _, _ = _normalized_joint_weights(state, dims, selected)
    return normalized


def _selected_indices(
    dims: tuple[int, ...], selected: dict[int, int]
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Flat indices of a branch, O(size of the unmeasured Hilbert space)."""
    remaining = tuple(i for i in range(len(dims)) if i not in selected)
    reduced = np.arange(prod(dims[i] for i in remaining), dtype=np.intp)
    indices = np.zeros(reduced.shape, dtype=np.intp)
    full_strides = _strides(dims)
    remaining_strides = _strides(tuple(dims[i] for i in remaining))
    rest_pos = {mode: i for i, mode in enumerate(remaining)}
    for mode, dimension in enumerate(dims):
        if mode in selected:
            indices += selected[mode] * full_strides[mode]
        else:
            pos = rest_pos[mode]
            indices += ((reduced // remaining_strides[pos]) % dimension) * full_strides[
                mode
            ]
    return indices, remaining


def _remaining_index(
    indices: np.ndarray, dims: tuple[int, ...], remaining: tuple[int, ...]
) -> np.ndarray:
    if not remaining:
        return np.zeros(indices.shape, dtype=np.intp)
    return _flat_outcomes(indices, dims, remaining)


def _reduced_dims(
    state: Qobj, remaining: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left, right = state.dims
    if state.qtype == Qtype.oper:
        subset = tuple(left[i] for i in remaining)
        return subset, subset
    if state.qtype == Qtype.ket:
        return (
            tuple(left[i] for i in remaining),
            tuple(right[i] for i in remaining) if len(right) == len(left) else right,
        )
    return (
        tuple(left[i] for i in remaining) if len(left) == len(right) else left,
        tuple(right[i] for i in remaining),
    )


def _branch_weight(
    state: Qobj, dims: tuple[int, ...], selected: dict[int, int]
) -> tuple[float, float, float]:
    """Compute ONE branch weight and total norm, without joint histogram.

    Avoids an allocation proportional to the product of the measured cutoffs.
    The norm/trace must still examine all present state amplitudes or diagonals.
    """
    data = state._data
    if isinstance(data, csr_array):
        indices, weights = _sparse_basis_weights(state)
        mask = np.ones(indices.size, dtype=bool)
        strides = _strides(dims)
        for mode, number in selected.items():
            mask &= (indices // strides[mode]) % dims[mode] == number
        scale = 1.0
        try:
            total = _checked_total(np.asarray(weights))
        except ValueError:
            weights, scale = _scaled_basis_weights(state)
            total = _checked_total(weights)
        return float(np.sum(weights[mask], dtype=np.float64)), total, scale

    positions, _ = _selected_indices(dims, selected)
    if isinstance(data, np.ndarray):
        weights = (
            _numpy_vector_weights(data.reshape(-1))
            if state.is_vector()
            else _checked_diagonal_numpy(np.diagonal(data))
        )
        scale = 1.0
        try:
            total = _checked_total(np.asarray(weights))
        except ValueError:
            weights, scale = _scaled_basis_weights(state)
            total = _checked_total(weights)
        return float(np.sum(weights[positions], dtype=np.float64)), total, scale

    import torch

    if not isinstance(data, torch.Tensor):
        raise TypeError("unsupported Qobj data backend")
    torch_weights = _torch_checked_weights(data, vector=state.is_vector())
    if isinstance(torch_weights, np.ndarray):
        scale = 1.0
        try:
            total = _checked_total(np.asarray(torch_weights))
        except ValueError:
            torch_weights, scale = _scaled_basis_weights(state)
            total = _checked_total(torch_weights)
        return float(np.sum(torch_weights[positions], dtype=np.float64)), total, scale
    selected_pos = torch.as_tensor(
        positions, device=torch_weights.device, dtype=torch.long
    )
    total = float(
        torch_weights.sum(
            dtype=(torch.float64 if torch_weights.device.type != "mps" else None)
        ).item()
    )
    if (
        not np.isfinite(total)
        or total <= 0
        or not torch.isfinite(torch_weights).all().item()
        or (torch_weights < 0).any().item()
    ):
        scaled, scale = _scaled_basis_weights(state)
        total = _checked_total(scaled)
        return float(np.sum(scaled[positions], dtype=np.float64)), total, scale
    branch = float(
        torch_weights.index_select(0, selected_pos)
        .sum(dtype=(torch.float64 if torch_weights.device.type != "mps" else None))
        .item()
    )
    return branch, total, 1.0


def _normalized_numpy_values(
    values: np.ndarray, norm: float, scale: float
) -> np.ndarray:
    """One branch's normalized data; handle an unrepresentable physical norm."""
    if scale != 1.0:
        values = _divide_numpy_real(
            values.astype(
                np.complex128 if np.iscomplexobj(values) else np.float64, copy=False
            ),
            scale,
        )
    return _divide_numpy_real(_safe_numpy_dividend(values, norm), norm)


def _project_joint(
    state: Qobj,
    dims: tuple[int, ...],
    selected: dict[int, int],
    weight: float,
    discard: bool,
    scale: float = 1.0,
) -> Qobj:
    """Exactly one projector application, with no intermediate Qobj states."""
    data = state._data
    norm = sqrt(weight) if state.is_vector() else weight
    remaining = tuple(i for i in range(len(dims)) if i not in selected)
    output_size = prod(dims[i] for i in remaining)
    output_dims = _reduced_dims(state, remaining) if discard else state.dims

    if isinstance(data, csr_array):
        strides = _strides(dims)
        if state.is_vector():
            indices = _csr_ket_rows(data) if state.qtype == Qtype.ket else data.indices
            mask = np.ones(indices.size, dtype=bool)
            for mode, number in selected.items():
                mask &= (indices // strides[mode]) % dims[mode] == number
            kept = (
                _remaining_index(indices[mask], dims, remaining)
                if discard
                else indices[mask]
            )
            values = data.data[mask]
            if _numpy_normal_fast_path(
                values.dtype, norm, scale
            ) or not _numpy_needs_stable_division(values.dtype, norm, scale):
                values = values / norm
            else:
                values = _normalized_numpy_values(values, norm, scale)
            if state.qtype == Qtype.ket:
                result = csr_array(
                    (values, (kept, np.zeros_like(kept))),
                    shape=((output_size, 1) if discard else data.shape),
                )
            else:
                result = csr_array(
                    (values, (np.zeros_like(kept), kept)),
                    shape=((1, output_size) if discard else data.shape),
                )
        else:
            entries = data.tocoo(copy=False)
            rows, cols = entries.row, entries.col
            mask = np.ones(rows.size, dtype=bool)
            for mode, number in selected.items():
                stride = strides[mode]
                mask &= ((rows // stride) % dims[mode] == number) & (
                    (cols // stride) % dims[mode] == number
                )
            kept_rows, kept_cols = rows[mask], cols[mask]
            if discard:
                kept_rows = _remaining_index(kept_rows, dims, remaining)
                kept_cols = _remaining_index(kept_cols, dims, remaining)
            values = entries.data[mask]
            if _numpy_normal_fast_path(
                values.dtype, norm, scale
            ) or not _numpy_needs_stable_division(values.dtype, norm, scale):
                values = values / norm
            else:
                values = _normalized_numpy_values(values, norm, scale)
            result = csr_array(
                (values, (kept_rows, kept_cols)),
                shape=((output_size, output_size) if discard else data.shape),
            )
        return Qobj(result, dims=output_dims, qtype=state.qtype, tol=state._tol)

    indices, _ = _selected_indices(dims, selected)
    if isinstance(data, np.ndarray):
        # Integer/bool arrays require floating-point normalization.
        dtype = np.dtype(np.float64 if data.dtype.kind in "biu" else data.dtype)
        stable = not _numpy_normal_fast_path(
            dtype, norm, scale
        ) and _numpy_needs_stable_division(dtype, norm, scale)
        if stable:
            dtype = np.dtype(np.complex128 if data.dtype.kind == "c" else np.float64)
        if state.is_vector():
            flat = data.reshape(-1)
            numpy_selected_data = (
                _normalized_numpy_values(flat[indices], norm, scale)
                if stable
                else flat[indices] / norm
            )
            if discard:
                numpy_output = numpy_selected_data.reshape(
                    (1, output_size) if state.qtype == Qtype.bra else (output_size, 1)
                )
            else:
                numpy_output = np.zeros(data.shape, dtype=dtype)
                numpy_output.reshape(-1)[indices] = numpy_selected_data
        else:
            block = np.ix_(indices, indices)
            numpy_selected_data = (
                _normalized_numpy_values(data[block], norm, scale)
                if stable
                else data[block] / norm
            )
            if discard:
                numpy_output = numpy_selected_data
            else:
                numpy_output = np.zeros(data.shape, dtype=dtype)
                numpy_output[block] = numpy_selected_data
        return Qobj(numpy_output, dims=output_dims, qtype=state.qtype, tol=state._tol)

    import torch

    if not isinstance(data, torch.Tensor):
        raise TypeError("unsupported Qobj data backend")
    data = _torch_coefficients(data)
    if scale != 1.0:
        if data.device.type != "mps" and data.dtype in (torch.float32, torch.complex64):
            data = data.to(torch.complex128 if data.is_complex() else torch.float64)
        data = _divide_torch_real(data, scale)
    data = _safe_torch_dividend(data, norm)
    positions = torch.as_tensor(indices, device=data.device, dtype=torch.long)
    if state.is_vector():
        torch_flat = data.reshape(-1)
        torch_selected_data = _divide_torch_real(
            torch_flat.index_select(0, positions), norm
        )
        if discard:
            torch_output = torch_selected_data.reshape(
                (1, output_size) if state.qtype == Qtype.bra else (output_size, 1)
            )
        else:
            torch_output = torch.zeros_like(data)
            torch_output.reshape(-1)[positions] = torch_selected_data
    else:
        torch_block = _divide_torch_real(
            data.index_select(0, positions).index_select(1, positions), norm
        )
        if discard:
            torch_output = torch_block
        else:
            torch_output = torch.zeros_like(data)
            torch_output[positions[:, None], positions[None, :]] = torch_block
    return Qobj(torch_output, dims=output_dims, qtype=state.qtype, tol=state._tol)


def post_select_joint(
    state: Qobj,
    outcomes: Sequence[int],
    modes: Sequence[int] | None = None,
    *,
    discard: bool = False,
) -> tuple[Qobj, float]:
    """Project a user-specified >=2-mode photon-count result, without RNG.

    Return ``(normalized_conditional_state, success_probability)``. The
    measured modes are retained unless ``discard=True``; then at least one
    unmeasured mode must remain. A zero-probability outcome raises ValueError.
    The input state is unchanged and the Qobj backend is preserved.

    Unlike requesting the full joint distribution, this function computes
    only the requested branch weight, without a potentially huge histogram.
    """
    dims, modes = _modes_and_dims(state, modes)
    numbers = _outcomes(outcomes, modes, dims)
    if not isinstance(discard, bool):
        raise TypeError("discard must be a bool")
    if discard and len(modes) == len(dims):
        raise ValueError("cannot discard every mode; pass discard=False")
    selected = dict(zip(modes, numbers))
    weight, total, scale = _branch_weight(state, dims, selected)
    if weight <= 0:
        raise ValueError("post-selected joint outcome has zero probability")
    return _project_joint(state, dims, selected, weight, discard, scale), float(
        weight / total
    )


def measure_joint_photon_number(
    state: Qobj,
    modes: Sequence[int] | None = None,
    *,
    outcomes: Sequence[int] | None = None,
    discard: bool = False,
) -> np.ndarray | tuple[Qobj, float]:
    """Backwards-compatible RNG-free probability/post-selection convenience API.

    Without ``outcomes`` return the joint Born tensor; with ``outcomes`` return
    ``(post_selected_state, probability)``. Never samples an observed result.
    """
    if outcomes is None:
        if discard:
            raise ValueError("discard=True requires explicit outcomes")
        return joint_photon_number_probabilities(state, modes)
    return post_select_joint(state, outcomes, modes, discard=discard)


def _sparse_branches(
    state: Qobj, dims: tuple[int, ...], modes: tuple[int, ...], discard: bool
) -> dict[tuple[int, ...], tuple[float, Qobj]]:
    """Enumerate only occupied sparse outcome bins; scan sparse entries once."""
    data = state._data
    assert isinstance(data, csr_array)
    indices, weights = _sparse_basis_weights(state)
    scale = 1.0
    try:
        total = _checked_total(np.asarray(weights))
    except ValueError:
        weights, scale = _scaled_basis_weights(state)
        total = _checked_total(weights)
    shape = tuple(dims[i] for i in modes)
    bins = _flat_outcomes(indices, dims, modes)
    unique, inverse = np.unique(bins, return_inverse=True)
    raw_by_bin = np.bincount(
        inverse, weights=np.asarray(weights, dtype=np.float64), minlength=len(unique)
    )
    remaining = tuple(i for i in range(len(dims)) if i not in modes)
    output_size = prod(dims[i] for i in remaining)
    output_dims = _reduced_dims(state, remaining) if discard else state.dims

    # All index/data arrays are initialized on both paths, so later access
    # does not depend on Pyright correlating independent state.is_vector calls.
    # The vector path still avoids converting a sparse vector to COO.
    vector = state.is_vector()
    if vector:
        row_indices = _csr_ket_rows(data) if state.qtype == Qtype.ket else data.indices
        col_indices = row_indices
        entry_values = data.data
        source_bins = _flat_outcomes(row_indices, dims, modes)
        ordered = np.argsort(source_bins, kind="stable")
        sorted_bins = source_bins[ordered]
    else:
        entries = data.tocoo(copy=False)
        row_indices, col_indices, entry_values = entries.row, entries.col, entries.data
        row_bins = _flat_outcomes(row_indices, dims, modes)
        col_bins = _flat_outcomes(col_indices, dims, modes)
        candidates = np.flatnonzero(row_bins == col_bins)
        ordered = candidates[np.argsort(row_bins[candidates], kind="stable")]
        sorted_bins = row_bins[ordered]

    output: dict[tuple[int, ...], tuple[float, Qobj]] = {}
    for packed, weight in zip(unique, raw_by_bin):
        if weight <= 0:
            continue
        first = np.searchsorted(sorted_bins, packed, side="left")
        last = np.searchsorted(sorted_bins, packed, side="right")
        chosen = ordered[first:last]
        normalization = sqrt(float(weight)) if vector else float(weight)
        if vector:
            indices_selected = row_indices[chosen]
            if discard:
                indices_selected = _remaining_index(indices_selected, dims, remaining)
            chosen_data = entry_values[chosen]
            amplitudes = (
                _normalized_numpy_values(chosen_data, normalization, scale)
                if not _numpy_normal_fast_path(chosen_data.dtype, normalization, scale)
                and _numpy_needs_stable_division(
                    chosen_data.dtype, normalization, scale
                )
                else chosen_data / normalization
            )
            if state.qtype == Qtype.ket:
                result = csr_array(
                    (amplitudes, (indices_selected, np.zeros_like(indices_selected))),
                    shape=((output_size, 1) if discard else data.shape),
                )
            else:
                result = csr_array(
                    (amplitudes, (np.zeros_like(indices_selected), indices_selected)),
                    shape=((1, output_size) if discard else data.shape),
                )
        else:
            rows, cols = row_indices[chosen], col_indices[chosen]
            if discard:
                rows = _remaining_index(rows, dims, remaining)
                cols = _remaining_index(cols, dims, remaining)
            chosen_data = entry_values[chosen]
            normalized_data = (
                _normalized_numpy_values(chosen_data, normalization, scale)
                if not _numpy_normal_fast_path(chosen_data.dtype, normalization, scale)
                and _numpy_needs_stable_division(
                    chosen_data.dtype, normalization, scale
                )
                else chosen_data / normalization
            )
            result = csr_array(
                (normalized_data, (rows, cols)),
                shape=((output_size, output_size) if discard else data.shape),
            )
        branch = tuple(int(n) for n in np.unravel_index(int(packed), shape))
        output[branch] = (
            float(weight / total),
            Qobj(result, dims=output_dims, qtype=state.qtype, tol=state._tol),
        )
    return output


def joint_photon_number_branches(
    state: Qobj, modes: Sequence[int] | None = None, *, discard: bool = False
) -> dict[tuple[int, ...], tuple[float, Qobj]]:
    """Enumerate all nonzero-probability joint measurement branches (no RNG).

    Maps ``(n_0, ..., n_k)`` in the *specified mode order* to
    ``(probability, normalized_conditional_Qobj)``. Zero-probability outcomes
    are omitted. Sparse states enumerate only occupied outcome bins without
    materializing the full probability tensor; dense states compute it once.
    Materializing many conditional states can still be very expensive.
    """
    dims, selected = _modes_and_dims(state, modes)
    if not isinstance(discard, bool):
        raise TypeError("discard must be a bool")
    if discard and len(selected) == len(dims):
        raise ValueError("cannot discard every mode; pass discard=False")
    if isinstance(state._data, csr_array):
        return _sparse_branches(state, dims, selected, discard)
    probabilities, raw, scale = _normalized_joint_weights(state, dims, selected)
    branches: dict[tuple[int, ...], tuple[float, Qobj]] = {}
    for counts in np.argwhere(raw > 0):
        outcome = tuple(int(n) for n in counts)
        projected = _project_joint(
            state,
            dims,
            dict(zip(selected, outcome)),
            float(raw[outcome]),
            discard,
            scale,
        )
        branches[outcome] = (float(probabilities[outcome]), projected)
    return branches


def _dephase_sparse_vector(
    state: Qobj, dims: tuple[int, ...], modes: tuple[int, ...]
) -> csr_array:
    """Build a sparse post-measurement density directly in CSR format.

    For each occupied basis row, emit only columns in the same joint outcome.
    CSR buffers are filled once, avoiding three large concatenated COO arrays.
    The output necessarily has sum_b (occupied amplitudes in b)^2 entries.
    """
    data = state._data
    assert isinstance(data, csr_array)
    size = data.shape[1] if state.qtype == Qtype.bra else data.shape[0]
    indices = _csr_ket_rows(data) if state.qtype == Qtype.ket else data.indices
    amplitudes = data.data if state.qtype == Qtype.ket else data.data.conj()
    amplitudes = _safe_numpy_amplitudes(amplitudes)
    groups = _flat_outcomes(indices, dims, modes)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1,
            np.array([len(order)], dtype=np.intp),
        )
    )
    row_counts = np.zeros(size, dtype=np.intp)
    for start, stop in itertools.pairwise(boundaries):
        row_counts[indices[order[start:stop]]] = stop - start
    indptr = np.empty(size + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    nnz_out = int(indptr[-1])
    col_indices = np.empty(nnz_out, dtype=np.intp)
    result_data = np.empty(nnz_out, dtype=np.result_type(amplitudes.dtype))
    for start, stop in itertools.pairwise(boundaries):
        positions = order[start:stop]
        branch_rows = indices[positions]
        branch_amps = amplitudes[positions]
        right = branch_amps.conj()
        length = stop - start
        for basis, amp in zip(branch_rows, branch_amps):
            at = indptr[basis]
            col_indices[at : at + length] = branch_rows
            result_data[at : at + length] = amp * right
    return csr_array((result_data, col_indices, indptr), shape=(size, size))


def _index_groups(labels: np.ndarray):
    """Basis indices sharing a joint photon count, each visited exactly once."""
    order = np.argsort(labels, kind="stable")
    ordered_labels = labels[order]
    start = 0
    for end in np.append(
        np.flatnonzero(ordered_labels[1:] != ordered_labels[:-1]) + 1,
        len(order),
    ):
        yield order[start:end]
        start = int(end)


def _dephase_numpy(state: Qobj, labels: np.ndarray) -> np.ndarray:
    """Dense dephasing using only within-outcome blocks (no D x D mask)."""
    data = state._data
    assert isinstance(data, np.ndarray)
    dimension = labels.size
    if state.is_vector():
        amplitudes = _safe_numpy_amplitudes(data.reshape(-1))
        if state.qtype == Qtype.bra:
            amplitudes = amplitudes.conj()
        result = np.zeros((dimension, dimension), dtype=amplitudes.dtype)
        for indices in _index_groups(labels):
            local = amplitudes[indices]
            result[np.ix_(indices, indices)] = local[:, None] * local.conj()[None, :]
    else:
        result = np.zeros_like(data)
        for indices in _index_groups(labels):
            selected = np.ix_(indices, indices)
            result[selected] = data[selected]
    return result


def _dephase_torch(state: Qobj, labels: np.ndarray) -> torch.Tensor:
    import torch

    data = state._data
    assert isinstance(data, torch.Tensor)
    dimension = labels.size
    amplitudes: torch.Tensor | None = None
    if state.is_vector():
        amplitudes = _safe_torch_amplitudes(data).reshape(-1)
        if state.qtype == Qtype.bra:
            amplitudes = amplitudes.conj()
        result = torch.zeros(
            (dimension, dimension), dtype=amplitudes.dtype, device=data.device
        )
    else:
        result = torch.zeros_like(data)

    if data.device.type == "cpu":
        # Few small blocks are much cheaper than a D x D equality mask.
        for indices in _index_groups(labels):
            selected = torch.as_tensor(indices, dtype=torch.long, device=data.device)
            chosen = (selected[:, None], selected[None, :])
            if amplitudes is not None:
                local = amplitudes.index_select(0, selected)
                result[chosen] = local[:, None] * local.conj()[None, :]
            else:
                result[chosen] = data[chosen]
    else:
        # On GPUs a few batched kernels generally beat one launch per branch.
        # Limit the temporary Boolean mask to ~4 MiB per chunk.
        labels_t = torch.as_tensor(labels, dtype=torch.long, device=data.device)
        chunk = max(1, 4_000_000 // max(dimension, 1))
        for start in range(0, dimension, chunk):
            stop = min(start + chunk, dimension)
            keep = labels_t[start:stop, None] == labels_t[None, :]
            if amplitudes is not None:
                block = amplitudes[start:stop, None] * amplitudes.conj()[None, :]
                result[start:stop] = torch.where(keep, block, 0)
            else:
                result[start:stop] = torch.where(keep, data[start:stop], 0)
    return result


def measure_joint_nonselective(state: Qobj, modes: Sequence[int] | None = None) -> Qobj:
    """Apply the non-selective, >=2-mode Fock measurement channel.

    Implements ``rho -> sum_n Pi_n rho Pi_n``. Coherences *between* different
    joint photon counts disappear; coherences *inside* the same outcome survive.
    Pure kets and bras become density operators. An existing density stays a
    density operator. Preserves norm-squared/trace, rather than renormalizing;
    original backend and subsystem dimensions are preserved. No RNG.

    The result is a D x D density matrix (potentially large); a sparse ket
    becomes a sparse density matrix, without densifying the input.
    """
    dims, selected = _modes_and_dims(state, modes)
    data = state._data
    # Early returns preserve backend-specific types without a mutable
    # ndarray | CSR | Tensor variable spanning multiple control-flow branches.
    if isinstance(data, csr_array):
        _, weights = _sparse_basis_weights(state)
        _checked_total(np.asarray(weights))
        if state.is_vector():
            sparse_result = _dephase_sparse_vector(state, dims, selected)
        else:
            entries = data.tocoo(copy=False)
            keep = _flat_outcomes(entries.row, dims, selected) == _flat_outcomes(
                entries.col, dims, selected
            )
            sparse_result = csr_array(
                (entries.data[keep], (entries.row[keep], entries.col[keep])),
                shape=data.shape,
            )
        return Qobj(sparse_result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol)

    if isinstance(data, np.ndarray):
        weights = (
            _numpy_vector_weights(data.reshape(-1))
            if state.is_vector()
            else _checked_diagonal_numpy(np.diagonal(data))
        )
        _checked_total(np.asarray(weights))
        dimension = data.shape[1] if state.qtype == Qtype.bra else data.shape[0]
        labels = _flat_outcomes(np.arange(dimension, dtype=np.intp), dims, selected)
        dense_result = _dephase_numpy(state, labels)
        return Qobj(dense_result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol)

    import torch

    if not isinstance(data, torch.Tensor):
        raise TypeError("unsupported Qobj data backend")
    torch_weights = _torch_checked_weights(data, vector=state.is_vector())
    if isinstance(torch_weights, np.ndarray):
        _checked_total(torch_weights)
    else:
        if (
            not torch.isfinite(torch_weights).all().item()
            or (torch_weights < 0).any().item()
        ):
            raise ValueError(
                "joint photon-number probabilities must be finite and nonnegative"
            )
        total = float(
            torch_weights.sum(
                dtype=(torch.float64 if data.device.type != "mps" else None)
            ).item()
        )
        if not np.isfinite(total) or total <= 0:
            raise ValueError("state must have positive finite norm or trace")
    dimension = data.shape[1] if state.qtype == Qtype.bra else data.shape[0]
    labels = _flat_outcomes(np.arange(dimension, dtype=np.intp), dims, selected)
    tensor_result = _dephase_torch(state, labels)
    return Qobj(tensor_result, dims=(dims, dims), qtype=Qtype.oper, tol=state._tol)
