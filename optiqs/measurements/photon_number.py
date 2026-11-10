"""Efficient photon-number-resolving (Fock-basis) projective measurements.

No number projectors, partial traces or outer products are constructed.
Dense states are processed on their existing NumPy/PyTorch backend; SciPy
CSR states are never converted to a dense state matrix.
"""

from __future__ import annotations

from math import isfinite, prod, sqrt
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch


# A few ulps are allowed when validating a numerically computed density
# operator.  We validate its *individual* diagonal entries before grouping:
# summing first could hide a negative or imaginary diagonal contribution.
_DIAGONAL_ULPS = 64


def _geometry(state: Qobj, mode: int) -> tuple[int, int]:
    """Return (mode cutoff, flat-index stride), checking the Qobj geometry."""
    if not isinstance(state, Qobj):
        raise TypeError("state must be an optiqs.Qobj")
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
        raise TypeError("mode must be an integer subsystem index")
    if state.qtype == Qtype.oper and state.dims[0] != state.dims[1]:
        raise ValueError("density operator must have matching subsystem dimensions")

    dims = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    if not 0 <= mode < len(dims):
        raise IndexError("mode is outside the available subsystems")
    next_mode = mode + 1
    return dims[mode], prod(dims[next_mode:])


def _squared_magnitudes_numpy(values: np.ndarray) -> np.ndarray:
    """Squared amplitudes in float64, avoiding integer/f32 overflow in squaring."""
    if values.dtype.kind == "c":
        # |z| uses a stable hypot implementation.  Convert *before* squaring
        # so a complex64 amplitude around 1e20 does not overflow float32.
        magnitude = np.abs(values).astype(np.float64, copy=False)
        return np.square(magnitude, dtype=np.float64)
    # Convert signed integers before abs/squaring (including int64 minimum).
    return np.square(values, dtype=np.float64)


def _checked_diagonal_numpy(diagonal: npt.NDArray[np.generic]) -> np.ndarray:
    """Validate density diagonals with minimal temporary arrays/reductions.

    Finite min/max are sufficient for real data.  For complex data, the
    maximum magnitude also checks finiteness while setting the same relative
    tolerance used for the imaginary and negative-real checks.  Invalid
    individual entries must not be hidden by summing modes first.
    """
    real = diagonal.real
    if diagonal.dtype.kind == "c":
        scale = float(np.max(np.abs(diagonal), initial=0.0))
        if not np.isfinite(scale):
            raise ValueError("density-operator diagonal must be finite")
        tolerance = _DIAGONAL_ULPS * np.finfo(real.dtype).eps * scale
        if float(np.max(np.abs(diagonal.imag), initial=0.0)) > tolerance:
            raise ValueError("density-operator diagonal must be real")
        minimum = real.min(initial=0)
    else:
        minimum = real.min(initial=0)
        maximum = real.max(initial=0)
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("density-operator diagonal must be finite")
        eps = np.finfo(real.dtype).eps if diagonal.dtype.kind == "f" else 0.0
        tolerance = _DIAGONAL_ULPS * eps * max(abs(float(minimum)), abs(float(maximum)))
    if minimum < -tolerance:
        raise ValueError("density-operator diagonal must be nonnegative")
    # Clamp tolerated tiny negative values without modifying the source.
    return np.maximum(real, 0) if minimum < 0 else real


def _checked_diagonal_torch(diagonal: torch.Tensor) -> torch.Tensor:
    import torch

    if diagonal.dtype == torch.bool:
        # Torch does not implement abs/clamp_min for Boolean tensors.
        # Boolean diagonal entries are already real and nonnegative.
        return diagonal.to(torch.float32)
    if not torch.isfinite(diagonal).all().item():
        raise ValueError("density-operator diagonal must be finite")
    real = diagonal.real
    eps = (
        torch.finfo(real.dtype).eps
        if diagonal.is_floating_point() or diagonal.is_complex()
        else 0.0
    )
    # Check on-device; transfer only tiny Boolean scalars back to the CPU.
    tolerance = _DIAGONAL_ULPS * eps * torch.abs(diagonal).amax()
    if diagonal.is_complex() and (torch.abs(diagonal.imag) > tolerance).any().item():
        raise ValueError("density-operator diagonal must be real")
    if (real < -tolerance).any().item():
        raise ValueError("density-operator diagonal must be nonnegative")
    return real.clamp_min(0)


def _torch_vector_weights(data: torch.Tensor, dim: int, stride: int) -> np.ndarray:
    import torch

    # fp16/bfloat16 can overflow during squaring or summation; promote first.
    # Integer coefficients also require conversion before squaring.
    if data.dtype in (torch.float16, torch.bfloat16):
        data = data.to(torch.float32)
    elif not (data.is_floating_point() or data.is_complex()):
        data = data.to(torch.float32 if data.device.type == "mps" else torch.float64)
    grouped = data.reshape(-1, dim, stride)
    weights = grouped.abs().square().sum(dim=(0, 2))
    # Only dim probabilities, not the quantum state, are brought to the CPU.
    result = weights.detach().to(device="cpu", dtype=torch.float64).numpy()
    if (
        not np.isfinite(result).all()
        and data.device.type != "mps"
        and data.dtype in (torch.float32, torch.complex64)
    ):
        # Rare extreme unnormalized states may overflow float32 *before* the
        # reduction.  Retry using float64 without slowing the common path.
        # MPS does not support float64/complex128, so it keeps native dtype.
        promoted = data.to(torch.complex128 if data.is_complex() else torch.float64)
        result = (
            promoted.reshape(-1, dim, stride)
            .abs()
            .square()
            .sum(dim=(0, 2))
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .numpy()
        )
    return result


def _csr_ket_rows(data: csr_array) -> np.ndarray:
    """Recover stored ket row indices, choosing the best CSR algorithm.

    CSR stores a row pointer for *every* row, even if the ket has only a
    handful of amplitudes.  Searching that pointer array at each occupied
    position avoids an O(D)-sized scan/temporary when nnz << D.
    The usual compiled COO conversion (or flatnonzero for dense occupancy)
    is faster once nnz is larger.  Qobj canonicalizes duplicate entries.
    """
    if data.nnz <= max(1, data.shape[0] // 100):
        # CSR indptr is sorted.  side='right' returns the row containing the
        # k-th stored entry even when several successive rows are empty.
        return np.searchsorted(
            data.indptr[1:], np.arange(data.nnz, dtype=data.indptr.dtype), side="right"
        )
    if data.nnz >= data.shape[0] // 8:
        return np.flatnonzero(data.indptr[1:] != data.indptr[:-1])
    return data.tocoo(copy=False).row


def _raw_weights(
    state: Qobj, dim: int, stride: int, sparse_ket_rows: list[np.ndarray] | None = None
) -> np.ndarray:
    """Unnormalized Born weights, backend-specialized and projector-free."""
    data = state._data
    if isinstance(data, csr_array):
        if state.qtype == Qtype.oper and data.nnz >= 4 * data.shape[0]:
            # When there are many stored offdiagonal entries per row,
            # SciPy's compiled diagonal extraction is cheaper than
            # converting every entry to COO and filtering them in Python.
            # This allocates O(D) diagonal data, never a dense D x D state.
            diagonal = _checked_diagonal_numpy(np.asarray(data.diagonal()))
            return diagonal.reshape(-1, dim, stride).sum(axis=(0, 2), dtype=np.float64)
        if state.qtype == Qtype.ket:
            indices, values = _csr_ket_rows(data), data.data
            if sparse_ket_rows is not None:
                sparse_ket_rows.append(indices)
        elif state.qtype == Qtype.bra:
            # A CSR bra has exactly one row: its column indices are available
            # directly (no COO conversion or row-index reconstruction).
            indices, values = data.indices, data.data
        else:
            entries = data.tocoo(copy=False)
            mask = entries.row == entries.col
            indices = entries.row[mask]
            values = _checked_diagonal_numpy(entries.data[mask])
            return np.bincount(
                (indices // stride) % dim,
                weights=np.asarray(values, dtype=np.float64),
                minlength=dim,
            )
        return np.bincount(
            (indices // stride) % dim,
            weights=_squared_magnitudes_numpy(values),
            minlength=dim,
        )

    if isinstance(data, np.ndarray):
        if state.qtype == Qtype.oper:
            diagonal = _checked_diagonal_numpy(np.diagonal(data))
            return diagonal.reshape(-1, dim, stride).sum(axis=(0, 2), dtype=np.float64)
        weights = _squared_magnitudes_numpy(data.reshape(-1, dim, stride))
        return weights.sum(axis=(0, 2), dtype=np.float64)

    import torch

    if not isinstance(data, torch.Tensor):
        raise TypeError("unsupported Qobj data backend")
    if state.is_vector():
        return _torch_vector_weights(data, dim, stride)
    diagonal_t = torch.diagonal(data)
    if data.device.type == "cpu" and data.dtype != torch.bfloat16:
        # A CPU tensor's NumPy view is zero-copy, including non-contiguous
        # diagonals.  NumPy validates all diagonal elements and reduces their
        # probabilities with far fewer interpreter/dispatch calls than the
        # equivalent series of small PyTorch operations.  No dense matrix is
        # copied or transferred, and invalid diagonals are still rejected.
        diagonal_np = diagonal_t.detach().resolve_conj().resolve_neg().numpy()
        diagonal = _checked_diagonal_numpy(diagonal_np)
        return diagonal.reshape(-1, dim, stride).sum(axis=(0, 2), dtype=np.float64)
    # CUDA/MPS (and bfloat16 CPU) stay on the tensor backend.  GPU
    # validation is performed on device; only the final probabilities cross
    # the CPU/device boundary for NumPy RNG sampling.
    checked_diagonal_t = _checked_diagonal_torch(diagonal_t)
    torch_weights = checked_diagonal_t.reshape(-1, dim, stride).sum(dim=(0, 2))
    return torch_weights.detach().to(device="cpu", dtype=torch.float64).numpy()


def _normalized_weights(
    state: Qobj,
    dim: int,
    stride: int,
    sparse_ket_rows: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = _raw_weights(state, dim, stride, sparse_ket_rows)
    if not np.isfinite(raw).all():
        raise ValueError("photon-number probabilities must be finite")
    if np.any(raw < 0):
        raise ValueError("photon-number probabilities must be nonnegative")
    total = float(np.sum(raw, dtype=np.float64))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("state must have positive finite norm or trace")
    return raw / total, raw


def photon_number_probabilities(state: Qobj, mode: int = 0) -> np.ndarray:
    """Return normalized probabilities of measuring 0, 1, ..., cutoff-1 photons.

    ``state`` may be a ket, bra, or density operator with NumPy, SciPy CSR, or
    PyTorch storage.  For a tensor product, the leftmost mode has index 0.
    Unnormalized inputs are accepted; zero/invalid weights are rejected.

    For a ket the weights are sums of |amplitude|^2; for density operators they
    are sums of diagonal entries.  Only the density diagonal is validated,
    not the full positive semidefiniteness/Hermiticity of the operator.
    The input is not modified.  No large measurement projectors are built.

    Complexity: O(D) for dense ket or density diagonal (D = total Hilbert
    dimension); O(nnz log D + cutoff) for extremely sparse CSR ket, otherwise
    O(D + nnz + cutoff); O(D + nnz + cutoff) for CSR density operators;
    O(nnz + cutoff) for CSR bra.  No sparse *state matrix* is densified.
    CUDA/MPS torch probabilities are reduced on device.
    """
    dim, stride = _geometry(state, mode)
    return _normalized_weights(state, dim, stride)[0]


pnrd = photon_number_probabilities


def _project(
    state: Qobj,
    dim: int,
    stride: int,
    outcome: int,
    weight: float,
    sparse_ket_rows: np.ndarray | None = None,
) -> Qobj:
    """Apply the selected number projector; preserve backend, dims and type."""
    data = state._data
    normalization = sqrt(weight) if state.is_vector() else weight

    if dim == 1:
        # The only projector is identity; no indexing/scattering is needed.
        result = data / normalization
    elif isinstance(data, csr_array):
        if state.qtype == Qtype.ket:
            rows = (
                sparse_ket_rows if sparse_ket_rows is not None else _csr_ket_rows(data)
            )
            keep = (rows // stride) % dim == outcome
            kept_rows = rows[keep]
            result = csr_array(
                (
                    data.data[keep] / normalization,
                    (kept_rows, np.zeros_like(kept_rows)),
                ),
                shape=data.shape,
            )
        elif state.qtype == Qtype.bra:
            keep = (data.indices // stride) % dim == outcome
            kept_cols = data.indices[keep]
            result = csr_array(
                (
                    data.data[keep] / normalization,
                    (np.zeros_like(kept_cols), kept_cols),
                ),
                shape=data.shape,
            )
        else:
            entries = data.tocoo(copy=False)
            keep = ((entries.row // stride) % dim == outcome) & (
                (entries.col // stride) % dim == outcome
            )
            result = csr_array(
                (
                    entries.data[keep] / normalization,
                    (entries.row[keep], entries.col[keep]),
                ),
                shape=data.shape,
            )
    elif state.is_vector():
        if isinstance(data, np.ndarray):
            # Convert only surviving coefficients when the input is integer:
            # no full-sized dtype-promotion copy is needed.
            output_dtype = np.float64 if data.dtype.kind in "biu" else data.dtype
            output = np.zeros(data.shape, dtype=output_dtype)
            output.reshape(-1, dim, stride)[:, outcome, :] = (
                data.reshape(-1, dim, stride)[:, outcome, :] / normalization
            )
            result = output
        else:
            import torch

            if not isinstance(data, torch.Tensor):
                raise TypeError("unsupported Qobj data backend")
            torch_output_dtype = data.dtype
            if not (data.is_floating_point() or data.is_complex()):
                torch_output_dtype = (
                    torch.float32 if data.device.type == "mps" else torch.float64
                )
            output_t = torch.zeros(
                data.shape, dtype=torch_output_dtype, device=data.device
            )
            output_t.reshape(-1, dim, stride)[:, outcome, :] = (
                data.reshape(-1, dim, stride)[:, outcome, :] / normalization
            )
            result = output_t
    else:
        # A post-measurement density operator has the same D x D shape,
        # but only the (D/cutoff) x (D/cutoff) selected block needs copying.
        size = data.shape[0]
        lead = size // (dim * stride)
        if isinstance(data, np.ndarray):
            output_dtype = np.float64 if data.dtype.kind in "biu" else data.dtype
            output = np.zeros(data.shape, dtype=output_dtype)
            if lead == 1:
                lo, hi = outcome * stride, (outcome + 1) * stride
                output[lo:hi, lo:hi] = data[lo:hi, lo:hi] / normalization
            elif data.flags.c_contiguous:
                # Basic slicing on six-dimensional views avoids np.ix_
                # arrays and the extra gather/scatter of advanced indexing.
                source6 = data.reshape(lead, dim, stride, lead, dim, stride)
                output6 = output.reshape(lead, dim, stride, lead, dim, stride)
                output6[:, outcome, :, :, outcome, :] = (
                    source6[:, outcome, :, :, outcome, :] / normalization
                )
            else:
                # An arbitrary strided view might be copied by reshape.
                # Index only the needed submatrix rather than copying D^2.
                idx = (
                    np.arange(lead, dtype=np.intp)[:, None] * (dim * stride)
                    + outcome * stride
                    + np.arange(stride, dtype=np.intp)[None, :]
                ).reshape(-1)
                chosen = np.ix_(idx, idx)
                output[chosen] = data[chosen] / normalization
            result = output
        else:
            import torch

            if not isinstance(data, torch.Tensor):
                raise TypeError("unsupported Qobj data backend")
            torch_output_dtype = data.dtype
            if not (data.is_floating_point() or data.is_complex()):
                torch_output_dtype = (
                    torch.float32 if data.device.type == "mps" else torch.float64
                )
            output_t = torch.zeros(
                data.shape, dtype=torch_output_dtype, device=data.device
            )
            if lead == 1:
                lo, hi = outcome * stride, (outcome + 1) * stride
                output_t[lo:hi, lo:hi] = data[lo:hi, lo:hi] / normalization
            elif data.is_contiguous():
                source6_t = data.reshape(lead, dim, stride, lead, dim, stride)
                output6_t = output_t.reshape(lead, dim, stride, lead, dim, stride)
                output6_t[:, outcome, :, :, outcome, :] = (
                    source6_t[:, outcome, :, :, outcome, :] / normalization
                )
            else:
                idx_t = (
                    torch.arange(lead, device=data.device)[:, None] * (dim * stride)
                    + outcome * stride
                    + torch.arange(stride, device=data.device)[None, :]
                ).reshape(-1)
                output_t[idx_t[:, None], idx_t[None, :]] = (
                    data[idx_t[:, None], idx_t[None, :]] / normalization
                )
            result = output_t
    return Qobj(result, dims=state.dims, qtype=state.qtype, tol=state._tol)


def measure_photon_number(
    state: Qobj,
    mode: int = 0,
    shots: int = 1,
    *,
    eta: float = 0.0,
    collapse: bool = False,
    rng: np.random.Generator | int | None = None,
) -> int | np.ndarray | tuple[int, Qobj]:
    """Sample a photon-number-resolving measurement with optional photon loss.

    By default return one integer outcome without forming a post-measurement
    state.  ``shots>1`` samples *independent identical preparations*, returns
    an int64 NumPy array, and calculates the probability distribution once.

    ``eta`` is the photon loss probability in [0, 1]: 0 means no photon
    loss (ideal detection), 1 means all photons are lost. For
    ``collapse=False`` and ``eta>0``, sample the *input* photon number and
    thin it with a binomial draw with survival probability ``1 - eta``,
    without constructing the lossy density matrix or even the full lossy
    photon-number distribution. This reproduces the photon-number statistics
    from ``photon_loss(state, eta=1 - eta, mode=mode)``, whose ``eta`` is
    the *transmissivity* rather than the loss probability.

    With ``collapse=True`` and ``shots=1``, return ``(outcome, collapsed_Qobj)``
    where the output retains the measured mode and is normalized using the
    unnormalized branch weight.  Coherences within the selected Fock subspace
    are preserved.  If ``eta>0``, the exact photon-loss channel is applied
    before projection, so the returned state is a density operator and the
    fast sampling-only path is not used. At ``eta=0``, the original state
    type and behavior are preserved. The input Qobj is never modified.

    ``rng`` may be a NumPy Generator, integer seed, or None.  For the behavior
    of the alternative one-shot API, use ``collapse=True`` explicitly.
    """
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, Integral):
        raise TypeError("shots must be a positive integer")
    if shots < 1:
        raise ValueError("shots must be positive")
    if not isinstance(collapse, bool):
        raise TypeError("collapse must be a bool")
    if collapse and shots != 1:
        raise ValueError("collapse=True requires shots=1")
    if isinstance(eta, (bool, np.bool_)) or not isinstance(eta, Real):
        raise TypeError("eta must be a real number in [0, 1]")
    if not isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("eta must be finite and in [0, 1]")
    if isinstance(rng, (bool, np.bool_)) or not (
        rng is None or isinstance(rng, (Integral, np.random.Generator))
    ):
        raise TypeError("rng must be a NumPy Generator, integer seed, or None")

    dim, stride = _geometry(state, mode)
    transmissivity = 1.0 - float(eta)
    if collapse and eta > 0:
        # A conditional post-measurement state contains coherences/correlations
        # that cannot be obtained from the marginal photon-number distribution.
        # Use the full channel only when the caller explicitly requests it.
        from optiqs.channels.photon_loss import photon_loss

        return measure_photon_number(
            photon_loss(state, eta=transmissivity, mode=mode),
            mode=mode,
            collapse=True,
            rng=rng,
        )

    # The sampled branch needs the same ket row indices as probability
    # grouping.  Reuse them when projecting rather than scanning/searching
    # the CSR structure twice.  No cache is held between API calls.
    cached_rows = (
        []
        if collapse and isinstance(state._data, csr_array) and state.qtype == Qtype.ket
        else None
    )
    probabilities, raw = _normalized_weights(
        state, dim, stride, sparse_ket_rows=cached_rows
    )
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    if eta > 0:
        if eta == 1:
            return 0 if shots == 1 else np.zeros(int(shots), dtype=np.int64)
        # For each independently prepared state, N follows the input photon
        # number distribution and the transmitted count is Binomial(N, 1 - eta).
        # Works for kets, bras and density operators on every Qobj backend.
        if shots == 1:
            initial_count = int(generator.choice(dim, p=probabilities))
            return int(generator.binomial(initial_count, transmissivity))
        initial_counts = generator.choice(dim, size=int(shots), p=probabilities)
        return generator.binomial(initial_counts, transmissivity)

    if shots == 1:
        outcome = int(generator.choice(dim, p=probabilities))
        return (
            (
                outcome,
                _project(
                    state,
                    dim,
                    stride,
                    outcome,
                    float(raw[outcome]),
                    sparse_ket_rows=cached_rows[0] if cached_rows else None,
                ),
            )
            if collapse
            else outcome
        )
    return generator.choice(dim, size=int(shots), p=probabilities)
