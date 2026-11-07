"""Continuous-variable, single-mode phase-space distributions."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.special import eval_genlaguerre, eval_laguerre, gammaln

from optiqs.core.q_obj import Qobj, Qtype

WignerMethod = Literal["auto", "clenshaw", "laguerre"]


def _axis(values: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real numbers")
    try:
        axis = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain only finite values")
    return axis


def _input_data(state: Qobj) -> tuple[NDArray[np.complex128] | sparse.csr_array, bool]:
    """Return a CPU representation without building a ket's density matrix."""
    data = state._data
    if isinstance(data, sparse.csr_array):
        if not np.all(np.isfinite(data.data)):
            raise ValueError("state contains non-finite entries")
        result: NDArray[np.complex128] | sparse.csr_array = data
    elif isinstance(data, np.ndarray):
        if not np.all(np.isfinite(data)):
            raise ValueError("state contains non-finite entries")
        result = np.asarray(data, dtype=np.complex128)
    else:
        # Only the result is on CPU; the original Torch Qobj is untouched.
        result = state.to_numpy().astype(np.complex128, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError("state contains non-finite entries")
    return result, state.qtype != Qtype.oper


def _hermitian(data: NDArray[np.complex128] | sparse.csr_array) -> bool:
    if isinstance(data, sparse.csr_array):
        skew = data - data.conj().T
        if skew.nnz == 0:
            return True
        scale = max(1.0, float(np.max(np.abs(data.data), initial=0.0)))
        return bool(np.max(np.abs(skew.data)) <= 1e-10 * scale)
    return bool(np.allclose(data, data.conj().T, rtol=1e-10, atol=1e-10))


def _coefficients(
    state: Qobj, data: NDArray[np.complex128] | sparse.csr_array
) -> NDArray[np.complex128]:
    """Turn a ket or bra into its ket coefficients (only O(dim) memory)."""
    if isinstance(data, sparse.csr_array):
        coo = data.tocoo()
        ket = np.zeros(max(data.shape), dtype=np.complex128)
        if state.qtype == Qtype.ket:
            ket[coo.row] = coo.data
        else:
            ket[coo.col] = np.conj(coo.data)
    elif state.qtype == Qtype.ket:
        ket = data[:, 0]
    else:
        ket = np.conj(data[0, :])
    return ket


def _upper_diagonals(
    coeff: NDArray[np.complex128] | None,
    rho: NDArray[np.complex128] | sparse.csr_array | None,
    dimension: int,
) -> dict[int, NDArray[np.complex128]]:
    """Prepare nonzero density-matrix superdiagonals without dense outer products."""
    diagonals: dict[int, NDArray[np.complex128]] = {}
    if isinstance(rho, sparse.csr_array) or (
        coeff is not None and np.count_nonzero(coeff) ** 2 * 8 < dimension
    ):
        # A few distant Fock levels must not trigger O(dim**2) scans.
        rows, cols, values = _upper_pairs(coeff, rho)
        offsets = cols - rows
        for offset in np.unique(offsets):
            mask = offsets == offset
            active_rows = rows[mask]
            diag = np.zeros(int(np.max(active_rows)) + 1, dtype=np.complex128)
            diag[active_rows] = values[mask] * (2 if offset else 1)
            diagonals[int(offset)] = diag
        return diagonals

    for offset in range(dimension):
        if coeff is not None:
            diag = coeff[: dimension - offset] * coeff[offset:dimension].conj()
        elif isinstance(rho, np.ndarray):
            diag = np.diagonal(rho, offset=offset)
        else:
            raise RuntimeError("missing state representation")
        populated = np.flatnonzero(diag)
        if populated.size:
            diag = np.asarray(diag[: populated[-1] + 1], dtype=np.complex128)
            if offset:
                diag = 2.0 * diag  # lower triangle is its complex conjugate
            diagonals[offset] = diag
    return diagonals


def _clenshaw_laguerre(
    coefficients: NDArray[np.complex128], offset: int, argument: NDArray[np.float64]
) -> NDArray[np.complex128] | complex:
    """Evaluate sum c_n (-1)^n sqrt(d!n!/(n+d)!) L_n^d(z).

    Backward Clenshaw recurrence avoids computing factorials or high powers.
    """
    if len(coefficients) == 1:
        return complex(coefficients[0])
    ahead: NDArray[np.complex128] | complex = 0j
    two_ahead: NDArray[np.complex128] | complex = 0j
    for n in range(len(coefficients) - 1, -1, -1):
        denom = math.sqrt((n + 1) * (n + offset + 1))
        recurrence = (argument - (2 * n + offset + 1)) / denom
        back_factor = denom / math.sqrt((n + 2) * (n + offset + 2))
        current = coefficients[n] + recurrence * ahead - back_factor * two_ahead
        ahead, two_ahead = current, ahead
    return ahead


def _wigner_clenshaw(
    diagonals: dict[int, NDArray[np.complex128]],
    a: NDArray[np.complex128],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not diagonals:
        return np.zeros(a.size, dtype=np.float64)
    accumulator: NDArray[np.complex128] | complex = 0j
    previous: int | None = None
    for offset in sorted(diagonals, reverse=True):
        if previous is not None:
            # F_{d-1} = R_{d-1} + a/sqrt(d) * F_d.
            # Process gaps without evaluating zero-coefficient Laguerre sums.
            for power in range(previous, offset, -1):
                accumulator = accumulator * (a / math.sqrt(power))
        accumulator = accumulator + _clenshaw_laguerre(diagonals[offset], offset, z)
        previous = offset
    if previous is not None:
        for power in range(previous, 0, -1):
            accumulator = accumulator * (a / math.sqrt(power))
    return np.asarray(np.real(accumulator), dtype=np.float64)


def _upper_pairs(
    coefficients: NDArray[np.complex128] | None,
    rho: NDArray[np.complex128] | sparse.csr_array | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]]:
    """Only nonzero upper-triangular density matrix elements, no ket outer product."""
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    values: NDArray[np.complex128]
    if coefficients is not None:
        support = np.flatnonzero(coefficients)
        i, j = np.triu_indices(support.size)
        rows = np.asarray(support[i], dtype=np.int64)
        cols = np.asarray(support[j], dtype=np.int64)
        values = coefficients[rows] * coefficients[cols].conj()
    elif isinstance(rho, sparse.csr_array):
        coo = sparse.triu(rho, format="coo")
        # SciPy may use 32-bit sparse indices; normalize for NumPy typing.
        rows = np.asarray(coo.row, dtype=np.int64)
        cols = np.asarray(coo.col, dtype=np.int64)
        values = np.asarray(coo.data, dtype=np.complex128)
    elif rho is not None:
        dense_rows, dense_cols = np.triu_indices(rho.shape[0])
        rows = np.asarray(dense_rows, dtype=np.int64)
        cols = np.asarray(dense_cols, dtype=np.int64)
        values = rho[rows, cols]
    else:
        raise RuntimeError("missing state representation")
    mask = values != 0
    return (
        np.asarray(rows[mask], dtype=np.int64),
        np.asarray(cols[mask], dtype=np.int64),
        np.asarray(values[mask], dtype=np.complex128),
    )


def _wigner_laguerre(
    pairs: tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]],
    a: NDArray[np.complex128],
    z: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Direct Fock kernels, with a log-scaled high-order off-diagonal factor."""
    rows, cols, vals = pairs
    total = np.zeros(z.shape, dtype=np.float64)
    gaussian = np.exp(-0.5 * z)
    log_radius = np.full(z.shape, -np.inf, dtype=np.float64)
    np.log(np.abs(a), out=log_radius, where=(z > 0))
    angle = np.angle(a)

    for m, n, amplitude in zip(rows, cols, vals):
        m, n = int(m), int(n)
        sign = 1 if m % 2 == 0 else -1
        if m == n:
            total += sign * amplitude.real * gaussian * eval_laguerre(m, z)
        else:
            power = n - m
            log_factor = 0.5 * float(gammaln(m + 1) - gammaln(n + 1))
            # Including the Gaussian here prevents overflow in a**power
            # when two populated Fock levels are far apart.
            factor = np.exp(log_factor + power * log_radius - 0.5 * z)
            angular = np.exp(1j * power * angle)
            total += 2 * np.real(
                amplitude * sign * eval_genlaguerre(m, power, z) * factor * angular
            )
    return total


def wigner(
    state: Qobj,
    xvec: ArrayLike,
    pvec: ArrayLike | None = None,
    *,
    hbar: float = 1.0,
    method: WignerMethod = "auto",
    chunk_size: int = 32768,
) -> NDArray[np.float64]:
    """Compute a one-mode state's Wigner function W(x, p).

    The result has shape ``(len(pvec), len(xvec))`` and uses the convention
    ``[x,p] = 1j*hbar``, ``alpha = (x+1j*p)/sqrt(2*hbar)``.  The vacuum is
    ``W(x,p) = exp(-(x*x+p*p)/hbar) / (pi*hbar)``.

    Accepts ket, bra, or Hermitian density-matrix Qobj, dense NumPy, SciPy
    CSR, or Torch backed.  Torch data is evaluated on CPU.  The input is
    neither changed nor normalized.  For multimode states first use
    ``state.ptrace(mode)`` to obtain one mode's reduced state.

    ``method='auto'`` uses direct associated Laguerre polynomials for very
    sparse states and normalized Clenshaw recurrences otherwise.  Both
    methods avoid displacement-operator exponentials and, for a pure state,
    avoid materializing its O(dim**2) density matrix.  Processing points
    in chunks limits working memory to O(chunk_size) plus the output.
    """
    if not isinstance(state, Qobj):
        raise TypeError("state must be an optiqs.Qobj")
    if method not in ("auto", "clenshaw", "laguerre"):
        raise ValueError("method must be 'auto', 'clenshaw', or 'laguerre'")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer")
    try:
        if np.iscomplexobj(hbar):
            raise TypeError
        hbar = float(hbar)
    except (TypeError, ValueError) as exc:
        raise ValueError("hbar must be a positive finite real number") from exc
    if not math.isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be a positive finite real number")

    x = _axis(xvec, "xvec")
    p = x if pvec is None else _axis(pvec, "pvec")
    if state.qtype == Qtype.ket:
        if len(state.dims[0]) != 1 or state.dims[1] != (1,):
            raise ValueError("wigner supports a single mode; use state.ptrace(mode)")
    elif state.qtype == Qtype.bra:
        if len(state.dims[1]) != 1 or state.dims[0] != (1,):
            raise ValueError("wigner supports a single mode; use state.ptrace(mode)")
    else:
        if len(state.dims[0]) != 1 or state.dims[0] != state.dims[1]:
            raise ValueError("wigner supports a single mode; use state.ptrace(mode)")

    data, pure = _input_data(state)
    if not pure and not _hermitian(data):
        raise ValueError("density operator must be Hermitian")

    coeff = _coefficients(state, data) if pure else None
    rho = None if pure else data
    if coeff is not None:
        # Narrow Optional[NDArray] before calling NumPy (for Pyright/mypy).
        support = np.flatnonzero(coeff)
        effective_dim = int(support[-1]) + 1 if support.size else 0
        pair_count = len(support) * (len(support) + 1) // 2
        max_gap = int(support[-1] - support[0]) if support.size else 0
    elif isinstance(data, sparse.csr_array):
        sparse_rows, sparse_cols = data.nonzero()
        effective_dim = (
            int(max(sparse_rows.max(), sparse_cols.max())) + 1
            if sparse_rows.size
            else 0
        )
        upper = sparse_rows <= sparse_cols
        pair_count = int(np.count_nonzero(upper))
        max_gap = (
            int(np.max(sparse_cols[upper] - sparse_rows[upper])) if np.any(upper) else 0
        )
    else:
        dense_rows, dense_cols = np.nonzero(data)
        effective_dim = (
            int(max(dense_rows.max(), dense_cols.max())) + 1 if dense_rows.size else 0
        )
        upper = dense_rows <= dense_cols
        pair_count = int(np.count_nonzero(upper))
        max_gap = (
            int(np.max(dense_cols[upper] - dense_rows[upper])) if np.any(upper) else 0
        )

    if method == "auto":
        method = "laguerre" if pair_count <= 16 and max_gap <= 100 else "clenshaw"

    diagonals: dict[int, NDArray[np.complex128]] | None = None
    pairs: (
        tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.complex128]] | None
    ) = None
    if method == "clenshaw":
        diagonals = _upper_diagonals(coeff, rho, effective_dim)
    else:
        pairs = _upper_pairs(coeff, rho)

    output = np.zeros((p.size, x.size), dtype=np.float64)
    flat = output.reshape(-1)
    scale = math.sqrt(2.0 / hbar)
    normalization = 1.0 / (math.pi * hbar)
    nx = x.size
    for start in range(0, flat.size, chunk_size):
        end = min(flat.size, start + chunk_size)
        indices = np.arange(start, end)
        a: NDArray[np.complex128] = np.asarray(
            scale * (x[indices % nx] + 1j * p[indices // nx]),
            dtype=np.complex128,
        )
        z = a.real**2 + a.imag**2
        if diagonals is not None:
            values = np.exp(-0.5 * z) * _wigner_clenshaw(diagonals, a, z)
        else:
            assert pairs is not None
            values = _wigner_laguerre(pairs, a, z)
        flat[start:end] = normalization * values
    return output
