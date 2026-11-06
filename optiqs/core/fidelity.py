"""State fidelity without unnecessary dense matrices or outer products."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import torch


def _is_torch(data: object) -> bool:
    return not isinstance(data, (np.ndarray, sparse.sparray))


def _scalar(data: object) -> complex:
    """Extract a scalar from a 1x1 dense/sparse product or a tensor."""
    if sparse.issparse(data):
        return complex(data[0, 0])
    if hasattr(data, "item"):
        return complex(data.item())
    return complex(data)


def _ket(data: np.ndarray | sparse.sparray | torch.Tensor, qtype: Qtype):
    """Return a ket view; a bra is conjugate-transposed, not copied to dense."""
    return data if qtype == Qtype.ket else data.conj().T


def _inner(left, right) -> complex:
    """Compute <left|right>, keeping sparse vectors sparse."""
    if _is_torch(left):
        import torch

        return complex(torch.vdot(left.reshape(-1), right.reshape(-1)).item())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return complex(np.vdot(left, right))
    if sparse.issparse(left):
        return _scalar(left.conj().T @ right)
    # NumPy's matmul with a SciPy sparse array is backend/version dependent.
    return _scalar(right.conj().T @ left).conjugate()


def _expectation(state, operator) -> complex:
    """Compute <psi|rho|psi>, never constructing |psi><psi|."""
    if isinstance(operator, np.ndarray) and sparse.issparse(state):
        # Only a length-d vector is densified; the d-by-d operator is already
        # dense. NumPy @ sparse is not reliable across NumPy/SciPy versions.
        state = state.toarray()
    return _inner(state, operator @ state)


def _positive_real(value: complex, name: str) -> float:
    """Check that a trace or squared vector norm is positive and real."""
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError(f"{name} must be finite")
    if abs(value.imag) > 1e-6 * max(abs(value.real), 1.0):
        raise ValueError(f"{name} must be real")
    if value.real <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value.real)


def _finite(data) -> bool:
    if sparse.issparse(data):
        return bool(np.isfinite(data.data).all())
    if isinstance(data, np.ndarray):
        return bool(np.isfinite(data).all())
    import torch

    return bool(torch.isfinite(data).all().item())


def _hermitian(data) -> bool:
    """Check Hermiticity without densifying a sparse operator."""
    if sparse.issparse(data):
        skew = data - data.conj().T
        if skew.nnz == 0:
            return True
        scale = max(1.0, float(np.max(np.abs(data.data), initial=0.0)))
        return bool(np.max(np.abs(skew.data)) <= 1e-10 * scale)
    if isinstance(data, np.ndarray):
        return bool(np.allclose(data, data.conj().T, rtol=1e-10, atol=1e-10))
    import torch

    tol = 1e-5 if data.dtype in (torch.float32, torch.complex64) else 1e-10
    return bool(torch.allclose(data, data.conj().T, rtol=tol, atol=tol))


def _diagonal_if_diagonal(data):
    """Return a diagonal view/array if all off-diagonal entries vanish."""
    if sparse.issparse(data):
        entries = data.tocoo()
        if np.any(entries.row != entries.col):
            return None
        return data.diagonal()
    if isinstance(data, np.ndarray):
        diag = np.diagonal(data)
        return diag if np.count_nonzero(data) == np.count_nonzero(diag) else None
    import torch

    diag = torch.diagonal(data)
    if torch.count_nonzero(data).item() == torch.count_nonzero(diag).item():
        return diag
    return None


def _nonnegative_eigenvalues(values, name: str):
    """Clip eigensolver roundoff, rejecting significantly negative values."""
    if isinstance(values, np.ndarray):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite eigenvalues")
        if np.iscomplexobj(values):
            if np.any(np.abs(values.imag) > 1e-10):
                raise ValueError(f"{name} has non-real eigenvalues")
            values = values.real
        values = np.asarray(
            values, dtype=np.float64 if values.dtype.kind not in "f" else None
        )
        eps = np.finfo(values.dtype).eps
        scale = max(
            float(np.max(np.abs(values), initial=0.0)),
            np.finfo(values.dtype).tiny,
        )
        threshold = 32 * eps * scale
        if np.any(values < -threshold):
            raise ValueError(f"{name} must be positive semidefinite")
        # Tiny positive eigenvalues from eigensolver noise are also dropped.
        return np.where(values > threshold, values, 0.0)

    import torch

    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} has non-finite eigenvalues")
    if values.is_complex():
        if torch.any(values.imag.abs() > 1e-5).item():
            raise ValueError(f"{name} has non-real eigenvalues")
        values = values.real
    if not values.is_floating_point():
        values = values.to(dtype=torch.get_default_dtype())
    eps = torch.finfo(values.dtype).eps
    scale = max(float(values.abs().max().item()), torch.finfo(values.dtype).tiny)
    threshold = 32 * eps * scale
    if torch.any(values < -threshold).item():
        raise ValueError(f"{name} must be positive semidefinite")
    return torch.where(values > threshold, values, torch.zeros_like(values))


def _is_rank_one_density(data, trace: float) -> bool:
    """For positive Hermitian rho, tr(rho**2) == tr(rho)**2 iff rank 1.

    This check reads only existing entries: O(nnz) for sparse matrices.
    """
    if sparse.issparse(data):
        purity = float(np.vdot(data.data, data.data).real)
        eps = np.finfo(
            np.float32 if data.dtype in (np.float32, np.complex64) else np.float64
        ).eps
    elif isinstance(data, np.ndarray):
        purity = float(np.vdot(data, data).real)
        eps = np.finfo(
            np.float32 if data.dtype in (np.float32, np.complex64) else np.float64
        ).eps
    else:
        import torch

        purity = float(torch.vdot(data.reshape(-1), data.reshape(-1)).real.item())
        eps = torch.finfo(data.real.dtype).eps
    target = trace * trace
    return abs(purity - target) <= 32 * eps * max(abs(purity), abs(target))


def _trace_product(first, second) -> complex:
    """Compute Tr(first @ second) in O(nnz) for sparse operators."""
    if sparse.issparse(first) and sparse.issparse(second):
        return complex(first.multiply(second.T).sum())
    if sparse.issparse(first):
        entries = first.tocoo()
        return complex(np.sum(entries.data * second[entries.col, entries.row]))
    if sparse.issparse(second):
        entries = second.tocoo()
        return complex(np.sum(entries.data * first[entries.col, entries.row]))
    if isinstance(first, np.ndarray):
        return complex(np.einsum("ij,ji->", first, second))
    import torch

    return complex(torch.sum(first * second.T).item())


def _root_fidelity_numpy(first, second) -> float:
    """Tr sqrt(sqrt(first) @ second @ sqrt(first)) on NumPy/SciPy."""
    first_diag = _diagonal_if_diagonal(first)
    second_diag = _diagonal_if_diagonal(second)

    if first_diag is not None and second_diag is not None:
        p = _nonnegative_eigenvalues(first_diag, "first density operator")
        q = _nonnegative_eigenvalues(second_diag, "second density operator")
        # Exact for commuting diagonal matrices; O(d) storage and arithmetic.
        return float(np.sqrt(p * q).sum())

    if first_diag is None and second_diag is not None:
        first, second = second, first
        first_diag = second_diag

    if first_diag is not None:
        # The eigenvectors of a diagonal density matrix are the basis vectors.
        # Skip its eigendecomposition; only one Hermitian eigensolve is needed.
        roots = np.sqrt(_nonnegative_eigenvalues(first_diag, "density operator"))
        other = second.toarray() if sparse.issparse(second) else second
        sandwich = (roots[:, None] * other) * roots[None, :]
    else:
        if sparse.issparse(first) and isinstance(second, np.ndarray):
            # Diagonalize the matrix already stored dense, instead of
            # unnecessarily densifying the other operand.
            first, second = second, first
        dense = first.toarray() if sparse.issparse(first) else first
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        eigenvalues = _nonnegative_eigenvalues(eigenvalues, "density operator")
        active = eigenvalues > 0
        if not np.any(active):
            return 0.0
        # Compress to the rank of the first operator before the second solve.
        basis = eigenvectors[:, active] * np.sqrt(eigenvalues[active])[None, :]
        sandwich = basis.conj().T @ (second @ basis)

    sandwich = (sandwich + sandwich.conj().T) * 0.5
    spectrum = _nonnegative_eigenvalues(
        np.linalg.eigvalsh(sandwich), "density-operator product"
    )
    return float(np.sqrt(spectrum).sum())


def _root_fidelity_torch(first, second) -> float:
    """Same spectral calculation, entirely on the original torch device."""
    import torch

    first_diag = _diagonal_if_diagonal(first)
    second_diag = _diagonal_if_diagonal(second)

    if first_diag is not None and second_diag is not None:
        p = _nonnegative_eigenvalues(first_diag, "first density operator")
        q = _nonnegative_eigenvalues(second_diag, "second density operator")
        return float(torch.sqrt(p * q).sum().item())

    if first_diag is None and second_diag is not None:
        first, second = second, first
        first_diag = second_diag

    if first_diag is not None:
        roots = torch.sqrt(_nonnegative_eigenvalues(first_diag, "density operator"))
        sandwich = (roots[:, None] * second) * roots[None, :]
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(first)
        eigenvalues = _nonnegative_eigenvalues(eigenvalues, "density operator")
        active = eigenvalues > 0
        if not torch.any(active).item():
            return 0.0
        basis = eigenvectors[:, active] * torch.sqrt(eigenvalues[active])[None, :]
        sandwich = basis.conj().T @ (second @ basis)

    sandwich = (sandwich + sandwich.conj().T) * 0.5
    spectrum = _nonnegative_eigenvalues(
        torch.linalg.eigvalsh(sandwich), "density-operator product"
    )
    return float(torch.sqrt(spectrum).sum().item())


def fidelity(first: Qobj, second: Qobj, *, squared: bool = True) -> float:
    """Return the (squared) quantum-state fidelity between two ``Qobj`` states.

    The convention is ``F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``.
    With ``squared=False``, return its square root (the convention used by
    QuTiP's ``fidelity`` function). Both inputs are normalized by their norm
    (kets/bras) or trace (density operators), *without modifying either*.

    Parameters
    ----------
    first, second : Qobj
        Kets, bras, or positive-semidefinite density operators acting on the
        same Hilbert space (including the same subsystem dimensions).
    squared : bool, default True
        If False, return the unsquared Uhlmann fidelity.

    Returns
    -------
    float
        Fidelity in [0, 1] for physical states.

    Notes
    -----
    Ket-ket uses only an inner product, and ket-density uses a quadratic form,
    so neither constructs a dense ket outer product. Two diagonal density
    operators use an O(d) diagonal-only formula. General mixed-mixed fidelity
    requires dense eigensolvers, O(d**3) time and O(d**2) memory; rank-deficient
    inputs reduce the size of the second eigenproblem. Rank-one density
    operators are detected with a linear-time purity test and use Tr(rho sigma)
    instead. NumPy/SciPy Qobj can be
    freely mixed; PyTorch Qobj must both be on the same device. The fast paths
    assume density operators are positive semidefinite; Hermiticity is checked.
    """
    if not isinstance(first, Qobj) or not isinstance(second, Qobj):
        raise TypeError("fidelity() arguments must be Qobj instances")
    if not isinstance(squared, bool):
        raise TypeError("squared must be a bool")

    def space(state: Qobj) -> tuple[int, ...]:
        if state.qtype == Qtype.ket:
            return state.dims[0]
        if state.qtype == Qtype.bra:
            return state.dims[1]
        if state.dims[0] != state.dims[1]:
            raise ValueError("Density operators must have matching row/column dims")
        return state.dims[0]

    if space(first) != space(second):
        raise ValueError("fidelity() requires matching Hilbert-space dimensions")

    a, b = first._data, second._data
    if _is_torch(a) != _is_torch(b):
        raise TypeError("Cannot mix PyTorch and NumPy/SciPy states")
    if _is_torch(a):
        import torch

        if a.device != b.device:
            raise ValueError("PyTorch states must be on the same device")
        dtype = torch.promote_types(a.dtype, b.dtype)
        if dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            dtype = torch.get_default_dtype()
        a, b = a.to(dtype=dtype), b.to(dtype=dtype)

    if not _finite(a) or not _finite(b):
        raise ValueError("Quantum states must have finite entries")
    pure_a = first.is_vector()
    pure_b = second.is_vector()

    if pure_a:
        a = _ket(a, first.qtype)
        norm_a = _positive_real(_inner(a, a), "first vector norm")
    else:
        if not _hermitian(a):
            raise ValueError("First density operator must be Hermitian")
        norm_a = _positive_real(_scalar(a.diagonal().sum()), "first trace")

    if pure_b:
        b = _ket(b, second.qtype)
        norm_b = _positive_real(_inner(b, b), "second vector norm")
    else:
        if not _hermitian(b):
            raise ValueError("Second density operator must be Hermitian")
        norm_b = _positive_real(_scalar(b.diagonal().sum()), "second trace")

    if pure_a and pure_b:
        overlap = _inner(a, b)
        result = abs(overlap) ** 2 / (norm_a * norm_b)
    elif pure_a or pure_b:
        vector, density = (a, b) if pure_a else (b, a)
        expectation = _expectation(vector, density)
        if abs(expectation.imag) > 1e-6 * max(abs(expectation.real), 1.0):
            raise ValueError("Density-operator expectation must be real")
        result = expectation.real / (norm_a * norm_b)
    elif _is_rank_one_density(a, norm_a) or _is_rank_one_density(b, norm_b):
        # A rank-one density operator is a pure state already expressed as a
        # matrix. Its fidelity with any rho needs no matrix square roots.
        overlap = _trace_product(a, b)
        if abs(overlap.imag) > 1e-6 * max(abs(overlap.real), 1.0):
            raise ValueError("Density-operator overlap must be real")
        result = overlap.real / (norm_a * norm_b)
    else:
        root = (
            _root_fidelity_torch(a, b) if _is_torch(a) else _root_fidelity_numpy(a, b)
        )
        result = root**2 / (norm_a * norm_b)

    if not math.isfinite(result):
        raise ValueError("Fidelity calculation produced a non-finite result")
    if result < -1e-6 or result > 1 + 1e-6:
        raise ValueError("States must be positive semidefinite")
    result = max(0.0, min(1.0, result))
    return result if squared else math.sqrt(result)
