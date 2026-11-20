"""State fidelity without unnecessary dense matrices or outer products."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, overload

import numpy as np
from scipy import sparse

from optiqs.core.low_rank_density import LowRankDensity
from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import torch


def _scalar(data: object) -> complex:
    """Extract a scalar from a 1x1 dense/sparse product or a tensor."""
    if isinstance(data, (sparse.csr_array, sparse.csc_array)):
        return complex(data[0, 0])
    if isinstance(data, (np.ndarray, np.generic)):
        return complex(data.item())
    if isinstance(data, (int, float, complex)):
        return complex(data)
    import torch

    if isinstance(data, torch.Tensor):
        return complex(data.item())
    raise TypeError(f"Expected scalar data, received {type(data)}")


@overload
def _ket(
    data: np.ndarray | sparse.csr_array, qtype: Qtype
) -> np.ndarray | sparse.csr_array: ...


@overload
def _ket(data: torch.Tensor, qtype: Qtype) -> torch.Tensor: ...


def _ket(
    data: np.ndarray | sparse.csr_array | torch.Tensor, qtype: Qtype
) -> np.ndarray | sparse.csr_array | torch.Tensor:
    """Return a ket; sparse bra transposes are converted to CSR, never dense."""
    if qtype == Qtype.ket:
        return data
    if isinstance(data, sparse.csr_array):
        return sparse.csr_array(data.conj().T)
    return data.conj().T


def _inner(
    left: np.ndarray | sparse.csr_array | torch.Tensor,
    right: np.ndarray | sparse.csr_array | torch.Tensor,
) -> complex:
    """Compute <left|right>, keeping sparse vectors sparse."""
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return complex(np.vdot(left, right))
    if isinstance(left, sparse.csr_array) and isinstance(
        right, (np.ndarray, sparse.csr_array)
    ):
        return _scalar(left.conj().T @ right)
    if isinstance(left, np.ndarray) and isinstance(right, sparse.csr_array):
        # NumPy @ SciPy sparse is backend/version dependent.
        return _scalar(right.conj().T @ left).conjugate()
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return complex(torch.vdot(left.reshape(-1), right.reshape(-1)).item())
    raise TypeError("Cannot mix NumPy/SciPy and PyTorch states")


def _expectation(
    state: np.ndarray | sparse.csr_array | torch.Tensor,
    operator: np.ndarray | sparse.csr_array | torch.Tensor,
) -> complex:
    """Compute <psi|rho|psi>, never constructing |psi><psi|."""
    if isinstance(operator, np.ndarray) and isinstance(state, sparse.csr_array):
        # Only a length-d vector is densified; the d-by-d operator is already
        # dense. NumPy @ sparse is not reliable across NumPy/SciPy versions.
        state = state.toarray()
    if isinstance(operator, (np.ndarray, sparse.csr_array)) and isinstance(
        state, (np.ndarray, sparse.csr_array)
    ):
        product = operator @ state
        if isinstance(product, np.ndarray):
            return _inner(state, product)
        if isinstance(product, sparse.csr_array):
            # CSR @ CSR returns CSR; reuse it without densifying or rewrapping.
            return _inner(state, product)
        raise TypeError("Unexpected NumPy/SciPy matrix-product result")
    import torch

    if isinstance(operator, torch.Tensor) and isinstance(state, torch.Tensor):
        return _inner(state, operator @ state)
    raise TypeError("Cannot mix NumPy/SciPy and PyTorch states")


def _positive_real(value: complex, name: str) -> float:
    """Check that a trace or squared vector norm is positive and real."""
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ValueError(f"{name} must be finite")
    if abs(value.imag) > 1e-6 * max(abs(value.real), 1.0):
        raise ValueError(f"{name} must be real")
    if value.real <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value.real)


def _finite(data: np.ndarray | sparse.csr_array | torch.Tensor) -> bool:
    if isinstance(data, sparse.csr_array):
        return bool(np.isfinite(data.data).all())
    if isinstance(data, np.ndarray):
        return bool(np.isfinite(data).all())
    import torch

    return bool(torch.isfinite(data).all().item())


def _hermitian(data: np.ndarray | sparse.csr_array | torch.Tensor) -> bool:
    """Check Hermiticity without densifying a sparse operator."""
    if isinstance(data, sparse.csr_array):
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


@overload
def _diagonal_if_diagonal(
    data: np.ndarray | sparse.csr_array,
) -> np.ndarray | None: ...


@overload
def _diagonal_if_diagonal(data: torch.Tensor) -> torch.Tensor | None: ...


def _diagonal_if_diagonal(
    data: np.ndarray | sparse.csr_array | torch.Tensor,
) -> np.ndarray | torch.Tensor | None:
    """Return a diagonal view/array if all off-diagonal entries vanish."""
    if isinstance(data, sparse.csr_array):
        entries = data.tocoo()
        if np.any(entries.row != entries.col):
            return None
        return data.diagonal()
    if isinstance(data, np.ndarray):
        diag = np.diagonal(data)
        return diag if np.count_nonzero(data) == np.count_nonzero(diag) else None
    import torch

    torch_diag = torch.diagonal(data)
    if torch.count_nonzero(data).item() == torch.count_nonzero(torch_diag).item():
        return torch_diag
    return None


@overload
def _nonnegative_eigenvalues(values: np.ndarray, name: str) -> np.ndarray: ...


@overload
def _nonnegative_eigenvalues(values: torch.Tensor, name: str) -> torch.Tensor: ...


def _nonnegative_eigenvalues(
    values: np.ndarray | torch.Tensor, name: str
) -> np.ndarray | torch.Tensor:
    """Clip eigensolver roundoff, rejecting significantly negative values."""
    if isinstance(values, np.ndarray):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} has non-finite eigenvalues")
        if np.iscomplexobj(values):
            if np.any(np.abs(np.imag(values)) > 1e-10):
                raise ValueError(f"{name} has non-real eigenvalues")
            values = np.real(values)
        # Preserve float32/float64 precision, but cast integer inputs to float.
        # NumPy functions narrow reliably in both Pyright and mypy.
        if np.issubdtype(values.dtype, np.floating):
            real_values = np.asarray(values)
        else:
            real_values = np.asarray(values, dtype=np.float64)
        eps = np.finfo(real_values.dtype).eps
        scale = max(
            float(np.max(np.abs(real_values), initial=0.0)),
            np.finfo(real_values.dtype).tiny,
        )
        threshold = 32 * eps * scale
        if np.any(real_values < -threshold):
            raise ValueError(f"{name} must be positive semidefinite")
        # Tiny positive eigenvalues from eigensolver noise are also dropped.
        return np.where(real_values > threshold, real_values, 0.0)

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


def _is_rank_one_density(
    data: np.ndarray | sparse.csr_array | torch.Tensor, trace: float
) -> bool:
    """For positive Hermitian rho, tr(rho**2) == tr(rho)**2 iff rank 1.

    This check reads only existing entries: O(nnz) for sparse matrices.
    """
    if isinstance(data, sparse.csr_array):
        purity = float(np.vdot(data.data, data.data).real)
        eps = float(
            np.finfo(
                np.float32 if data.dtype in (np.float32, np.complex64) else np.float64
            ).eps
        )
    elif isinstance(data, np.ndarray):
        purity = float(np.vdot(data, data).real)
        eps = float(
            np.finfo(
                np.float32 if data.dtype in (np.float32, np.complex64) else np.float64
            ).eps
        )
    else:
        import torch

        purity = float(torch.vdot(data.reshape(-1), data.reshape(-1)).real.item())
        eps = float(torch.finfo(data.real.dtype).eps)
    target = trace * trace
    return abs(purity - target) <= 32 * eps * max(abs(purity), abs(target))


def _trace_product(
    first: np.ndarray | sparse.csr_array | torch.Tensor,
    second: np.ndarray | sparse.csr_array | torch.Tensor,
) -> complex:
    """Compute Tr(first @ second) in O(nnz) for sparse operators."""
    if isinstance(first, sparse.csr_array) and isinstance(second, sparse.csr_array):
        return complex(first.multiply(second.T).sum())
    if isinstance(first, sparse.csr_array) and isinstance(second, np.ndarray):
        entries = first.tocoo()
        return complex(np.sum(entries.data * second[entries.col, entries.row]))
    if isinstance(second, sparse.csr_array) and isinstance(first, np.ndarray):
        entries = second.tocoo()
        return complex(np.sum(entries.data * first[entries.col, entries.row]))
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return complex(np.einsum("ij,ji->", first, second))
    import torch

    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return complex(torch.sum(first * second.T).item())
    raise TypeError("Cannot mix NumPy/SciPy and PyTorch states")


def _root_fidelity_numpy(
    first: np.ndarray | sparse.csr_array, second: np.ndarray | sparse.csr_array
) -> float:
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
        other = second.toarray() if isinstance(second, sparse.csr_array) else second
        sandwich = (roots[:, None] * other) * roots[None, :]
    else:
        if isinstance(first, sparse.csr_array) and isinstance(second, np.ndarray):
            # Diagonalize the matrix already stored dense, instead of
            # unnecessarily densifying the other operand.
            first, second = second, first
        dense = first.toarray() if isinstance(first, sparse.csr_array) else first
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


def _root_fidelity_torch(first: torch.Tensor, second: torch.Tensor) -> float:
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


def _small_factor_product(
    left: np.ndarray | sparse.csr_array,
    right: np.ndarray | sparse.csr_array,
) -> np.ndarray:
    """Compute X*Y in rank space, densifying at most the small result.

    In particular, dense @ CSR is avoided because its behavior depends on
    the NumPy/SciPy versions. Never materialize X @ X* or Y @ Y*.
    """
    if isinstance(left, np.ndarray) and isinstance(right, sparse.csr_array):
        product = right.conj().T @ left
        return np.asarray(product).conj().T
    product = left.conj().T @ right
    # SciPy's type stubs expose `sparray` for matrix products, but concrete
    # CSR/CSC arrays provide `toarray()`. Narrow before calling that method.
    if isinstance(product, (sparse.csr_array, sparse.csc_array)):
        return product.toarray()
    if isinstance(product, np.ndarray):
        return product
    raise TypeError("Unexpected NumPy/SciPy factor-product result")


def _fidelity_with_low_rank(
    first: Qobj | LowRankDensity,
    second: Qobj | LowRankDensity,
    *,
    squared: bool,
) -> float:
    """Use the nonzero spectrum of X*Y, not a D-by-D matrix square root."""
    if isinstance(first, LowRankDensity) and isinstance(second, LowRankDensity):
        if first.dims != second.dims:
            raise ValueError("fidelity() requires matching Hilbert-space dimensions")
        a = first._numpy_factors()
        b = second._numpy_factors()
        norm_a = _positive_real(complex(first.trace), "first trace")
        norm_b = _positive_real(complex(second.trace), "second trace")
        root = float(np.linalg.svd(_small_factor_product(a, b), compute_uv=False).sum())
    else:
        low_rank = first if isinstance(first, LowRankDensity) else second
        other = second if isinstance(first, LowRankDensity) else first
        assert isinstance(low_rank, LowRankDensity) and isinstance(other, Qobj)
        other_dims = (
            other.dims[0]
            if other.qtype == Qtype.ket
            else other.dims[1]
            if other.qtype == Qtype.bra
            else other.dims[0]
        )
        if low_rank.dims != other_dims or (
            other.qtype == Qtype.oper and other.dims[0] != other.dims[1]
        ):
            raise ValueError("fidelity() requires matching Hilbert-space dimensions")
        data = other._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("Cannot mix PyTorch and NumPy/SciPy states")
        if not _finite(data):
            raise ValueError("Quantum states must have finite entries")
        factors = low_rank._numpy_factors()
        low_rank_norm = _positive_real(complex(low_rank.trace), "low-rank trace")
        if other.is_vector():
            ket = _ket(data, other.qtype)
            other_norm = _positive_real(_inner(ket, ket), "vector norm")
            projected = _small_factor_product(factors, ket)
            # A vector is a rank-one factor. Its only singular value is the
            # Euclidean norm of X*ket, avoiding even a small SVD.
            root = float(np.linalg.norm(projected))
        else:
            if not _hermitian(data):
                raise ValueError("Density operator must be Hermitian")
            other_norm = _positive_real(_scalar(data.diagonal().sum()), "density trace")
            # X* rho X is only rank-by-rank even when rho is large.
            if isinstance(data, np.ndarray) and isinstance(factors, sparse.csr_array):
                dense_factors = factors.toarray()
                reduced = dense_factors.conj().T @ (data @ dense_factors)
            else:
                applied = data @ factors
                right: np.ndarray | sparse.csr_array
                # Matrix multiplication is typed as a general SciPy sparse
                # base class. Normalize its concrete storage to CSR before
                # passing it to the rank-space helper.
                if isinstance(applied, (np.ndarray, sparse.csr_array)):
                    right = applied
                elif isinstance(applied, sparse.csc_array):
                    right = sparse.csr_array(applied)
                else:
                    raise TypeError("Unexpected density-factor product")
                reduced = _small_factor_product(factors, right)
            reduced = (reduced + reduced.conj().T) * 0.5
            spectrum = _nonnegative_eigenvalues(
                np.linalg.eigvalsh(reduced), "density-operator product"
            )
            root = float(np.sqrt(spectrum).sum())
        norm_a, norm_b = low_rank_norm, other_norm

    # Divide the root fidelity by each norm's square root separately so a
    # product of two large but finite traces cannot overflow float64.
    root = (root / math.sqrt(norm_a)) / math.sqrt(norm_b)
    result = root * root
    if not math.isfinite(result) or result < -1e-6 or result > 1 + 1e-6:
        raise ValueError("States must be positive semidefinite")
    result = max(0.0, min(1.0, result))
    return result if squared else math.sqrt(result)


def fidelity(
    first: Qobj | LowRankDensity,
    second: Qobj | LowRankDensity,
    *,
    squared: bool = True,
) -> float:
    """Return the (squared) quantum-state fidelity between two states.

    The convention is ``F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))**2``.
    With ``squared=False``, return its square root (the convention used by
    QuTiP's ``fidelity`` function). Both inputs are normalized by their norm
    (kets/bras) or trace (density operators), *without modifying either*.

    Parameters
    ----------
    first, second : Qobj or LowRankDensity
        Kets, bras, positive-semidefinite density operators or low-rank
        factors acting on the same Hilbert space (including subsystem dims).
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
    For two ``LowRankDensity`` objects ``rho=X X*`` and ``sigma=Y Y*``,
    compute ``||X*Y||_*`` using a rank-by-rank SVD, without materializing
    either density matrix. A low-rank state can also be compared to a
    NumPy/CSR ``Qobj``; a mixed ``Qobj`` uses the small matrix ``X* rho X``.
    """
    if not isinstance(squared, bool):
        raise TypeError("squared must be a bool")
    if not isinstance(first, (Qobj, LowRankDensity)) or not isinstance(
        second, (Qobj, LowRankDensity)
    ):
        raise TypeError("fidelity() arguments must be Qobj or LowRankDensity")
    if isinstance(first, LowRankDensity) or isinstance(second, LowRankDensity):
        return _fidelity_with_low_rank(first, second, squared=squared)

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
    numpy_a = isinstance(a, (np.ndarray, sparse.csr_array))
    numpy_b = isinstance(b, (np.ndarray, sparse.csr_array))
    if numpy_a != numpy_b:
        raise TypeError("Cannot mix PyTorch and NumPy/SciPy states")
    if not numpy_a:
        import torch

        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise TypeError("Unsupported Qobj data type")
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
        if isinstance(a, (np.ndarray, sparse.csr_array)) and isinstance(
            b, (np.ndarray, sparse.csr_array)
        ):
            root = _root_fidelity_numpy(a, b)
        else:
            import torch

            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError("Unsupported Qobj data type")
            root = _root_fidelity_torch(a, b)
        result = root**2 / (norm_a * norm_b)

    if not math.isfinite(result):
        raise ValueError("Fidelity calculation produced a non-finite result")
    if result < -1e-6 or result > 1 + 1e-6:
        raise ValueError("States must be positive semidefinite")
    result = max(0.0, min(1.0, result))
    return result if squared else math.sqrt(result)


class PreparedFidelityTarget:
    """Precompute a NumPy/SciPy density target's matrix square root.

    Use this when comparing many different states against the same *mixed*
    target. The target is copied at construction and eigendecomposed once.
    ``fidelity(other)`` follows the same squared-fidelity convention as the
    module-level function; vectors use its cheaper quadratic-form path.

    General targets need O(D**2) storage. A diagonal target instead caches
    only O(D) square-root entries. PyTorch states are deliberately unsupported
    by this optional CPU-only API; use ``fidelity`` for device tensors.
    """

    def __init__(self, target: Qobj) -> None:
        if not isinstance(target, Qobj) or target.qtype != Qtype.oper:
            raise TypeError("target must be a density-operator Qobj")
        if target.dims[0] != target.dims[1]:
            raise ValueError("density operator must have matching dimensions")
        data = target._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("PreparedFidelityTarget supports NumPy/SciPy only")
        if not _finite(data) or not _hermitian(data):
            raise ValueError("target must be finite and Hermitian")
        self._trace = _positive_real(_scalar(data.diagonal().sum()), "target trace")
        self._target = target.copy()
        self._dims = target.dims[0]
        diagonal = _diagonal_if_diagonal(data)
        self._diagonal_roots: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        if diagonal is not None:
            self._diagonal_roots = np.sqrt(
                _nonnegative_eigenvalues(diagonal, "target density operator")
            )
        else:
            dense = data.toarray() if isinstance(data, sparse.csr_array) else data
            eigenvalues, eigenvectors = np.linalg.eigh(dense)
            eigenvalues = _nonnegative_eigenvalues(
                eigenvalues, "target density operator"
            )
            active = eigenvalues > 0
            self._basis = (
                eigenvectors[:, active] * np.sqrt(eigenvalues[active])[None, :]
            )

    def fidelity(self, other: Qobj, *, squared: bool = True) -> float:
        """Compare another state to the prepared target without re-diagonalizing."""
        if not isinstance(other, Qobj):
            raise TypeError("other must be a Qobj")
        if not isinstance(squared, bool):
            raise TypeError("squared must be a bool")
        dims = (
            other.dims[0]
            if other.qtype == Qtype.ket
            else other.dims[1]
            if other.qtype == Qtype.bra
            else other.dims[0]
        )
        if dims != self._dims or (
            other.qtype == Qtype.oper and other.dims[0] != other.dims[1]
        ):
            raise ValueError("states must have matching Hilbert-space dimensions")
        data = other._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("PreparedFidelityTarget supports NumPy/SciPy only")
        if other.is_vector():
            return fidelity(self._target, other, squared=squared)
        if not _finite(data) or not _hermitian(data):
            raise ValueError("other density operator must be finite and Hermitian")
        norm = _positive_real(_scalar(data.diagonal().sum()), "other trace")

        if self._diagonal_roots is not None:
            roots = self._diagonal_roots
            dense = data.toarray() if isinstance(data, sparse.csr_array) else data
            sandwich = (roots[:, None] * dense) * roots[None, :]
        else:
            assert self._basis is not None
            sandwich = self._basis.conj().T @ (data @ self._basis)
        sandwich = (sandwich + sandwich.conj().T) * 0.5
        eigenvalues = _nonnegative_eigenvalues(
            np.linalg.eigvalsh(sandwich), "density-operator product"
        )
        root = float(np.sqrt(eigenvalues).sum())
        result = root * root / (self._trace * norm)
        if not math.isfinite(result) or result < -1e-6 or result > 1 + 1e-6:
            raise ValueError("states must be positive semidefinite")
        result = max(0.0, min(1.0, result))
        return result if squared else math.sqrt(result)

    __call__ = fidelity
