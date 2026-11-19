"""Opt-in low-rank density representation ``rho = factors @ factors.dag()``.

Only NumPy/SciPy are supported.  Unlike ``Qobj`` the representation stores
``D x rank`` factors, so it is especially useful for pure-loss trajectories
that have few Kraus branches relative to the full Hilbert-space dimension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
from scipy import sparse
from scipy.special import gammaln

from optiqs.core.apply_local import LocalGate, _apply_rows, _selected_modes
from optiqs.core.q_obj import Qobj, Qtype


class LowRankDensity:
    """Store a positive-semidefinite (not necessarily normalized) ``rho=XX*``.

    A dense factor needs ``O(D * rank)`` memory, instead of ``O(D**2)``.
    The ``rank`` property is the number of columns, not necessarily the
    mathematical rank. ``to_qobj()`` explicitly materializes the density.

    This is an *opt-in* representation: it is not a ``Qobj`` subclass and
    existing ``photon_loss`` callers keep returning ordinary ``Qobj``.
    """

    def __init__(
        self,
        factors: np.ndarray | sparse.csr_array,
        dims: Sequence[int],
        *,
        tol: float = 1e-17,
    ) -> None:
        if not isinstance(factors, (np.ndarray, sparse.csr_array)):
            raise TypeError("factors must be a 2D NumPy or SciPy CSR array")
        if factors.ndim != 2 or factors.shape[1] < 1:
            raise ValueError("factors must have at least one column")
        if not dims or any(
            isinstance(dim, (bool, np.bool_))
            or not isinstance(dim, Integral)
            or dim <= 0
            for dim in dims
        ):
            raise ValueError("dims must be a sequence of positive integers")
        subsystems = tuple(int(d) for d in dims)
        if math.prod(subsystems) != factors.shape[0]:
            raise ValueError("dims are incompatible with factors' row count")
        entries = factors.data if isinstance(factors, sparse.csr_array) else factors
        if not np.isfinite(entries).all():
            raise ValueError("factors must be finite")
        if np.iscomplexobj(factors):
            dtype = np.result_type(factors.dtype, np.complex64)
        else:
            dtype = np.result_type(factors.dtype, np.float64)
        # Qobj defensively copies caller-provided storage.
        self._factor = Qobj(
            factors.astype(dtype, copy=False),
            dims=(subsystems, (factors.shape[1],)),
            qtype=Qtype.oper,
            tol=tol,
        )

    def _numpy_factors(self) -> np.ndarray | sparse.csr_array:
        """Narrow Qobj's NumPy/CSR/Torch storage to the supported backends.

        All public constructors require NumPy/CSR data; this guard also
        catches future changes that accidentally introduce Torch factors.
        """
        data = self._factor._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("LowRankDensity requires NumPy/SciPy-backed factors")
        return data

    @classmethod
    def _adopt(cls, factor: Qobj) -> LowRankDensity:
        instance = cls.__new__(cls)
        instance._factor = factor
        return instance

    @classmethod
    def from_ket(cls, state: Qobj) -> LowRankDensity:
        """Normalize a NumPy/CSR ket or bra and store its single factor."""
        if not isinstance(state, Qobj) or not state.is_vector():
            raise TypeError("state must be a ket or bra Qobj")
        data = state._data
        if not isinstance(data, (np.ndarray, sparse.csr_array)):
            raise TypeError("LowRankDensity supports NumPy/SciPy states only")
        dims = state.dims[0] if state.qtype == Qtype.ket else state.dims[1]
        ket: np.ndarray | sparse.csr_array
        if isinstance(data, sparse.csr_array):
            # CSR's conjugate transpose may be CSC: normalize its storage.
            ket = (
                data
                if state.qtype == Qtype.ket
                else sparse.csr_array(cast(Any, data.conj().T))
            )
            values = ket.data
        else:
            ket = data if state.qtype == Qtype.ket else data.conj().T
            values = ket
        norm = float(np.linalg.norm(values))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("state must have a finite positive norm")
        return cls(ket / norm, dims, tol=state._tol)

    @property
    def dims(self) -> tuple[int, ...]:
        return self._factor.dims[0]

    @property
    def rank(self) -> int:
        return self._factor._data.shape[1]

    @property
    def factors(self) -> np.ndarray | sparse.csr_array:
        """Return an independent copy of the factors (not a writable view)."""
        return self._numpy_factors().copy()

    @property
    def trace(self) -> float:
        data = self._numpy_factors()
        values = data.data if isinstance(data, sparse.csr_array) else data
        return float(np.vdot(values, values).real)

    def to_qobj(self) -> Qobj:
        """Materialize the full density; use only when ``D x D`` fits."""
        factors = self._numpy_factors()
        density: np.ndarray | sparse.csr_array
        if isinstance(factors, sparse.csr_array):
            # SciPy sparse matmul has an imprecise `sparray` stub. At runtime
            # the resulting product is a sparse array, normalized to CSR.
            density = sparse.csr_array(cast(Any, factors @ factors.conj().T))
        else:
            density = factors @ factors.conj().T
        return Qobj._from_owned_data(
            density,
            dims=(self.dims, self.dims),
            qtype=Qtype.oper,
            tol=self._factor._tol,
        )

    def expect(self, operator: Qobj) -> complex:
        """Compute ``Tr(rho @ operator)`` without materializing ``rho``."""
        if not isinstance(operator, Qobj) or operator.dims != (
            self.dims,
            self.dims,
        ):
            raise ValueError("operator must act on the same Hilbert space")
        operator_data = operator._data
        if operator.qtype != Qtype.oper or not isinstance(
            operator_data, (np.ndarray, sparse.csr_array)
        ):
            raise TypeError("operator must be a NumPy/SciPy Qobj operator")
        factors = self._numpy_factors()
        product = operator_data @ factors
        if isinstance(product, sparse.sparray):
            # Narrow `sparray` for both type checkers and SciPy's CSR methods.
            product = sparse.csr_array(cast(Any, product))
        if isinstance(factors, sparse.csr_array) and isinstance(
            product, sparse.csr_array
        ):
            return complex(factors.conj().multiply(product).sum())
        left = factors.toarray() if isinstance(factors, sparse.csr_array) else factors
        right = product.toarray() if isinstance(product, sparse.csr_array) else product
        return complex(np.vdot(left, right))

    def apply_local(
        self, gate: LocalGate, modes: int | Sequence[int]
    ) -> LowRankDensity:
        """Apply a local operator to *all factors* without forming ``rho``."""
        from optiqs.core.expm import Expm
        from optiqs.core.lazy_cached_operator import LazyCachedOperator

        if not isinstance(gate, (Qobj, Expm, LazyCachedOperator)):
            raise TypeError("gate must be a local operator")
        selected = _selected_modes(modes, len(self.dims))
        required = tuple(self.dims[i] for i in selected)
        if gate.qtype != Qtype.oper or gate.dims != (required, required):
            raise ValueError("gate subsystem dimensions do not match selected modes")
        if isinstance(gate, Qobj) and not isinstance(
            gate._data, (np.ndarray, sparse.csr_array)
        ):
            raise TypeError("LowRankDensity requires a NumPy/SciPy-backed gate")
        return self._adopt(_apply_rows(self._factor, gate, selected))

    def photon_loss(self, eta: float, mode: int = 0) -> LowRankDensity:
        """Apply one pure-loss channel to factors (rank may grow by ``dim``).

        The final factor columns are the Kraus images of the incoming factor
        columns. CSR factors remain sparse; transitions are built in bounded
        batches. Zero Kraus columns are discarded, but general linear
        dependencies are not (call ``compress`` for approximate truncation).
        """
        if isinstance(eta, (bool, np.bool_)) or not isinstance(eta, Real):
            raise TypeError("eta must be a real transmissivity")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral):
            raise TypeError("mode must be an integer")
        if not 0 <= mode < len(self.dims):
            raise IndexError("mode is outside the available subsystems")
        if eta == 1:
            return self._adopt(self._factor.copy())

        dim = self.dims[mode]
        stride = math.prod(self.dims[mode + 1 :])
        factors = self._numpy_factors()
        rows_dim, rank = factors.shape
        log_factorials = gammaln(np.arange(dim, dtype=np.float64) + 1)
        # Build only up to the highest occupied Fock number. If the factor
        # is sparse and high Fock levels are absent, this saves columns.
        if isinstance(factors, sparse.csr_array):
            occupied = factors.tocoo(copy=False)
            positions = occupied.row
            columns = occupied.col
            amplitudes = occupied.data
        else:
            positions, columns = np.nonzero(factors)
            amplitudes = factors[positions, columns]
        if positions.size == 0:
            raise ValueError("factors must have positive trace")
        photons = (positions // stride) % dim
        losses = int(np.max(photons)) + 1
        result_shape = (rows_dim, rank * losses)
        dtype = factors.dtype
        output = sparse.csr_array(result_shape, dtype=dtype)
        log_eta = math.log(eta) if eta > 0 else 0.0
        log_loss = math.log1p(-eta) if eta > 0 else 0.0
        counts = np.asarray(
            photons + 1 if eta > 0 else np.ones_like(photons), dtype=np.int64
        )
        cumulative = np.cumsum(counts, dtype=np.int64)
        start = 0
        while start < positions.size:
            previous = int(cumulative[start - 1]) if start else 0
            stop = max(
                start + 1,
                int(np.searchsorted(cumulative, previous + 250_000, side="right")),
            )
            parents = np.repeat(
                np.arange(start, stop, dtype=np.intp), counts[start:stop]
            )
            if eta == 0:
                lost = photons[parents]
                weights = np.ones(parents.size)
            else:
                widths = counts[start:stop]
                offsets = np.repeat(np.cumsum(widths) - widths, widths)
                lost = np.arange(parents.size) - offsets
                number = photons[parents]
                log_prob = (
                    log_factorials[number]
                    - log_factorials[lost]
                    - log_factorials[number - lost]
                    + (number - lost) * log_eta
                    + lost * log_loss
                )
                weights = np.exp(0.5 * log_prob)
            contributions = amplitudes[parents] * weights
            active = contributions != 0
            if np.any(active):
                part = sparse.csr_array(
                    (
                        contributions[active].astype(dtype, copy=False),
                        (
                            positions[parents[active]] - lost[active] * stride,
                            columns[parents[active]] * losses + lost[active],
                        ),
                    ),
                    shape=result_shape,
                )
                output = output + part
            start = stop
        # Eliminate structurally empty Kraus columns without densification.
        active_columns = np.unique(output.indices)
        if not active_columns.size:
            raise ValueError("loss produced a zero-trace state")
        output = sparse.csr_array(output[:, active_columns])
        if isinstance(factors, np.ndarray):
            return LowRankDensity(output.toarray(), self.dims, tol=self._factor._tol)
        return LowRankDensity(output, self.dims, tol=self._factor._tol)

    def compress(self, max_rank: int, *, relative_tol: float = 0.0) -> LowRankDensity:
        """Optionally reduce rank via SVD; discarded singular values lose trace.

        This is an approximate operation and may densify CSR factors. Use
        ``relative_tol=0`` to keep the largest ``max_rank`` singular values.
        """
        if (
            isinstance(max_rank, bool)
            or not isinstance(max_rank, Integral)
            or max_rank < 1
        ):
            raise ValueError("max_rank must be a positive integer")
        if not math.isfinite(relative_tol) or relative_tol < 0:
            raise ValueError("relative_tol must be finite and nonnegative")
        factors = self._numpy_factors()
        matrix = factors.toarray() if isinstance(factors, sparse.csr_array) else factors
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        keep = min(
            int(max_rank), max(1, int(np.count_nonzero(s > s[0] * relative_tol)))
        )
        return LowRankDensity(u[:, :keep] * s[:keep], self.dims, tol=self._factor._tol)


def photon_loss_low_rank(
    state: Qobj | LowRankDensity, eta: float, mode: int = 0
) -> LowRankDensity:
    """Opt-in low-rank photon loss for a pure Qobj or existing factors."""
    if isinstance(state, LowRankDensity):
        return state.photon_loss(eta, mode)
    return LowRankDensity.from_ket(state).photon_loss(eta, mode)
