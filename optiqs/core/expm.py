"""Lazy matrix exponential operators for NumPy/SciPy-backed ``Qobj`` objects.

``Expm`` represents ``exp(scale * generator)`` without constructing the
exponential matrix.  Applying the operator uses SciPy's ``expm_multiply``;
this is particularly useful for large, sparse beam-splitter generators.
"""

from __future__ import annotations

import math
from numbers import Number
from types import NotImplementedType
from typing import Any, Callable, TypeAlias, cast

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from optiqs.core.q_obj import Qobj, Qtype

# Qobj normalizes SciPy sparse inputs to CSR.  Avoid annotating the stored
# generator as `sparray`: scipy-stubs does not expose shape, dtype, transpose,
# etc. on that abstract class, though it supports those on concrete CSR arrays.
_MatrixData: TypeAlias = np.ndarray | sparse.csr_array[Any, tuple[int, int]]


def _transpose(data: _MatrixData) -> _MatrixData:
    """Transpose while preserving a concrete CSR/NumPy array type."""
    if isinstance(data, np.ndarray):
        return data.T
    # A CSR transpose is CSC; keep the internal sparse representation CSR.
    return sparse.csr_array(cast(Any, data.T))


def _expm_action(a: _MatrixData, b: _MatrixData) -> _MatrixData:
    """Apply exp(a) to b, normalizing SciPy's result to NumPy or CSR.

    SciPy's expm_multiply accepts sparse right-hand sides and preserves their
    sparse representation at runtime.  Current scipy-stubs overloads are more
    restrictive for combinations of dense/sparse matrices.  The cast is
    deliberately limited to this one third-party API boundary, followed by a
    runtime type check of its result.
    """
    scipy_call = cast(Callable[[_MatrixData, _MatrixData], object], expm_multiply)
    result = scipy_call(a, b)
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, sparse.sparray):
        return sparse.csr_array(cast(Any, result))
    raise TypeError(f"Unexpected expm_multiply output type: {type(result)}")


class Expm:
    """A lazy representation of ``exp(scale * generator)``.

    ``generator`` must be a square, NumPy/SciPy-backed operator ``Qobj`` with
    matching input and output subsystem dimensions.  Its data is copied on
    construction.  ``Expm @ state`` and ``state @ Expm`` avoid materializing
    the exponential, including for sparse kets and density matrices.

    ``to_qobj()`` explicitly creates the full exponential, which can require
    quadratic memory.  PyTorch-backed objects are not supported.
    """

    def __init__(
        self, generator: Qobj, scale: float | complex | np.number[Any] = 1.0
    ) -> None:
        if not isinstance(generator, Qobj):
            raise TypeError("generator must be a Qobj")
        if generator.qtype != Qtype.oper or generator.dims[0] != generator.dims[1]:
            raise ValueError(
                "generator must be a square Qtype.oper with matching subsystem dims"
            )
        source = generator._data
        if not isinstance(source, (np.ndarray, sparse.csr_array)):
            raise TypeError("Expm supports only NumPy/SciPy-backed Qobj generators")
        if isinstance(scale, (bool, np.bool_)) or not isinstance(scale, Number):
            raise TypeError("scale must be a finite real or complex number")

        # Converting once to Python complex removes `numbers.Number` from the
        # arithmetic/np.isfinite type-checking path.  Keep real scales real so
        # a real generator does not unnecessarily become complex-valued.
        # Numeric ABCs (and NumPy's base number type) do not statically
        # promise __complex__, although accepted scalar instances support it.
        scalar = complex(cast(float | complex, scale))
        if not (math.isfinite(scalar.real) and math.isfinite(scalar.imag)):
            raise ValueError("scale must be finite")
        factor: float | complex = scalar.real if scalar.imag == 0 else scalar

        self._data: _MatrixData = source.copy()
        # Promote integer and unsigned generators before negating or scaling.
        if self._data.dtype.kind not in "fc":
            self._data = self._data.astype(np.float64)
        if factor != 1:
            self._data = self._data * factor
        self._dims = generator.dims
        self._tol = generator._tol

    @property
    def dims(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._dims

    @property
    def shape(self) -> tuple[int, int]:
        shape = self._data.shape
        return shape[0], shape[1]

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._data.dtype

    @property
    def qtype(self) -> Qtype:
        return Qtype.oper

    def is_sparse(self) -> bool:
        """Whether the *generator* is stored as a SciPy CSR array."""
        return isinstance(self._data, sparse.csr_array)

    def _apply(self, other: Qobj, *, from_right: bool) -> Qobj:
        other_data = other._data
        if not isinstance(other_data, (np.ndarray, sparse.csr_array)):
            raise TypeError("Expm supports only NumPy/SciPy-backed Qobj operands")

        if from_right:
            if other.dims[1] != self.dims[0]:
                raise ValueError("Qobj @ Expm requires left.dims[1] == Expm.dims[0]")
            # B @ exp(A) = (exp(A.T) @ B.T).T, without creating exp(A).
            output = _transpose(
                _expm_action(_transpose(self._data), _transpose(other_data))
            )
            dims = (other.dims[0], self.dims[1])
        else:
            if self.dims[1] != other.dims[0]:
                raise ValueError("Expm @ Qobj requires Expm.dims[1] == right.dims[0]")
            output = _expm_action(self._data, other_data)
            dims = (self.dims[0], other.dims[1])

        return Qobj(
            output,
            dims=dims,
            qtype=other.qtype,
            tol=max(self._tol, other._tol),
        )

    def __matmul__(self, other: object) -> Qobj:
        if not isinstance(other, Qobj):
            raise TypeError("Expm @ operand requires a Qobj")
        return self._apply(other, from_right=False)

    def __rmatmul__(self, other: object) -> Qobj | NotImplementedType:
        if not isinstance(other, Qobj):
            return NotImplemented
        return self._apply(other, from_right=True)

    def __mul__(self, other: object) -> Qobj:
        """Support OptiQS's ``Qobj * Qobj`` matrix-product convention."""
        return self.__matmul__(other)

    def __rmul__(self, other: object) -> Qobj | NotImplementedType:
        return self.__rmatmul__(other)

    def dag(self) -> Expm:
        """Lazy adjoint, ``exp(A).dag() == exp(A.dag())``."""
        data = self._data
        adjoint: _MatrixData
        if isinstance(data, np.ndarray):
            adjoint = data.conj().T
        else:
            adjoint = sparse.csr_array(cast(Any, data.conj().T))
        return self._from_data(adjoint)

    def inverse(self) -> Expm:
        """Lazy inverse, ``exp(-A)``."""
        return self._from_data(-self._data)

    def _from_data(self, data: _MatrixData) -> Expm:
        """Create a new lazy operator without Qobj's small-value pruning."""
        obj = object.__new__(Expm)
        obj._data = data.copy()
        obj._dims = self._dims
        obj._tol = self._tol
        return obj

    def to_qobj(self) -> Qobj:
        """Materialize the full exponential as a dense ``Qobj`` (O(N**2))."""
        size = self.shape[0]
        dtype = np.result_type(self.dtype, np.float32)
        identity = np.eye(size, dtype=dtype)
        result = _expm_action(self._data, identity)
        dense_result = (
            result.toarray() if isinstance(result, sparse.csr_array) else result
        )
        return Qobj(dense_result, dims=self.dims, qtype=Qtype.oper, tol=self._tol)

    def __repr__(self) -> str:
        return (
            f"Expm(dims={self.dims!r}, dtype={self.dtype!r}, sparse={self.is_sparse()})"
        )
