from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from numbers import Integral, Number
from os import PathLike
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy as sp
from numpy.typing import DTypeLike

from optiqs.utils import _to_numpy_dtype, get_device

if TYPE_CHECKING:
    import torch


class Qtype(StrEnum):
    ket = "ket"
    bra = "bra"
    oper = "oper"


class Qobj:
    def __init__(
        self,
        data: list[float] | np.ndarray | sp.sparse.sparray | torch.Tensor,
        dims: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
        dtype: DTypeLike | torch.dtype | None = None,
        qtype: Qtype | None = None,
        tol: float = 1e-17,
    ) -> None:
        if not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be a finite non-negative number")

        _data: (
            np.ndarray
            | sp.sparse.csr_array[Any, tuple[int] | tuple[int, int]]
            | torch.Tensor
        )
        if isinstance(data, sp.sparse.sparray):
            np_dtype = _to_numpy_dtype(dtype)
            _data = sp.sparse.csr_array(cast(Any, data), dtype=np_dtype, copy=True)
            _data.sum_duplicates()
            # np.abs(np.int64(-2**63)) wraps around to a negative value.
            # No nonzero integer can be smaller than a tolerance <= 1;
            # otherwise, promote signed integers before the magnitude check.
            if _data.dtype.kind != "i":
                _data.data[np.abs(_data.data) < tol] = 0
            elif tol > 1:
                _data.data[np.abs(_data.data.astype(np.float64)) < tol] = 0
            _data.eliminate_zeros()
        elif isinstance(data, (np.ndarray, list)):
            np_dtype = _to_numpy_dtype(dtype)
            _data = np.array(data, dtype=np_dtype)
        else:
            try:
                import torch
            except ImportError as exc:
                raise TypeError(f"Unsupported data type: {type(data)}") from exc

            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Unsupported data type: {type(data)}")

            if data.layout != torch.strided:
                raise TypeError("PyTorch sparse tensors are not supported")

            if dtype is not None and not isinstance(dtype, torch.dtype):
                raise TypeError("dtype must be torch.dtype when data is a torch.Tensor")

            _data = data.clone() if dtype is None else data.to(dtype=dtype).clone()

        if _data.ndim == 1:
            if isinstance(_data, sp.sparse.csr_array):
                reshaped = cast(Any, _data).reshape((-1, 1))
                _data = sp.sparse.csr_array(reshaped)
            else:
                _data = _data.reshape((-1, 1))
        elif _data.ndim != 2:
            raise ValueError("Data must be a 1D or 2D array")

        if isinstance(_data, sp.sparse.csr_array):
            _data = cast("sp.sparse.csr_array[Any, tuple[int, int]]", _data)

        if dims is None:
            dims = ((_data.shape[0],), (_data.shape[1],))
        if any(
            not isinstance(d, int) or isinstance(d, bool) or d <= 0
            for side in dims
            for d in side
        ):
            raise ValueError("dims must contain positive integers")
        if math.prod(dims[0]) != _data.shape[0] or math.prod(dims[1]) != _data.shape[1]:
            raise ValueError(f"dims {dims} are incompatible with shape {_data.shape}")
        if not dims[0] or not dims[1]:
            raise ValueError("dims must contain at least one subsystem on each side")

        if qtype is not None and not isinstance(qtype, Qtype):
            raise TypeError("qtype must be a Qtype or None")
        if qtype is None:
            if _data.shape[1] == 1:
                qtype = Qtype.ket
            elif _data.shape[0] == 1:
                qtype = Qtype.bra
            else:
                qtype = Qtype.oper
        if qtype == Qtype.ket and _data.shape[1] != 1:
            raise ValueError("A ket must have exactly one column")
        if qtype == Qtype.bra and _data.shape[0] != 1:
            raise ValueError("A bra must have exactly one row")

        self._data: np.ndarray | sp.sparse.csr_array | torch.Tensor = _data
        self._qtype = qtype
        self._dims = dims
        self._tol = tol

    @property
    def dims(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._dims

    @property
    def dtype(self) -> np.dtype[Any] | torch.dtype:
        return self._data.dtype

    @property
    def qtype(self) -> Qtype:
        return self._qtype

    def is_sparse(self) -> bool:
        return isinstance(self._data, sp.sparse.csr_array)

    def is_vector(self) -> bool:
        return self.qtype in (Qtype.ket, Qtype.bra)

    def to_numpy(self) -> np.ndarray:
        """Return an independent NumPy array."""
        data = self._data

        if isinstance(data, np.ndarray):
            return data.copy()

        if isinstance(data, sp.sparse.csr_array):
            return data.toarray()

        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")

        return data.detach().cpu().resolve_conj().resolve_neg().numpy().copy()

    def save(self, filename: str | PathLike[str]) -> None:
        """Save this quantum object; see :func:`optiqs.save_state`."""
        from optiqs.core.io import save_state

        save_state(self, filename)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        """Compare Qobj values with an absolute tolerance, ignoring backend/dtype.

        Subsystem dimensions and quantum type must match.  The larger of the
        two tolerances is used; relative tolerance is deliberately zero.
        """
        if not isinstance(other, Qobj):
            return NotImplemented
        if self.dims != other.dims or self.qtype != other.qtype:
            return False

        tol = max(self._tol, other._tol)
        left, right = self._data, other._data

        if isinstance(left, sp.sparse.csr_array) and isinstance(
            right, sp.sparse.csr_array
        ):
            # Comparing only stored differences avoids converting large sparse
            # matrices into dense arrays.  Exact comparisons also avoid integer
            # subtraction overflow and correctly compare matching infinities.
            if tol == 0 or (
                tol < 1 and left.dtype.kind in "biu" and right.dtype.kind in "biu"
            ):
                # SciPy's equality operator returns a sparse array at runtime,
                # but its type stubs may also declare ``bool``. Narrow the
                # result explicitly before accessing ``nnz``.
                unequal = left != right
                if isinstance(unequal, sp.sparse.csr_array):
                    return unequal.nnz == 0
                return not bool(unequal)
            if np.isfinite(left.data).all() and np.isfinite(right.data).all():
                # Promote integers before subtracting to avoid wraparound.
                if left.dtype.kind in "biu" or right.dtype.kind in "biu":
                    left = left.astype(np.result_type(left.dtype, np.float64))
                    right = right.astype(np.result_type(right.dtype, np.float64))
                difference = left - right
                return bool(np.all(np.abs(difference.data) <= tol))
            # Infrequent non-finite case: inf - inf is NaN, though equal infs
            # should compare equal.  NumPy handles these values consistently.
            return bool(
                np.allclose(left.toarray(), right.toarray(), atol=tol, rtol=0.0)
            )

        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if tol == 0 or (
                tol < 1 and left.dtype.kind in "biu" and right.dtype.kind in "biu"
            ):
                return bool(np.array_equal(left, right))
            return bool(np.allclose(left, right, atol=tol, rtol=0.0))

        # Cross-backend comparisons are allowed, unlike arithmetic operations.
        # Convert only in this case (or when the two tensors are on different
        # devices), so normal PyTorch equality remains device-local.
        if isinstance(left, (np.ndarray, sp.sparse.csr_array)) or isinstance(
            right, (np.ndarray, sp.sparse.csr_array)
        ):
            left_numpy, right_numpy = self.to_numpy(), other.to_numpy()
            if tol == 0 or (
                tol < 1
                and left_numpy.dtype.kind in "biu"
                and right_numpy.dtype.kind in "biu"
            ):
                return bool(np.array_equal(left_numpy, right_numpy))
            return bool(np.allclose(left_numpy, right_numpy, atol=tol, rtol=0.0))

        import torch

        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("Unsupported Qobj data type")
        if left.device != right.device:
            left_numpy, right_numpy = self.to_numpy(), other.to_numpy()
            if tol == 0 or (
                tol < 1
                and left_numpy.dtype.kind in "biu"
                and right_numpy.dtype.kind in "biu"
            ):
                return bool(np.array_equal(left_numpy, right_numpy))
            return bool(np.allclose(left_numpy, right_numpy, atol=tol, rtol=0.0))

        if tol == 0 or (
            tol < 1
            and not (left.is_floating_point() or left.is_complex())
            and not (right.is_floating_point() or right.is_complex())
        ):
            return bool(torch.equal(left, right))

        promoted_dtype = torch.promote_types(left.dtype, right.dtype)
        return bool(
            torch.allclose(
                left.to(dtype=promoted_dtype),
                right.to(dtype=promoted_dtype),
                atol=tol,
                rtol=0.0,
            )
        )

    def _normalize_index(
        self, key: int | slice | tuple[int | slice, int | slice]
    ) -> tuple[int | slice, int | slice]:
        """Normalize matrix indices, including one-index ket/bra access."""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("Qobj requires exactly two matrix indices")
            row, col = key
        elif self.qtype == Qtype.ket:
            row, col = key, 0
        elif self.qtype == Qtype.bra:
            row, col = 0, key
        else:
            raise IndexError("Use obj[row, col] to index an operator")

        def axis_index(index: object) -> int | slice:
            if isinstance(index, (bool, np.bool_)):
                raise TypeError("Qobj indices must be integers or slices")
            if isinstance(index, slice):
                return index
            if isinstance(index, (int, np.integer)):
                return int(index)
            raise TypeError("Qobj indices must be integers or slices")

        return axis_index(row), axis_index(col)

    def __getitem__(self, key: int | slice | tuple[int | slice, int | slice]) -> Any:
        """Return a matrix element or an independent, backend-native slice.

        Two integer indices return a NumPy/SciPy scalar or a zero-dimensional
        PyTorch tensor (to preserve autograd). Slices return copies rather
        than views, so modifications must go through ``obj[...] = value``.
        Kets and bras also support one-dimensional indexing.
        """
        index = self._normalize_index(key)
        data = self._data
        if isinstance(data, np.ndarray):
            selected = data[index]
            return selected.copy() if isinstance(selected, np.ndarray) else selected
        if isinstance(data, sp.sparse.csr_array):
            selected = cast(Any, data)[index]

            if isinstance(selected, sp.sparse.csr_array):
                return selected.copy()

            return selected

        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")
        return data[index].clone()

    def __setitem__(
        self, key: int | slice | tuple[int | slice, int | slice], value: Any
    ) -> None:
        """Update selected entries in place without changing dims or qtype.

        Sparse updates rebuild a CSR array via LIL to support inserting new
        nonzero entries without SparseEfficiencyWarning, then discard entries
        with magnitude below ``tol`` as in the constructor.
        """
        index = self._normalize_index(key)
        if isinstance(value, Qobj):
            raise TypeError("Assign a scalar or array-like value, not a Qobj")

        data = self._data
        if isinstance(data, np.ndarray):
            data[index] = value
            return

        if isinstance(data, sp.sparse.csr_array):
            mutable = sp.sparse.lil_array(data)
            cast(Any, mutable)[index] = value

            updated = sp.sparse.csr_array(mutable)
            updated.sum_duplicates()
            updated.data[np.abs(updated.data) < self._tol] = 0
            updated.eliminate_zeros()
            self._data = updated
            return

        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")
        # Tensor indexing supports Tensor/scalar assignment, but not Python
        # lists or NumPy arrays directly. Convert array-like RHS on the same
        # device without detaching existing Tensor inputs.
        if isinstance(value, (list, tuple, np.ndarray)):
            value = torch.as_tensor(value, device=data.device)
        data[index] = value

    @staticmethod
    def _binary_data(left: Qobj, right: Qobj) -> tuple[Any, Any]:
        """Align dense/sparse backends and promote PyTorch dtypes."""
        left_data, right_data = left._data, right._data
        left_torch = not isinstance(left_data, (np.ndarray, sp.sparse.csr_array))
        right_torch = not isinstance(right_data, (np.ndarray, sp.sparse.csr_array))

        if left_torch != right_torch:
            raise TypeError(
                "Cannot mix PyTorch and NumPy/SciPy Qobj data. "
                "Convert both objects to the same backend first."
            )

        if left_torch:
            import torch

            # Boolean flags alone do not narrow a union for Pyright/mypy.
            if not isinstance(left_data, torch.Tensor) or not isinstance(
                right_data, torch.Tensor
            ):
                raise TypeError("Expected PyTorch tensors for both operands")

            if left_data.device != right_data.device:
                raise ValueError("PyTorch Qobj operands must be on the same device")
            dtype = torch.promote_types(left_data.dtype, right_data.dtype)
            return left_data.to(dtype=dtype), right_data.to(dtype=dtype)

        # A sparse + dense expression is made dense explicitly. Two sparse
        # operands stay sparse; two NumPy arrays stay NumPy arrays.
        if isinstance(left_data, sp.sparse.csr_array) and isinstance(
            right_data, np.ndarray
        ):
            left_data = left_data.toarray()
        elif isinstance(left_data, np.ndarray) and isinstance(
            right_data, sp.sparse.csr_array
        ):
            right_data = right_data.toarray()
        return left_data, right_data

    def _is_scalar(self, other: object) -> bool:
        """Accept Python/NumPy numbers and 0-d tensors for torch-backed Qobj."""
        if isinstance(other, Number) and not isinstance(other, (bool, np.bool_)):
            return True
        if isinstance(self._data, (np.ndarray, sp.sparse.csr_array)):
            return False
        import torch

        return isinstance(other, torch.Tensor) and other.ndim == 0

    def _shift_identity(self, scalar: Any) -> Qobj:
        """Return A + scalar*I, defined for square operators only."""
        if self.qtype != Qtype.oper or self.dims[0] != self.dims[1]:
            raise TypeError(
                "Adding a nonzero scalar requires a square Qtype.oper "
                "with matching input/output dims"
            )

        data = self._data
        size = data.shape[0]

        if isinstance(data, sp.sparse.csr_array):
            # eye_array returns a sparse array (not the legacy csr_matrix).
            sparse_identity = sp.sparse.eye_array(size, format="csr", dtype=data.dtype)
            return Qobj(
                data + scalar * sparse_identity,
                dims=self.dims,
                qtype=self.qtype,
                tol=self._tol,
            )

        if isinstance(data, np.ndarray):
            numpy_identity = np.eye(size, dtype=data.dtype)
            return Qobj(
                data + scalar * numpy_identity,
                dims=self.dims,
                qtype=self.qtype,
                tol=self._tol,
            )

        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported data type: {type(data)}")
        torch_identity = torch.eye(size, dtype=data.dtype, device=data.device)
        return Qobj(
            data + scalar * torch_identity,
            dims=self.dims,
            qtype=self.qtype,
            tol=self._tol,
        )

    def __add__(self, other: object) -> Qobj | NotImplementedType:
        if isinstance(other, Qobj):
            if self.dims != other.dims or self.qtype != other.qtype:
                raise ValueError("Qobj addition requires identical dims and qtype")
            left, right = self._binary_data(self, other)
            return Qobj(
                left + right,
                dims=self.dims,
                qtype=self.qtype,
                tol=max(self._tol, other._tol),
            )
        if self._is_scalar(other):
            scalar = cast(Any, other)
            return (
                self.copy()
                if isinstance(scalar, Number) and scalar == 0
                else self._shift_identity(scalar)
            )
        return NotImplemented

    def __radd__(self, other: object) -> Qobj | NotImplementedType:
        return self.__add__(other)

    def __sub__(self, other: object) -> Qobj | NotImplementedType:
        if isinstance(other, Qobj):
            if self.dims != other.dims or self.qtype != other.qtype:
                raise ValueError("Qobj subtraction requires identical dims and qtype")
            left, right = self._binary_data(self, other)
            return Qobj(
                left - right,
                dims=self.dims,
                qtype=self.qtype,
                tol=max(self._tol, other._tol),
            )
        if self._is_scalar(other):
            # Avoid narrowing the operand to "Any | Number": numbers.Number
            # does not declare __neg__ in its static type definition.
            if isinstance(other, Number) and other == 0:
                return self.copy()
            return self._shift_identity(-cast(Any, other))
        return NotImplemented

    def __rsub__(self, other: object) -> Qobj | NotImplementedType:
        if self._is_scalar(other):
            scalar = cast(Any, other)
            return (
                -self
                if isinstance(scalar, Number) and scalar == 0
                else (-self)._shift_identity(scalar)
            )
        return NotImplemented

    def __neg__(self) -> Qobj:
        return Qobj(-self._data, dims=self.dims, qtype=self.qtype, tol=self._tol)

    def __matmul__(self, other: object) -> Qobj | NotImplementedType:
        if not isinstance(other, Qobj):
            return NotImplemented
        if self.dims[1] != other.dims[0]:
            raise ValueError(
                "Qobj matrix multiplication requires "
                f"left.dims[1] == right.dims[0], got "
                f"{self.dims[1]} and {other.dims[0]}"
            )

        left, right = self._binary_data(self, other)
        result = left @ right
        dims = (self.dims[0], other.dims[1])

        # Preserve explicitly declared operators, even for 1x1 matrices.
        # An outer product and an inner product are treated as operators;
        # the 1x1 inner-product value is available via .to_numpy().item().
        if (self.qtype, other.qtype) in (
            (Qtype.oper, Qtype.oper),
            (Qtype.ket, Qtype.bra),
            (Qtype.bra, Qtype.ket),
        ):
            qtype = Qtype.oper
        elif result.shape[0] == 1 and self.qtype == Qtype.bra:
            qtype = Qtype.bra
        elif result.shape[1] == 1 and other.qtype == Qtype.ket:
            qtype = Qtype.ket
        elif result.shape[0] == 1 and other.qtype == Qtype.bra:
            qtype = Qtype.bra
        elif result.shape[1] == 1 and self.qtype == Qtype.ket or result.shape[1] == 1:
            qtype = Qtype.ket
        elif result.shape[0] == 1:
            qtype = Qtype.bra
        else:
            qtype = Qtype.oper

        return Qobj(result, dims=dims, qtype=qtype, tol=max(self._tol, other._tol))

    def __mul__(self, other: object) -> Qobj | NotImplementedType:
        if isinstance(other, Qobj):
            return self.__matmul__(other)
        if self._is_scalar(other):
            return Qobj(
                self._data * cast(Any, other),
                dims=self.dims,
                qtype=self.qtype,
                tol=self._tol,
            )
        return NotImplemented

    def __rmul__(self, other: object) -> Qobj | NotImplementedType:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> Qobj | NotImplementedType:
        if not self._is_scalar(other):
            return NotImplemented
        scalar = cast(Any, other)
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Qobj by zero")
        return Qobj(
            self._data / scalar,
            dims=self.dims,
            qtype=self.qtype,
            tol=self._tol,
        )

    def __pow__(self, exponent: object, modulo: object = None) -> Qobj:
        """Return the integer matrix power of a square operator.

        ``A ** 0`` is the identity; negative powers use the matrix inverse.
        The result keeps the original backend, subsystem dims, and tolerance.
        For sparse matrices, exponentiation by squaring avoids densification.
        """
        if modulo is not None:
            raise TypeError("Qobj matrix powers do not support a modulo argument")
        if isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, Integral):
            raise TypeError("Qobj matrix powers require an integer exponent")
        if self.qtype != Qtype.oper:
            raise TypeError("Qobj matrix powers require a Qtype.oper")
        if self.dims[0] != self.dims[1]:
            raise ValueError(
                "Qobj matrix powers require matching input/output subsystem dims"
            )

        power = int(exponent)
        data = self._data
        result: np.ndarray | sp.sparse.csr_array | torch.Tensor
        if isinstance(data, np.ndarray):
            result = np.linalg.matrix_power(data, power)
        elif isinstance(data, sp.sparse.csr_array):
            if power < 0:
                if data.shape[0] == 1:
                    # SciPy's sparse inverse returns a 1-D ndarray for 1x1
                    # inputs, and returns NaN rather than raising if singular.
                    value = data[0, 0]
                    if value == 0:
                        raise np.linalg.LinAlgError("Singular matrix")
                    base = sp.sparse.csr_array([[1 / value]])
                else:
                    try:
                        inverse = sp.sparse.linalg.inv(data.tocsc())
                    except RuntimeError as exc:
                        if "singular" in str(exc).lower():
                            raise np.linalg.LinAlgError("Singular matrix") from exc
                        raise
                    base = sp.sparse.csr_array(inverse)
            else:
                base = data
            result = sp.sparse.eye_array(data.shape[0], format="csr", dtype=base.dtype)
            remaining = abs(power)
            while remaining:
                if remaining & 1:
                    result = sp.sparse.csr_array(result @ base)
                remaining >>= 1
                if remaining:
                    base = sp.sparse.csr_array(base @ base)
        else:
            import torch

            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Unsupported data type: {type(data)}")
            if power < 0 and not (data.is_floating_point() or data.is_complex()):
                data = data.to(dtype=torch.get_default_dtype())
            result = torch.linalg.matrix_power(data, power)

        return Qobj(result, dims=self.dims, qtype=self.qtype, tol=self._tol)

    def unit(self) -> Qobj:
        """
        Return a trace-one density operator from a ket or bra.

        The input vector is normalized in a copy; this object is not changed.
        """
        norm: float | complex

        if self.qtype == Qtype.oper and self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Trace normalization requires a square operator")

        if isinstance(self._data, sp.sparse.csr_array):
            raw_norm = (
                np.linalg.norm(self._data.data)
                if self.is_vector()
                else self._data.diagonal().sum()
            )
            norm = raw_norm.item() if isinstance(raw_norm, np.generic) else raw_norm

        elif isinstance(self._data, np.ndarray):
            raw_norm = np.linalg.norm(self._data) if self.is_vector() else self.tr()
            norm = raw_norm.item() if isinstance(raw_norm, np.generic) else raw_norm

        else:
            import torch

            if not (self._data.is_floating_point() or self._data.is_complex()):
                self._data = self._data.to(dtype=torch.get_default_dtype())

            if self.is_vector():
                norm_tensor = torch.linalg.vector_norm(self._data)
            else:
                norm_tensor = torch.trace(self._data)

            if not torch.isfinite(norm_tensor).item() or norm_tensor.abs().item() == 0:
                raise ValueError("Cannot normalize an object with zero or invalid norm")

            self._data = self._data / norm_tensor
            return self

        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Cannot normalize an object with zero or invalid norm")

        if isinstance(self._data, sp.sparse.csr_array):
            self._data = sp.sparse.csr_array(self._data / norm)
        else:
            self._data = self._data / norm

        return self

    def gpu(self, *, allow_dense: bool = False) -> Qobj:
        """Move the state to GPU memory"""
        if self.is_sparse() and not allow_dense:
            raise ValueError("Sparse GPU conversion requires allow_dense=True")

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Install GPU support with: pip install optiqs[gpu]"
            ) from exc

        device = get_device()
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
        elif device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available")
        else:
            raise RuntimeError(f"Unsupported GPU device: {device}")

        if isinstance(self._data, sp.sparse.csr_array):
            array = self._data.toarray()
            tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
        elif isinstance(self._data, np.ndarray):
            tensor = torch.from_numpy(np.array(self._data, copy=True, order="C"))
        else:
            tensor = self._data

        if device == "mps" and tensor.dtype in (torch.float64, torch.complex128):
            raise TypeError(
                "MPS does not support float64 or complex128. "
                "Specify torch.float32 or torch.complex64 explicitly."
            )

        self._data = tensor.to(device)
        return self

    def cpu(self) -> Qobj:
        """Move data to CPU as a NumPy array."""
        if isinstance(self._data, (np.ndarray, sp.sparse.csr_array)):
            return self
        self._data = (
            self._data.detach().cpu().resolve_conj().resolve_neg().numpy().copy()
        )
        return self

    def copy(self) -> Qobj:
        """Create a copy of the quantum state."""
        if isinstance(self._data, (np.ndarray, sp.sparse.csr_array)):
            return Qobj(
                self._data.copy(), dims=self.dims, tol=self._tol, qtype=self.qtype
            )

        return Qobj(self._data.clone(), dims=self.dims, tol=self._tol, qtype=self.qtype)

    def dag(self) -> Qobj:
        """Hermitian conjugate of the state"""
        adjoint_type = {
            Qtype.ket: Qtype.bra,
            Qtype.bra: Qtype.ket,
            Qtype.oper: Qtype.oper,
        }[self.qtype]
        return Qobj(
            self._data.conj().T,
            dims=(self.dims[1], self.dims[0]),
            tol=self._tol,
            qtype=adjoint_type,
        )

    def dm(self) -> Qobj:
        """Convert a pure quantum state into a density operator."""
        if not self.is_vector():
            raise TypeError("Density operator conversion requires a ket or bra")
        state = self.copy()
        if state.qtype == Qtype.bra:
            state = state.dag()
        state.unit()
        rho: np.ndarray | sp.sparse.csr_array | torch.Tensor
        if isinstance(state._data, sp.sparse.csr_array):
            rho = sp.sparse.csr_array(state._data @ state._data.conj().T)
        else:
            rho = state._data @ state._data.conj().T
        return Qobj(
            rho, dims=(state.dims[0], state.dims[0]), tol=self._tol, qtype=Qtype.oper
        )

    def diag(self) -> np.ndarray:
        """Return the diagonal elements as a 1D NumPy array."""
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Diagonal extraction requires a square matrix")

        if isinstance(self._data, sp.sparse.csr_array):
            return np.asarray(self._data.diagonal()).copy()

        if isinstance(self._data, np.ndarray):
            return np.diagonal(self._data).copy()

        return (
            self._data.diagonal()
            .detach()
            .cpu()
            .resolve_conj()
            .resolve_neg()
            .numpy()
            .copy()
        )

    def norm(self) -> float:
        """Return the Euclidean or Frobenius norm of the quantum object."""
        if isinstance(self._data, sp.sparse.csr_array):
            return float(np.linalg.norm(self._data.data))

        if isinstance(self._data, np.ndarray):
            return float(np.linalg.norm(self._data))

        import torch

        data = self._data
        if not (data.is_floating_point() or data.is_complex()):
            data = data.to(dtype=torch.get_default_dtype())

        return float(torch.linalg.vector_norm(data).item())

    def tr(self) -> float | complex:
        """Return the trace of the quantum operator."""
        if self._data.shape[0] != self._data.shape[1]:
            raise ValueError("Trace is only defined for square matrices")

        if isinstance(self._data, sp.sparse.csr_array):
            trace = self._data.diagonal().sum()
        elif isinstance(self._data, np.ndarray):
            trace = np.trace(self._data)
        else:
            import torch

            trace = torch.trace(self._data)

        value = trace.item()
        return complex(value) if isinstance(value, complex) else float(value)

    def ptrace(self, keep: int | Sequence[int]) -> Qobj:
        """
        Return the partial trace over all subsystems not in keep.

        The retained subsystems are returned in ascending index order.
        """
        keep_indices: tuple[int, ...]

        if isinstance(keep, Sequence):
            keep_indices = tuple(keep)
        else:
            if isinstance(keep, bool):
                raise TypeError("keep must not be a boolean")
            keep_indices = (keep,)

        if (
            not keep_indices
            or any(not isinstance(i, int) or isinstance(i, bool) for i in keep_indices)
            or len(set(keep_indices)) != len(keep_indices)
        ):
            raise ValueError("keep must contain distinct subsystem indices")

        if self.qtype == Qtype.ket:
            subsystems = self.dims[0]
        elif self.qtype == Qtype.bra:
            subsystems = self.dims[1]
        else:
            if self.dims[0] != self.dims[1]:
                raise ValueError("Operator must have matching input/output dims")
            subsystems = self.dims[0]

        if any(i < 0 or i >= len(subsystems) for i in keep_indices):
            raise IndexError("Subsystem index out of range")

        kept = tuple(sorted(keep_indices))
        traced = tuple(i for i in range(len(subsystems)) if i not in kept)

        kept_dims = tuple(subsystems[i] for i in kept)
        output_dims = (kept_dims, kept_dims)
        out_size = math.prod(kept_dims)

        data = self._data
        result: np.ndarray | sp.sparse.csr_array | torch.Tensor

        if self.is_vector():
            if isinstance(data, sp.sparse.csr_array):
                entries = data.tocoo()
                if self.qtype == Qtype.ket:
                    indices = entries.row
                    amplitudes = entries.data
                else:
                    indices = entries.col
                    amplitudes = entries.data.conj()

                multi = np.unravel_index(indices, subsystems)

                row = np.ravel_multi_index(tuple(multi[i] for i in kept), kept_dims)

                if traced:
                    traced_dims = tuple(subsystems[i] for i in traced)
                    col = np.ravel_multi_index(
                        tuple(multi[i] for i in traced), traced_dims
                    )
                else:
                    col = np.zeros_like(row)

                sparse_matrix = sp.sparse.csr_array(
                    (amplitudes, (row, col)),
                    shape=(out_size, math.prod(subsystems) // out_size),
                )
                result = sp.sparse.csr_array(sparse_matrix @ sparse_matrix.conj().T)

            elif isinstance(data, np.ndarray):
                np_ket = data if self.qtype == Qtype.ket else data.conj().T

                np_matrix = (
                    np_ket.reshape(subsystems)
                    .transpose(kept + traced)
                    .reshape(out_size, -1)
                )
                result = np_matrix @ np_matrix.conj().T

            else:
                import torch

                if not isinstance(data, torch.Tensor):
                    raise TypeError(f"Unsupported data type: {type(data)}")

                torch_ket = data if self.qtype == Qtype.ket else data.conj().T

                torch_matrix = (
                    torch_ket.reshape(*subsystems)
                    .permute(*(kept + traced))
                    .reshape(out_size, -1)
                )
                result = torch_matrix @ torch_matrix.conj().T

        elif isinstance(data, sp.sparse.csr_array):
            entries = data.tocoo()
            rows = np.unravel_index(entries.row, subsystems)
            cols = np.unravel_index(entries.col, subsystems)

            mask = np.ones(entries.nnz, dtype=bool)
            for i in traced:
                mask &= rows[i] == cols[i]

            row = np.ravel_multi_index(tuple(rows[i][mask] for i in kept), kept_dims)
            col = np.ravel_multi_index(tuple(cols[i][mask] for i in kept), kept_dims)

            result = sp.sparse.csr_array(
                (entries.data[mask], (row, col)),
                shape=(out_size, out_size),
            )

        elif isinstance(data, np.ndarray):
            tensor = data.reshape(subsystems + subsystems)
            n = len(subsystems)

            for i in reversed(traced):
                tensor = np.trace(tensor, axis1=i, axis2=i + n)
                n -= 1

            result = tensor.reshape(out_size, out_size)

        else:
            import torch

            if not isinstance(data, torch.Tensor):
                raise TypeError(f"Unsupported data type: {type(data)}")

            torch_tensor = data.reshape(*(subsystems + subsystems))
            n = len(subsystems)

            for i in reversed(traced):
                torch_tensor = torch_tensor.diagonal(
                    dim1=i,
                    dim2=i + n,
                ).sum(-1)
                n -= 1

            result = torch_tensor.reshape(out_size, out_size)

        return Qobj(result, dims=output_dims, tol=self._tol, qtype=Qtype.oper)
