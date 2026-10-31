from enum import StrEnum

import numpy as np
import scipy as sp


class Qtype(StrEnum):
    ket = "ket"
    matrix = "matrix"


class Qobj:
    def __init__(
        self,
        data: list | np.ndarray | sp.sparse.spmatrix,
        dims=None,
        dtype=np.complex128,
        tol=1e-17,
    ):
        self.dtype = dtype
        self.tol = tol

        if isinstance(data, list):
            _data = np.array(data)
        elif isinstance(data, sp.sparse.spmatrix):
            _data = data.tocsr()  # type: ignore
            _data.data[np.abs(_data.data) < tol] = 0
            _data.eliminate_zeros()
        else:
            _data = data

        if _data.ndim == 1:
            _data = _data[:, np.newaxis]
        elif _data.ndim != 2:
            raise ValueError("Data must be a 1D or 2D array")

        self._data = _data.astype(dtype)

        if dims is not None:
            assert np.prod(_data.shape) == np.prod(dims)
            self._dims = dims
        else:
            self._dims = ([_data.shape[0]], [_data.shape[1]])

    @property
    def data(self) -> np.ndarray | sp.sparse.spmatrix:
        return self._data

    @property
    def dims(self) -> tuple[list[int], list[int]]:
        return self._dims

    @property
    def qtype(self) -> Qtype:
        if 1 in self.data.shape:
            return Qtype.ket
        else:
            return Qtype.matrix

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.data, sp.sparse.spmatrix)

    def numpy(self) -> "Qobj":
        if self.is_sparse:
            data = self.data.toarray().copy()  # type: ignore
        else:
            data = self.data.copy()
        return Qobj(data, dims=self.dims)

    def unit(self) -> "Qobj":
        """Normalize the state vector"""
        if self.qtype == Qtype.ket and self.is_sparse:
            normalize = np.sqrt(
                (self.data.conjugate().multiply(self.data)).sum(),  # type:ignore
            )
        elif self.qtype == Qtype.ket:
            normalize = np.linalg.norm(self.data)
        elif self.qtype == Qtype.matrix and self.is_sparse:
            normalize = self.data.trace()
        elif self.qtype == Qtype.matrix:
            normalize = np.trace(self.data)
        else:
            raise NotImplementedError(
                "Normalization is only implemented for ket states and density matrices."
            )
        if normalize == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Qobj(self.data / normalize, dims=self.dims)

    def gpu(self):
        """Move the state to GPU memory"""
        try:
            import cupy  # type: ignore
        except ImportError:
            raise ImportError("pip install optiqs[gpu]")
        data = cupy.asarray(self.data)
        raise NotImplementedError(type(data))

    def copy(self) -> "Qobj":
        """Create a copy of the quantum state"""
        return Qobj(self.data.copy(), dims=self.dims)

    def dag(self) -> "Qobj":
        """Hermitian conjugate of the state"""
        return Qobj(self.data.conj().T, dims=(self.dims[1], self.dims[0]))

    def dm(self) -> "Qobj":
        if self.qtype == Qtype.ket and self.is_sparse:
            return Qobj(
                self.data @ self.data.getH(),  # type:ignore
                dims=(self.dims[0], self.dims[0]),
            )
        elif self.qtype == Qtype.ket:
            return Qobj(
                np.outer(self.data, self.data.conj()), dims=(self.dims[0], self.dims[0])
            )
        elif self.qtype == Qtype.matrix:
            return Qobj(self.data, dims=self.dims)
        raise ValueError("Invalid quantum state")

    def diag(self) -> np.ndarray:
        """Extract the diagonal elements of a density matrix as a state vector"""
        if self.qtype == Qtype.matrix and self.is_sparse:
            return self.data.diagonal()
        elif self.qtype == Qtype.matrix:
            return np.diag(self.data)
        raise ValueError("diag() is only applicable to density matrices")

    def norm(self) -> float:
        """Compute the norm of the state vector"""
        if self.is_sparse:
            return sp.sparse.linalg.norm(self.data).item()
        else:
            return np.linalg.norm(self.data).item()

    def tr(self) -> complex:
        """Compute the trace of the density matrix"""
        if self.qtype == Qtype.matrix and self.is_sparse:
            return self.data.trace().item()
        elif self.qtype == Qtype.matrix:
            return np.trace(self.data).item()
        raise ValueError("tr() is only applicable to density matrices")

    def ptrace(self, p: int) -> "Qobj":
        """Partial trace over specified subsystems"""
        if self.qtype != Qtype.matrix:
            raise ValueError("Partial trace is only applicable to density matrices")

        n_cur = len(self.dims[0])

        if self.is_sparse:
            import sparse

            data = sparse.COO.from_scipy_sparse(self.data)
            data = data.reshape(self.dims[0] + self.dims[1])
            for c in sorted([i for i in range(n_cur) if i != p], reverse=True):
                data = sparse.diagonal(data, axis1=c, axis2=c + n_cur).sum(axis=-1)
                n_cur -= 1
            data = data.reshape((self.dims[0][p], self.dims[1][p]))
            data = sp.sparse.csr_matrix(data.tocsr())  # type:ignore
        else:
            data = self.data.reshape(self.dims[0] + self.dims[1])
            for c in sorted([i for i in range(n_cur) if i != p], reverse=True):
                data = np.trace(data, axis1=c, axis2=c + n_cur)
                n_cur -= 1
        return Qobj(data, dims=([self.dims[0][p]], [self.dims[1][p]]))

    def __add__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims == other.dims, "Dimensions must match for addition"
            return Qobj(self.data + other.data, dims=self.dims)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return Qobj(self.data + other, dims=self.dims)
        raise NotImplementedError()

    def __radd__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims == other.dims, "Dimensions must match for addition"
            return Qobj(other.data + self.data, dims=self.dims)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return Qobj(other + self.data, dims=self.dims)
        raise NotImplementedError()

    def __sub__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims == other.dims, "Dimensions must match for subtracition"
            return Qobj(self.data - other.data, dims=self.dims)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return Qobj(self.data - other, dims=self.dims)
        raise NotImplementedError()

    def __rsub__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims == other.dims, "Dimensions must match for subtracition"
            return Qobj(other.data - self.data, dims=self.dims)
        elif isinstance(other, (int, float, complex, np.ndarray)):
            return Qobj(other - self.data, dims=self.dims)
        raise NotImplementedError()

    def __mul__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims[1] == other.dims[0]
            return Qobj(self.data @ other.data, dims=(self.dims[0], other.dims[1]))
        elif isinstance(other, (int, float, complex)):
            return Qobj(self.data * other, dims=self.dims)
        raise NotImplementedError()

    def __rmul__(self, other) -> "Qobj":
        if isinstance(other, Qobj):
            assert self.dims[1] == other.dims[0]
            return Qobj(other.data @ self.data, dims=(self.dims[0], other.dims[1]))
        elif isinstance(other, (int, float, complex)):
            return Qobj(other * self.data, dims=self.dims)
        raise NotImplementedError()

    def __truediv__(self, other) -> "Qobj":
        if isinstance(other, (int, float, complex)):
            return Qobj(self.data / other, dims=self.dims)
        raise NotImplementedError()

    def __pow__(self, power) -> "Qobj":
        if self.qtype == Qtype.matrix and self.is_sparse:
            return Qobj(sp.sparse.linalg.matrix_power(self.data, power), dims=self.dims)
        elif self.qtype == Qtype.matrix:
            return Qobj(np.linalg.matrix_power(self.data, power), dims=self.dims)
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Qobj):
            return False
        return np.array_equal(self.data, other.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self) -> str:
        _data = np.real_if_close(self.numpy().data, tol=self.tol)
        prefix = "qstate("
        body = np.array2string(_data, prefix=prefix)
        return f"{prefix}{body})"
