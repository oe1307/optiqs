"""Expectation values and variances of quantum observables.

The routines avoid constructing squared observable matrices for variances
and retain the input NumPy/SciPy or PyTorch backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse

from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    from torch import Tensor


def _validate(operator: Qobj, state: Qobj) -> None:
    """Check observable/state dimensions and avoid backend conversions."""
    if not isinstance(operator, Qobj):
        raise TypeError("operator must be a Qobj")
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")
    if operator.qtype != Qtype.oper or operator.dims[0] != operator.dims[1]:
        raise ValueError("operator must be square with matching subsystem dims")
    if state.qtype == Qtype.oper and state.dims[0] != state.dims[1]:
        raise ValueError("density operator must have matching subsystem dims")

    state_dims = state.dims[1] if state.qtype == Qtype.bra else state.dims[0]
    if operator.dims[0] != state_dims:
        raise ValueError(
            f"operator subsystem dims {operator.dims[0]} do not match "
            f"state subsystem dims {state_dims}"
        )

    numpy_operator = isinstance(operator._data, (np.ndarray, sparse.csr_array))
    numpy_state = isinstance(state._data, (np.ndarray, sparse.csr_array))
    if numpy_operator != numpy_state:
        raise TypeError(
            "Cannot mix PyTorch and NumPy/SciPy Qobj data; "
            "convert the operator and state to the same backend"
        )
    if not numpy_operator:
        import torch

        op_data, state_data = operator._data, state._data
        if not isinstance(op_data, torch.Tensor) or not isinstance(
            state_data, torch.Tensor
        ):
            raise TypeError("Unsupported Qobj data type")
        if op_data.device != state_data.device:
            raise ValueError("PyTorch operator and state must use the same device")


def _scalar(value: Any) -> float | complex:
    """Convert a NumPy/SciPy scalar into a Python scalar (not a 1x1 array)."""
    scalar = np.asarray(value).item()
    return complex(scalar) if isinstance(scalar, complex) else float(scalar)


def _numpy_matmul(
    left: np.ndarray | sparse.csr_array,
    right: np.ndarray | sparse.csr_array,
) -> np.ndarray | sparse.csr_array:
    """Multiply compatible arrays without converting a sparse input to dense."""
    product = cast(Any, left) @ cast(Any, right)
    if isinstance(product, np.ndarray):
        return product
    if isinstance(product, sparse.sparray):
        return sparse.csr_array(cast(Any, product))
    raise TypeError(f"Unexpected matrix product type: {type(product)}")


def _numpy_inner(
    vector: np.ndarray | sparse.csr_array,
    transformed: np.ndarray | sparse.csr_array,
) -> float | complex:
    """Evaluate <vector|transformed> with sparse inputs still sparse."""
    if isinstance(vector, sparse.csr_array):
        # Elementwise multiplication touches only stored amplitudes.
        return _scalar(vector.conjugate().multiply(cast(Any, transformed)).sum())
    if isinstance(transformed, sparse.csr_array):
        transformed = (
            transformed.toarray()
        )  # Only a length-N vector, not an NxN matrix.
    return _scalar(np.vdot(vector, transformed))


def _numpy_trace_product(
    left: np.ndarray | sparse.csr_array,
    right: np.ndarray | sparse.csr_array,
) -> float | complex:
    """Compute Tr(left @ right) without forming the matrix product."""
    if isinstance(left, sparse.csr_array):
        return _scalar(left.multiply(cast(Any, right.T)).sum())
    if isinstance(right, sparse.csr_array):
        return _scalar(right.multiply(cast(Any, left.T)).sum())
    return _scalar(np.einsum("ij,ji->", left, right))


def _numpy_moments(
    operator: Qobj, state: Qobj, *, second: bool
) -> tuple[float | complex, float | complex | None]:
    op_data = operator._data
    state_data = state._data
    if not isinstance(op_data, (np.ndarray, sparse.csr_array)) or not isinstance(
        state_data, (np.ndarray, sparse.csr_array)
    ):
        raise TypeError("Expected NumPy/SciPy data")

    if state.qtype == Qtype.oper:
        first = _numpy_trace_product(op_data, state_data)
        if not second:
            return first, None
        # Form A @ rho instead of A @ A; sparse inputs remain sparse.
        transformed = _numpy_matmul(op_data, state_data)
        return first, _numpy_trace_product(op_data, transformed)

    vector: np.ndarray | sparse.csr_array
    if state.qtype == Qtype.ket:
        vector = state_data
    elif isinstance(state_data, sparse.csr_array):
        vector = sparse.csr_array(cast(Any, state_data.conjugate().T))
    else:
        vector = state_data.conjugate().T
    transformed = _numpy_matmul(op_data, vector)
    first = _numpy_inner(vector, transformed)
    if not second:
        return first, None
    # A @ (A @ psi), rather than constructing the potentially dense A**2.
    twice = _numpy_matmul(op_data, transformed)
    return first, _numpy_inner(vector, twice)


def _torch_moments(
    operator: Qobj, state: Qobj, *, second: bool
) -> tuple[Tensor, Tensor | None]:
    import torch

    op_data = operator._data
    state_data = state._data
    if not isinstance(op_data, torch.Tensor) or not isinstance(
        state_data, torch.Tensor
    ):
        raise TypeError("Expected PyTorch tensors")
    dtype = torch.promote_types(op_data.dtype, state_data.dtype)
    if not (dtype.is_floating_point or dtype.is_complex):
        dtype = torch.get_default_dtype()
    matrix = op_data.to(dtype=dtype)
    data = state_data.to(dtype=dtype)

    if state.qtype == Qtype.oper:
        first = torch.sum(matrix * data.transpose(0, 1))
        if not second:
            return first, None
        transformed = matrix @ data
        return first, torch.sum(matrix * transformed.transpose(0, 1))

    vector = data if state.qtype == Qtype.ket else data.conj().transpose(0, 1)
    transformed = matrix @ vector
    first = torch.sum(vector.conj() * transformed)
    if not second:
        return first, None
    twice = matrix @ transformed
    return first, torch.sum(vector.conj() * twice)


def expect(operator: Qobj, state: Qobj) -> float | complex | Tensor:
    """Return the quantum expectation value ``<A>`` of an operator.

    For a ket (or bra), computes ``<psi|A|psi>``; for a density operator,
    computes ``Tr(A rho)``. Inputs must have matching subsystem dimensions,
    and may use NumPy, SciPy CSR, or PyTorch backends. Dense and sparse
    NumPy/SciPy inputs may be mixed; PyTorch must not be mixed with either.

    The state is **not normalized automatically**: callers should supply
    a normalized vector or trace-one density operator when they need a
    physical expectation value. Non-Hermitian operators are supported and
    may produce complex results.

    NumPy and SciPy return a Python ``float`` or ``complex``. PyTorch
    returns a zero-dimensional tensor on the original device with autograd
    intact; no implicit CPU transfer or detaching is performed.

    Parameters
    ----------
    operator : Qobj
        Square observable (``Qtype.oper``).
    state : Qobj
        Ket, bra, or density operator.

    Returns
    -------
    float, complex, or torch.Tensor
        Scalar expectation value.

    Examples
    --------
    >>> expect(number(8), fock(8, 3))
    (3+0j)
    """
    _validate(operator, state)
    result: float | complex | Tensor
    if isinstance(state._data, (np.ndarray, sparse.csr_array)):
        result = _numpy_moments(operator, state, second=False)[0]
    else:
        result = _torch_moments(operator, state, second=False)[0]
    return result


def variance(operator: Qobj, state: Qobj) -> float | complex | Tensor:
    """Return the algebraic quantum variance ``<A^2> - <A>^2``.

    For Hermitian observables and normalized states this is the ordinary
    nonnegative variance (up to floating-point roundoff). For a
    non-Hermitian operator the result can be complex; this is **not** the
    alternative quantity ``<A.dag() @ A> - abs(<A>)**2``.

    No squared observable matrix is built: for pure states the function
    applies ``A`` twice to the state vector. The state is not normalized
    automatically. NumPy/SciPy results are Python scalars; PyTorch results
    remain differentiable zero-dimensional tensors.

    Parameters
    ----------
    operator : Qobj
        Square observable (``Qtype.oper``).
    state : Qobj
        Ket, bra, or density operator, with matching subsystem dimensions.

    Returns
    -------
    float, complex, or torch.Tensor
        Scalar variance.

    Examples
    --------
    >>> variance(number(8), fock(8, 3))
    0j
    """
    _validate(operator, state)
    if isinstance(state._data, (np.ndarray, sparse.csr_array)):
        mean, second = _numpy_moments(operator, state, second=True)
        if second is None:
            raise RuntimeError("The second moment was not computed")
        return second - mean * mean
    torch_mean, torch_second = _torch_moments(operator, state, second=True)
    if torch_second is None:
        raise RuntimeError("The second moment was not computed")
    return torch_second - torch_mean * torch_mean
