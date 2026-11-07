"""Kronecker products of quantum states and operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from optiqs.core.q_obj import Qobj, Qtype

if TYPE_CHECKING:
    import torch


def _kron_data(
    left: np.ndarray | sparse.csr_array | torch.Tensor,
    right: np.ndarray | sparse.csr_array | torch.Tensor,
) -> np.ndarray | sparse.csr_array | torch.Tensor:
    """Dispatch without moving tensors between CPU and accelerator backends."""
    if isinstance(left, (np.ndarray, sparse.csr_array)):
        if not isinstance(right, (np.ndarray, sparse.csr_array)):
            raise TypeError(
                "Cannot mix PyTorch and NumPy/SciPy Qobj data. "
                "Convert all objects to the same backend first."
            )
        if isinstance(left, sparse.csr_array) or isinstance(right, sparse.csr_array):
            # CSR outputs do not allocate an otherwise huge dense product.
            return sparse.csr_array(sparse.kron(left, right, format="csr"), copy=False)
        return np.kron(left, right)

    if isinstance(right, (np.ndarray, sparse.csr_array)):
        raise TypeError(
            "Cannot mix PyTorch and NumPy/SciPy Qobj data. "
            "Convert all objects to the same backend first."
        )

    import torch

    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("Unsupported Qobj data backend")
    if left.device != right.device:
        raise ValueError("PyTorch Qobj operands must be on the same device")

    # torch.kron runs on the original device and retains autograd history.
    promoted_dtype = torch.promote_types(left.dtype, right.dtype)
    # torch.kron can reject non-contiguous views, e.g. Qobj.dag().
    # Materialize contiguous inputs only when needed.
    left_tensor = left.to(dtype=promoted_dtype)
    right_tensor = right.to(dtype=promoted_dtype)
    return torch.kron(left_tensor.contiguous(), right_tensor.contiguous())


def tensor(*objects: Qobj | Sequence[Qobj]) -> Qobj:
    """Compute the tensor (Kronecker) product of one or more quantum objects.

    Usage: ``tensor(a, b, c)`` or ``tensor([a, b, c])``.

    Subsystem dimensions are concatenated in input order (the rightmost
    subsystem index varies fastest). Kets remain kets, bras remain bras, and
    operators remain operators when all operands share the same quantum type.
    Mixed quantum types return an operator.

    If any input is SciPy CSR, the product is kept sparse (including mixed
    NumPy/SciPy inputs), preventing accidental densification of large
    products. All-NumPy inputs produce a dense NumPy array. PyTorch inputs
    remain on their original device and support autograd. NumPy/SciPy and
    PyTorch operands cannot be mixed.

    The returned Qobj is always independent of its inputs. Like any explicit
    Kronecker product, its storage scales with the number of output elements
    (or output nonzeros for sparse inputs).

    Parameters
    ----------
    *objects : Qobj or sequence of Qobj
        One or more factors, or one sequence containing the factors.

    Returns
    -------
    Qobj
        Tensor product with combined ``dims`` and the largest input tolerance.
    """
    # Normalize the two supported calling styles to a definite Qobj list.
    # Validating each item also narrows the type for Pyright and mypy.
    if len(objects) == 1 and isinstance(objects[0], Sequence):
        candidates: Sequence[Qobj | Sequence[Qobj]] = objects[0]
    else:
        candidates = objects

    factors: list[Qobj] = []
    for i, obj in enumerate(candidates):
        if not isinstance(obj, Qobj):
            raise TypeError(f"tensor() argument {i} must be a Qobj")
        factors.append(obj)
    if not factors:
        raise ValueError("tensor() requires at least one Qobj")

    first = factors[0]
    torch_backends = [
        not isinstance(obj._data, (np.ndarray, sparse.csr_array)) for obj in factors
    ]
    if any(torch_backends) and not all(torch_backends):
        raise TypeError(
            "Cannot mix PyTorch and NumPy/SciPy Qobj data. "
            "Convert all objects to the same backend first."
        )
    if all(torch_backends):
        import torch

        first_data = first._data
        if not isinstance(first_data, torch.Tensor):
            raise TypeError("Unsupported Qobj data backend")
        device = first_data.device
        for obj in factors[1:]:
            data = obj._data
            if not isinstance(data, torch.Tensor):
                raise TypeError("Unsupported Qobj data backend")
            if data.device != device:
                raise ValueError("PyTorch Qobj operands must be on the same device")

    # Convert dense factors *before* the first product when there is any
    # sparse factor, so early products cannot accidentally grow dense.
    use_sparse = any(obj.is_sparse() for obj in factors)
    result_data = first._data
    if use_sparse and isinstance(result_data, np.ndarray):
        result_data = sparse.csr_array(result_data)
    left_dims = list(first.dims[0])
    right_dims = list(first.dims[1])
    result_type = first.qtype
    tolerance = first._tol

    for obj in factors[1:]:
        right_data = obj._data
        if use_sparse and isinstance(right_data, np.ndarray):
            right_data = sparse.csr_array(right_data)
        result_data = _kron_data(result_data, right_data)
        left_dims.extend(obj.dims[0])
        right_dims.extend(obj.dims[1])
        if obj.qtype != result_type:
            result_type = Qtype.oper
        tolerance = max(tolerance, obj._tol)

    return Qobj(
        result_data,
        dims=(tuple(left_dims), tuple(right_dims)),
        qtype=result_type,
        tol=tolerance,
    )
