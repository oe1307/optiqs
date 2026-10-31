import numpy as np
import scipy as sp

from optiqs.utils.typing import Qobj


def tensor(op1: Qobj, op2: Qobj) -> Qobj:
    """tensor product"""
    if op1.is_sparse or op2.is_sparse:
        return Qobj(
            sp.sparse.kron(op1.data, op2.data),
            dims=(op1.dims[0] + op2.dims[0], op1.dims[1] + op2.dims[1]),
        )
    return Qobj(
        np.kron(op1.data, op2.data),
        dims=(op1.dims[0] + op2.dims[0], op1.dims[1] + op2.dims[1]),
    )
