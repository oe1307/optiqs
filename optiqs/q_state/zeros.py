import numpy as np
import scipy as sp

from optiqs.utils import Qobj


def zeros(input_dims: int | list, output_dims: int | list = [1]) -> Qobj:
    if isinstance(input_dims, int):
        input_dims = [input_dims]
    if isinstance(output_dims, int):
        output_dims = [output_dims]

    return Qobj(
        sp.sparse.csr_matrix((np.prod(input_dims), np.prod(output_dims))),
        dims=(input_dims, output_dims),
    )
