import os

import numpy as np
import scipy as sp

from optiqs.utils.typing import Qobj


def qsave(state: Qobj, path: str) -> None:
    if state.is_sparse:
        sp.sparse.save_npz(path, state.data)
    else:
        np.save(path, state.data)


def qload(path: str, input_dims: list[int], output_dims: list[int]) -> Qobj:
    if os.path.splitext(path)[-1] == ".npz":
        data = sp.sparse.load_npz(path)
    elif os.path.splitext(path)[-1] == ".npy":
        data = np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .npy or .csr for sparse arrays.")
    return Qobj(data, dims=(input_dims, output_dims))
