import numpy as np

from optiqs.utils import Qobj, Qtype, wigner


def homodyne(state: Qobj, axis: np.ndarray):
    if state.qtype == Qtype.ket:
        wigner_function = wigner(state, axis, axis)
        distribution = wigner_function.sum(axis=1)
        return distribution / distribution.sum()
    elif state.qtype == Qtype.matrix:
        raise NotImplementedError(
            "homodyne measurement for density matrix is not implemented yet"
        )
    else:
        raise ValueError("state must be ket for homodyne measurement")
