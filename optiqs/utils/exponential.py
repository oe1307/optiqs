import scipy as sp

from optiqs.utils.typing import Qobj


def expm(state: Qobj) -> Qobj:
    if state.is_sparse:
        return Qobj(
            sp.sparse.linalg.expm(state.data.tocsc()),  # type: ignore
            dims=(state.dims[0], state.dims[1]),
        )
    else:
        return Qobj(sp.linalg.expm(state.data), dims=(state.dims[0], state.dims[1]))
