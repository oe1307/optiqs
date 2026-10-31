import scipy as sp

from optiqs.utils.typing import Qobj


def qeye(dim: int) -> Qobj:
    return Qobj(sp.sparse.eye(dim))
