from optiqs.operator.annihilation import destroy
from optiqs.utils.typing import Qobj


def num(dim: int) -> Qobj:
    """Number operator"""
    a = destroy(dim)
    return a.dag() * a
