import scipy as sp

from optiqs.utils import Qobj


def fock(dim: int, n: int):
    """
    Return the Fock state |n>
    """
    return Qobj(sp.sparse.csr_matrix(([1], ([n], [0])), shape=(dim, 1)))


def vacuum(dim: int):
    """
    Return the vacuum state |0>, i.e. the Fock state with n=0.
    """
    return fock(dim, n=0)
