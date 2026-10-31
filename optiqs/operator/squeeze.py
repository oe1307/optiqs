from optiqs.operator.annihilation import destroy
from optiqs.utils import expm
from optiqs.utils.typing import Qobj


def squeeze(dim: int, sq_param: complex) -> Qobj:
    a = destroy(dim)
    hamiltonian = 0.5 * (sq_param.conjugate() * a * a - sq_param * a.dag() * a.dag())
    return expm(hamiltonian)
