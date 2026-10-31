import numpy as np
import scipy as sp

from optiqs.utils.typing import Qobj


def destroy(dim: int) -> Qobj:
    """Annihilation operator q_operator."""
    return Qobj(sp.sparse.diags(np.sqrt(np.arange(1, dim)), offsets=1))
