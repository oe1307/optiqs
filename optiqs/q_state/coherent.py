import numpy as np
from scipy.special import factorial

from optiqs.utils import Qobj


def coherent(dim: int, alpha: float | complex) -> Qobj:
    return Qobj(alpha ** np.arange(dim) / np.sqrt(factorial(np.arange(dim)))).unit()
