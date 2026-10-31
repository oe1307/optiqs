from functools import lru_cache

import numpy as np

from optiqs.utils.exponential import expm
from optiqs.utils.typing import Qobj


@lru_cache()
def rot(dim: int, theta: float) -> Qobj:
    return expm(Qobj(1j * theta * np.diag(np.arange(dim))))
