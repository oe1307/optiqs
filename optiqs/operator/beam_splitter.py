import os
from functools import lru_cache

import numpy as np

from optiqs.operator.annihilation import destroy
from optiqs.operator.qeye import qeye
from optiqs.utils.exponential import expm
from optiqs.utils.save_load import qload, qsave
from optiqs.utils.tensor import tensor
from optiqs.utils.typing import Qobj


@lru_cache()
def bs(dim: int, cache: str | None = "~/.cache/optiqs/bs") -> Qobj:
    """Beam splitter unitary q_operator."""
    if cache is not None:
        cache = os.path.expanduser(cache + f"_{dim}.npz")

    if cache is not None and os.path.exists(cache):
        return qload(cache, [dim, dim], [dim, dim])

    a1 = tensor(destroy(dim), qeye(dim))
    a2 = tensor(qeye(dim), destroy(dim))
    u_bs = expm(-(np.pi / 4) * (a1.dag() * a2 - a1 * a2.dag()))

    if cache is not None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        qsave(u_bs, cache)

    return u_bs
