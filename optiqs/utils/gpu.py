from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    import torch


def get_device() -> torch.device:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_numpy_dtype(
    dtype: DTypeLike | torch.dtype | None,
) -> np.dtype[Any] | None:
    if dtype is None:
        return None
    try:
        return np.dtype(cast(DTypeLike, dtype))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid NumPy dtype: {dtype!r}") from exc
