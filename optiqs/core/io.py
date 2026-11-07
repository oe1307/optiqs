"""Safe, portable serialization of quantum objects.

The .npz format contains only NumPy arrays (no pickled Python objects).
Sparse CSR data is stored as CSR components to avoid densification.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
from scipy.sparse import csr_array

from optiqs.core.q_obj import Qobj, Qtype

_FORMAT_VERSION = 1


def save_state(state: Qobj, filename: str | PathLike[str]) -> None:
    """Save a quantum state or operator to an ``.npz`` file.

    Preserves complex amplitudes, dtype, subsystem dimensions, quantum type,
    tolerance, and dense/CSR/PyTorch storage backend. CSR data is saved
    without converting the matrix to dense storage. PyTorch tensors are
    detached and copied to the CPU for serialization; autograd history and
    the original device are not saved.

    Parameters
    ----------
    state : Qobj
        The ket, bra, or operator to save.
    filename : str or path-like
        Destination path, typically ending with ``.npz``. The exact path is
        used, even when it does not have a ``.npz`` suffix.
    """
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")

    data = state._data
    payload: dict[str, Any] = {
        "version": np.array(_FORMAT_VERSION, dtype=np.int64),
        "dims_left": np.asarray(state.dims[0], dtype=np.int64),
        "dims_right": np.asarray(state.dims[1], dtype=np.int64),
        "qtype": np.array(state.qtype.value),
        "tol": np.array(state._tol, dtype=np.float64),
        "shape": np.asarray(data.shape, dtype=np.int64),
    }

    if isinstance(data, csr_array):
        payload.update(
            backend=np.array("csr"),
            data=data.data,
            indices=data.indices,
            indptr=data.indptr,
        )
    elif isinstance(data, np.ndarray):
        payload.update(backend=np.array("numpy"), data=data)
    else:
        # torch is an optional dependency. Import it only for torch states.
        import torch

        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Unsupported Qobj data type: {type(data)}")
        try:
            values = state.to_numpy()
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                f"Cannot serialize PyTorch dtype {data.dtype} as a NumPy array"
            ) from exc
        payload.update(backend=np.array("torch"), data=values)

    # Opening a file ourselves keeps the caller's exact filename. np.savez*
    # would silently append '.npz' to filenames without that suffix.
    with open(filename, "wb") as output:
        np.savez_compressed(output, **payload)


def load_state(filename: str | PathLike[str]) -> Qobj:
    """Load a quantum state or operator written by :func:`save_state`.

    Uses ``allow_pickle=False`` when reading. PyTorch-backed objects are
    restored as CPU tensors (and require PyTorch to be installed); NumPy and
    sparse CSR objects retain their original storage type. The returned Qobj
    is independent of the saved object.

    Raises
    ------
    ValueError
        If the file has an unsupported format version or invalid metadata.
    ImportError
        If the file holds a PyTorch tensor but PyTorch is not installed.
    """
    with np.load(filename, allow_pickle=False) as archive:
        required = {
            "version",
            "backend",
            "dims_left",
            "dims_right",
            "qtype",
            "tol",
            "shape",
            "data",
        }
        if not required.issubset(archive.files):
            raise ValueError("Invalid quantum state file: missing required fields")

        def scalar(key: str) -> Any:
            value = archive[key]
            if value.shape != ():
                raise ValueError(f"Invalid quantum state file: {key} must be scalar")
            return value.item()

        if scalar("version") != _FORMAT_VERSION:
            raise ValueError("Unsupported quantum state file version")

        backend = scalar("backend")
        shape_values = archive["shape"]
        if shape_values.ndim != 1 or shape_values.size != 2:
            raise ValueError("Invalid quantum state file: shape must have 2 entries")
        # SciPy sparse constructors require a statically known 2D shape.
        shape: tuple[int, int] = (int(shape_values[0]), int(shape_values[1]))
        if any(dim <= 0 for dim in shape):
            raise ValueError("Invalid quantum state file: non-positive shape")

        def dimensions(key: str) -> tuple[int, ...]:
            values = archive[key]
            if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Invalid quantum state file: {key} must be integers")
            return tuple(int(value) for value in values)

        dims = (dimensions("dims_left"), dimensions("dims_right"))
        try:
            qtype = Qtype(scalar("qtype"))
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid quantum state file: unknown qtype") from exc
        tol = float(scalar("tol"))
        data = archive["data"]

        if backend == "csr":
            if not {"indices", "indptr"}.issubset(archive.files):
                raise ValueError("Invalid quantum state file: missing CSR fields")
            sparse_data = csr_array(
                (data, archive["indices"], archive["indptr"]), shape=shape
            )
            return Qobj(sparse_data, dims=dims, qtype=qtype, tol=tol)

        if backend == "numpy" or backend == "torch":
            if data.shape != shape:
                raise ValueError("Invalid quantum state file: data shape mismatch")
            if backend == "torch":
                try:
                    import torch
                except ImportError as exc:
                    raise ImportError(
                        "Loading this PyTorch-backed state requires PyTorch"
                    ) from exc
                torch_data = torch.from_numpy(data.copy())
                return Qobj(torch_data, dims=dims, qtype=qtype, tol=tol)
            return Qobj(data, dims=dims, qtype=qtype, tol=tol)

        raise ValueError("Invalid quantum state file: unknown backend")
