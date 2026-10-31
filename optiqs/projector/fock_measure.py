import math

import numpy as np

from optiqs.utils import Qobj, Qtype


def _loss_matrix(dim: int, eta: float) -> np.ndarray:
    """
    Photon-loss transition matrix on number probabilities.

    Maps input photon-number distribution p_in[n] to
    output distribution p_out[m] = sum_n U[m, n] p_in[n].
    """
    u = np.zeros((dim, dim), dtype=float)
    for n in range(dim):
        for m in range(n + 1):
            u[m, n] = math.comb(n, m) * ((1 - eta) ** m) * (eta ** (n - m))
    return u


def _apply_loss_nd(prob: np.ndarray, loss_decibel) -> np.ndarray:
    """
    Apply independent photon loss to each mode of an N-dimensional
    joint photon-number probability array.
    """
    if isinstance(loss_decibel, int | float):
        loss_list = [float(loss_decibel)] * prob.ndim
    else:
        loss_list = list(loss_decibel)
        if len(loss_list) != prob.ndim:
            raise ValueError("loss_decibel must be a scalar or have one value per mode")

    for axis, loss_db in enumerate(loss_list):
        if loss_db < 0:
            raise ValueError("loss_decibel must be non-negative")

        if loss_db == 0:
            continue

        eta = 1 - 10 ** (-loss_db / 10)
        u = _loss_matrix(prob.shape[axis], eta)

        # Apply u along the given axis:
        # new_prob[..., m, ...] = sum_n u[m, n] prob[..., n, ...]
        prob = np.moveaxis(prob, axis, 0)
        orig_shape = prob.shape
        prob = prob.reshape(orig_shape[0], -1)
        prob = u @ prob
        prob = prob.reshape(orig_shape)
        prob = np.moveaxis(prob, 0, axis)

    return prob


def pnrd(state: Qobj, loss_decibel: int | float | list = 0) -> np.ndarray:
    """
    Photon-number-resolving detector (PNRD) measurement for an arbitrary
    number of modes.

    Args:
        state: input quantum state (ket or density matrix)
        loss_decibel:
            - scalar: same loss applied to every mode
            - sequence: per-mode loss in dB

    Returns:
        Joint photon-number probability array with shape state.dims[0].
        For M modes, returns P(n1, ..., nM).
    """
    state_array = state.numpy().data
    dims = tuple(state.dims[0])

    if len(dims) == 0:
        raise ValueError("state must have at least one mode")

    if state.qtype == Qtype.ket:
        expected_size = np.prod(dims)
        if state_array.size != expected_size:
            raise ValueError("ket size does not match dims")
        prob = np.abs(state_array.reshape(dims)) ** 2

    elif state.qtype == Qtype.matrix:
        if state_array.ndim != 2 or state_array.shape[0] != state_array.shape[1]:
            raise ValueError("density matrix must be square")

        expected_size = np.prod(dims)
        if state_array.shape[0] != expected_size:
            raise ValueError("density matrix size does not match dims")

        diag = np.real_if_close(np.diag(state_array))
        prob = diag.reshape(dims)

    else:
        raise ValueError("state must be ket or density matrix")

    prob = np.real_if_close(prob)
    prob = np.asarray(prob, dtype=float)
    assert np.isclose(prob.sum(), 1)
    prob = _apply_loss_nd(prob, loss_decibel)
    assert np.isclose(prob.sum(), 1)

    return prob
