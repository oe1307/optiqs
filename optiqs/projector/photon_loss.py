import math
from functools import reduce

from optiqs.operator.annihilation import destroy
from optiqs.operator.qeye import qeye
from optiqs.utils.tensor import tensor
from optiqs.utils.typing import Qobj, Qtype


def dissipator(rho: Qobj, L: Qobj) -> Qobj:
    return L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L)


def lindblad_rhs(rho: Qobj, Ls: list[Qobj]) -> Qobj:
    out = 0 * rho
    for L in Ls:
        out = out + dissipator(rho, L)
    return out


def rk4_step(rho: Qobj, dt: float, Ls: list[Qobj]) -> Qobj:
    k1 = lindblad_rhs(rho, Ls)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, Ls)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, Ls)
    k4 = lindblad_rhs(rho + dt * k3, Ls)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def local_destroy(dims: list[int], target: int) -> Qobj:
    ops = [qeye(dim) for dim in dims]
    ops[target] = destroy(dims[target])
    return reduce(tensor, ops)


def _normalize_mode_args(num_modes: int, decibels, targets):
    if targets is None:
        targets = list(range(num_modes))
    elif isinstance(targets, int):
        targets = [targets]
    else:
        targets = list(targets)

    if isinstance(decibels, (int, float)):
        decibels = [float(decibels)] * len(targets)
    else:
        decibels = list(decibels)

    if len(decibels) != len(targets):
        raise ValueError("len(decibels) must match len(targets)")

    for t in targets:
        if t < 0 or t >= num_modes:
            raise ValueError(f"invalid target mode: {t}")

    for db in decibels:
        if db < 0:
            raise ValueError("decibel must be non-negative for loss")

    return decibels, targets


def loss(
    state: Qobj,
    decibels,
    targets=None,
    t_total: float = 1.0,
    steps: int = 15,
    renormalize_trace: bool = True,
) -> Qobj:
    """
    Generic photon loss for 1-mode / 2-mode / multi-mode states.

    Parameters
    ----------
    state:
        ket or density matrix
    decibels:
        - scalar: same loss for all selected targets
        - list: one dB value per target
    targets:
        - None: apply to all modes
        - int: apply to one mode
        - list[int]: apply to selected modes
    """
    if state.qtype == Qtype.ket:
        state = state.dm()

    dims = list(state.dims[0])
    num_modes = len(dims)

    decibels, targets = _normalize_mode_args(num_modes, decibels, targets)

    dt = t_total / steps
    Ls = []

    for db, target in zip(decibels, targets):
        eta = 1 - 10 ** (-db / 10.0)
        kappa = -math.log(1 - eta) / t_total
        a = local_destroy(dims, target)
        Ls.append(math.sqrt(kappa) * a)

    for _ in range(steps):
        state = rk4_step(state, dt, Ls)

    if renormalize_trace:
        tr = state.tr()
        if abs(tr) > 0:
            state = state / tr

    return state
