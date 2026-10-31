import numpy as np

from optiqs.utils.typing import Qobj, Qtype


def fidelity(state1: Qobj, state2: Qobj) -> float:
    if state1.qtype == Qtype.ket and state2.qtype == Qtype.ket:
        _fidelity = (np.abs((state1.dag() * state2).numpy().data) ** 2).item()
    elif state1.qtype == Qtype.matrix and state2.qtype == Qtype.matrix:
        sqrt_state1 = np.sqrt(state1.numpy().data)
        _fidelity = (np.trace(np.sqrt(sqrt_state1 * state2.data * sqrt_state1))) ** 2
    else:
        rho = state2 if state2.qtype == Qtype.matrix else state1
        psi = state1 if state1.qtype == Qtype.ket else state2
        _fidelity = (psi.dag() * rho * psi).numpy().data.item()

    assert abs(np.imag(_fidelity)) <= 1e-12
    _fidelity = np.real(_fidelity)
    assert _fidelity >= -1e-12 and _fidelity <= 1.0 + 1e-12
    return float(np.clip(_fidelity, 0.0, 1.0))
