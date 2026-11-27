"""Targeted numerical regressions for extreme complex64 PNRD inputs."""

import numpy as np
import pytest
from scipy.sparse import csr_array

import optiqs as opt
from optiqs.core.q_obj import Qtype


@pytest.fixture(params=[np.float32(3e38), np.nextafter(np.float32(0), np.float32(1))])
def coefficients(request):
    scale = float(request.param)
    return np.array([complex(scale, scale), scale], dtype=np.complex64)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("bra", [False, True])
def test_pnrd_extreme_complex64(coefficients, sparse, bra):
    data = coefficients.reshape(1, -1) if bra else coefficients.reshape(-1, 1)
    if sparse:
        data = csr_array(data)
    # Qobj's sparse pruning evaluates abs in the storage dtype. The
    # measurement itself must work with floating-point errors enabled.
    with np.errstate(over="ignore"):
        state = opt.Qobj(data, tol=0, qtype=Qtype.bra if bra else Qtype.ket)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        probabilities = opt.pnrd(state)
        outcome, collapsed = opt.measure_photon_number(
            state, collapse=True, rng=0
        )

    np.testing.assert_allclose(probabilities, [2 / 3, 1 / 3], rtol=1e-12)
    expected = np.zeros(2, dtype=np.complex128)
    amplitude = np.complex128(coefficients[outcome])
    expected[outcome] = amplitude / abs(amplitude)
    np.testing.assert_allclose(
        collapsed.to_numpy().reshape(-1), expected, rtol=1e-6
    )
    assert collapsed.dtype == np.complex64
    assert collapsed.dims == state.dims
    assert collapsed.qtype == state.qtype
    assert collapsed.is_sparse() == sparse


@pytest.mark.parametrize("sparse", [False, True])
def test_identity_projector_extreme_complex64(coefficients, sparse):
    data = coefficients.reshape(-1, 1)
    if sparse:
        data = csr_array(data)
    with np.errstate(over="ignore"):
        state = opt.Qobj(data, dims=((1, 2), (1, 1)), tol=0)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        outcome, collapsed = opt.measure_photon_number(
            state, mode=0, collapse=True, rng=0
        )

    assert outcome == 0
    expected = coefficients.astype(np.complex128)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(
        collapsed.to_numpy().reshape(-1), expected, rtol=1e-6
    )
    assert collapsed.dtype == np.complex64
    assert collapsed.is_sparse() == sparse


def test_joint_and_low_rank_extreme_complex64(coefficients):
    values = np.zeros((4, 1), dtype=np.complex64)
    values[::2, 0] = coefficients
    state = opt.Qobj(values, dims=((2, 2), (1, 1)), tol=0)
    factors = opt.LowRankDensity(values, (2, 2), tol=0)
    expected = np.array([[2 / 3, 0], [1 / 3, 0]])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for obj in (state, factors):
            np.testing.assert_allclose(
                opt.joint_pnrd(obj), expected, rtol=1e-12
            )
            np.testing.assert_allclose(
                opt.pnrd(obj, mode=0), expected.sum(axis=1), rtol=1e-12
            )
