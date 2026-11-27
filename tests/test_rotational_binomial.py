import math

import numpy as np
import pytest

from optiqs import rotational_binomial, rs_binomial


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("cutoff", [1, 4, 10])
@pytest.mark.parametrize("logical", [0, 1])
def test_paper_codeword(order, cutoff, logical):
    dim = order * cutoff + 1
    state = rs_binomial(dim, cutoff, order, sector=logical * order)
    expected = np.zeros(dim, dtype=np.complex128)
    for k in range(cutoff + 1):
        expected[k * order] = (-1) ** (logical * k) * math.sqrt(
            math.comb(cutoff, k) / 2**cutoff
        )
    actual = state.to_numpy().ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual), 1, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("cutoff", [1, 5])
def test_logical_pair_is_orthogonal(order, cutoff):
    dim = order * cutoff + 1
    zero = rs_binomial(dim, cutoff, order).to_numpy().ravel()
    one = rs_binomial(dim, cutoff, order, sector=order).to_numpy().ravel()
    np.testing.assert_allclose(np.vdot(zero, one), 0, atol=1e-14)


@pytest.mark.parametrize("sector", [0, 1, 2, 3, 4, 5])
def test_sector_with_nonzero_remainder(sector):
    order, cutoff = 3, 4
    logical, remainder = divmod(sector, order)
    dim = remainder + cutoff * order + 1
    state = rotational_binomial(dim, cutoff=cutoff, order=order, sector=sector)
    expected = np.zeros(dim, dtype=np.complex128)
    for k in range(cutoff + 1):
        expected[remainder + k * order] = (-1) ** (logical * k) * math.sqrt(
            math.comb(cutoff, k) / 2**cutoff
        )
    np.testing.assert_allclose(state.to_numpy().ravel(), expected, atol=1e-14)


def test_cutoff_zero_is_single_fock_state():
    state = rs_binomial(3, cutoff=0, order=4, sector=6)
    np.testing.assert_array_equal(state.to_numpy().ravel(), [0, 0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 2, "cutoff": -1},
        {"order": 2, "cutoff": 4},
        {"order": 2, "cutoff": 3, "sector": 1},
        {"order": 0, "cutoff": 1},
        {"order": 2, "cutoff": 1, "sector": -1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rs_binomial(7, **kwargs)


@pytest.mark.parametrize("cutoff", [True, 2.5, "3"])
def test_cutoff_must_be_integer(cutoff):
    with pytest.raises(TypeError):
        rs_binomial(10, cutoff, 2)


def test_cutoff_accepts_numpy_integer():
    state = rs_binomial(7, cutoff=np.int64(3), order=np.int64(2))
    assert state.to_numpy().ravel()[6] != 0
