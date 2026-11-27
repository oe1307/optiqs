import math

import numpy as np
import pytest

from optiqs import rotational_binomial, rs_binomial


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("degree", [1, 4, 10])
@pytest.mark.parametrize("logical", [0, 1])
def test_paper_codeword(order, degree, logical):
    dim = order * degree + 1
    state = rs_binomial(dim, order=order, degree=degree, sector=logical * order)
    expected = np.zeros(dim, dtype=np.complex128)
    for k in range(degree + 1):
        expected[k * order] = (-1) ** (logical * k) * math.sqrt(
            math.comb(degree, k) / 2**degree
        )
    actual = state.to_numpy().reshape(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual), 1, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("degree", [1, 5])
def test_logical_pair_is_orthogonal(order, degree):
    dim = order * degree + 1
    zero = rs_binomial(dim, order=order, degree=degree).to_numpy().ravel()
    one = rs_binomial(
        dim, order=order, degree=degree, sector=order
    ).to_numpy().ravel()
    np.testing.assert_allclose(np.vdot(zero, one), 0, atol=1e-14)


@pytest.mark.parametrize("sector", [0, 1, 2, 3, 4, 5])
def test_legacy_cutoff_with_nonzero_remainder(sector):
    order, cutoff, dim = 3, 11, 12
    degree = (cutoff - sector % order) // order
    legacy = rotational_binomial(dim, cutoff, order, sector)
    explicit = rotational_binomial(
        dim, order=order, degree=degree, sector=sector
    )
    np.testing.assert_allclose(legacy.to_numpy(), explicit.to_numpy())


def test_degree_zero_is_single_fock_state():
    state = rs_binomial(3, order=4, degree=0, sector=6)
    np.testing.assert_array_equal(state.to_numpy().ravel(), [0, 0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 2},
        {"order": 2, "degree": 2, "cutoff": 4},
        {"order": 2, "degree": -1},
        {"order": 2, "degree": 4},
        {"order": 2, "degree": 3, "sector": 1},
        {"order": 0, "degree": 1},
        {"order": 2, "degree": 1, "sector": -1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        rs_binomial(7, **kwargs)


@pytest.mark.parametrize("degree", [True, 2.5, "3"])
def test_degree_must_be_integer(degree):
    with pytest.raises(TypeError):
        rs_binomial(10, order=2, degree=degree)


def test_degree_accepts_numpy_integer():
    state = rs_binomial(7, order=np.int64(2), degree=np.int64(3))
    assert state.to_numpy().ravel()[6] != 0
