import numpy as np
import pytest
from scipy import sparse

from optiqs.core.low_rank_density import LowRankDensity
from optiqs.core.q_obj import Qobj, Qtype


def make_state(amplitudes, use_sparse, qtype, tol=0):
    data = np.asarray(amplitudes).reshape(-1, 1)
    if qtype == Qtype.bra:
        data = data.conj().T
    if use_sparse:
        data = sparse.csr_array(data)
    return Qobj(data, qtype=qtype, tol=tol)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("qtype", [Qtype.ket, Qtype.bra])
@pytest.mark.parametrize("amplitudes", [
    [0, 0], [np.nan, 1], [np.inf, 1], [-np.inf, 1],
    [complex(1, np.nan), 1], [complex(1, np.inf), 1],
])
def test_invalid_amplitudes(use_sparse, qtype, amplitudes):
    state = make_state(amplitudes, use_sparse, qtype)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        with pytest.raises(ValueError):
            LowRankDensity.from_ket(state)


@pytest.mark.parametrize("use_sparse", [False, True])
@pytest.mark.parametrize("qtype", [Qtype.ket, Qtype.bra])
@pytest.mark.parametrize("amplitudes, expected", [
    ([3 + 4j, 2 - 1j], [3 + 4j, 2 - 1j]),
    ([1e308, -1e308], [1, -1]),
    ([1e308 + 1e308j, 1e308 - 1e308j], [1 + 1j, 1 - 1j]),
    ([5e-324, 1e-323], [1, 2]),
    ([complex(5e-324, 5e-324), 0], [1 + 1j, 0]),
])
def test_stable_normalization(use_sparse, qtype, amplitudes, expected, monkeypatch):
    state = make_state(amplitudes, use_sparse, qtype)
    before = state._data.copy()
    expected = np.asarray(expected, dtype=np.complex128)
    expected /= np.linalg.norm(expected)

    def forbid_dense(*args, **kwargs):
        pytest.fail("Unexpected sparse-to-dense conversion")

    if use_sparse:
        monkeypatch.setattr(sparse.csr_array, "toarray", forbid_dense)
        monkeypatch.setattr(sparse.csr_array, "todense", forbid_dense)
        monkeypatch.setattr(sparse.csc_array, "toarray", forbid_dense)
        monkeypatch.setattr(sparse.csc_array, "todense", forbid_dense)

    result = LowRankDensity.from_ket(state)
    assert result.trace == pytest.approx(1, rel=0, abs=1e-14)
    assert result.rank == 1
    assert result.dims == (2,)
    factors = result.factors
    if use_sparse:
        assert isinstance(factors, sparse.csr_array)
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(
                getattr(state._data, name), getattr(before, name)
            )
        coo = factors.tocoo()
        actual = np.zeros(2, dtype=factors.dtype)
        actual[coo.row] = coo.data
    else:
        assert isinstance(factors, np.ndarray)
        np.testing.assert_array_equal(state._data, before)
        actual = factors[:, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("qtype", [Qtype.ket, Qtype.bra])
def test_sparse_tolerance(qtype):
    state = make_state([100.0, 1.0], True, qtype, tol=0.1)
    before = state._data.copy()
    result = LowRankDensity.from_ket(state)
    assert result.trace == pytest.approx(1, rel=0, abs=1e-14)
    assert result.factors.nnz == 1
    np.testing.assert_allclose(result.factors.data, [1.0])
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(
            getattr(state._data, name), getattr(before, name)
        )


@pytest.mark.parametrize("qtype", [Qtype.ket, Qtype.bra])
@pytest.mark.parametrize("amplitudes, tol", [
    ([100.0, 1.0], 0.99999),
    ([1e-20, 1e-21], 1e-17),
])
def test_sparse_tolerance_removes_all(qtype, amplitudes, tol):
    state = make_state(amplitudes, True, qtype, tol=tol)
    with pytest.raises(ValueError):
        LowRankDensity.from_ket(state)
