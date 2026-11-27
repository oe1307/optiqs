"""Rotation-symmetric binomial bosonic code states."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike
from scipy.sparse import csr_array
from scipy.special import gammaln

from optiqs.core.q_obj import Qobj


def rotational_binomial(
    dim: int,
    cutoff: int | None = None,
    order: int | None = None,
    sector: int = 0,
    dtype: DTypeLike = np.complex128,
    tol: float = 1e-17,
    *,
    degree: int | None = None,
) -> Qobj:
    """Construct a normalized rotation-symmetric binomial codeword.

    ``order=N`` specifies an N-fold rotation symmetry with angle ``2*pi/N``
    (180, 120, and 90 degrees for N=2, 3, 4). The codeword uses Fock
    levels ``r + k*N`` for ``k=0, ..., K``, with coefficients

        sqrt(comb(K, k) / 2**K) * (-1)**(logical*k),

    where ``r = sector % order``, ``logical = (sector // order) % 2``,
    and ``K = degree`` when explicitly specified. Alternatively, the legacy
    ``cutoff`` argument gives ``K = floor((cutoff - r) / order)``.
    Exactly one of ``degree`` and ``cutoff`` must be supplied. In particular,
    ``sector=0`` generates the all-positive state and ``sector=order``
    generates the alternating-sign state. They are orthogonal for K >= 1.
    Both have N-fold rotational symmetry; a rotation by ``pi/N``
    interchanges them (up to a global phase).

    Parameters
    ----------
    dim : int
        Fock-space simulation dimension. Photon numbers from 0 to
        ``dim - 1`` are representable. It must exceed ``cutoff`` in legacy
        mode or ``r + degree * order`` in degree mode.
    cutoff : int, optional
        Legacy maximum allowed photon number (inclusive), not the binomial
        degree K. Mutually exclusive with ``degree``. A nontrivial logical
        pair with ``sector=0`` and ``sector=order`` requires
        ``cutoff >= order``.
    order : int
        Positive rotation-symmetry order N. The angular period is
        ``360 / order`` degrees when K >= 1 and both parities occur.
    sector : int, optional
        Selects the Fock-number remainder ``r = sector % order`` and
        the logical phase ``(sector // order) % 2``. Must be nonnegative;
        in legacy cutoff mode it must also satisfy ``sector <= cutoff``.
        Default is 0.
    dtype : DTypeLike, optional
        Coefficient data type (default: numpy.complex128).
    tol : float, optional
        Sparse coefficient-pruning threshold satisfying 0 <= tol < 1.
        Default is 1e-17.
    degree : int, optional, keyword-only
        Binomial degree K (nonnegative), as in the RS-binomial definition.
        The highest occupied Fock level is ``r + K * order`` before
        pruning. Use ``degree >= 1`` for an orthogonal logical pair.
        For ``sector=0`` or ``sector=order`` this is exactly ``K * order``.

    Returns
    -------
    Qobj
        Normalized single-mode ket, stored as a CSR sparse column vector.

    Notes
    -----
    Logarithmic binomial coefficients avoid factorial overflow. Each
    codeword is normalized before sparsity pruning; large ``tol`` values
    may reduce its final norm. For ``K=0``, a single Fock state is returned
    (its rotation symmetry is not uniquely N-fold).

    For ``order=2, cutoff=4``, ``sector=0`` generates
    ``(|0> + sqrt(2)|2> + |4>)/2``, while ``sector=2`` generates
    ``(|0> - sqrt(2)|2> + |4>)/2``. The same states can be constructed
    with ``rotational_binomial(dim, order=2, degree=2, sector=...)``.
    """
    if (cutoff is None) == (degree is None):
        raise ValueError("specify exactly one of cutoff or degree")

    for name, value in (
        ("dim", dim),
        ("order", order),
        ("sector", sector),
        (
            "cutoff" if cutoff is not None else "degree",
            cutoff if cutoff is not None else degree,
        ),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    assert order is not None
    dim, order, sector = int(dim), int(order), int(sector)
    if dim < 1:
        raise ValueError("dim must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    if sector < 0:
        raise ValueError("sector must be nonnegative")
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")

    logical, remainder = divmod(sector, order)
    if cutoff is not None:
        cutoff = int(cutoff)
        if cutoff < 0:
            raise ValueError("cutoff must be nonnegative")
        if sector > cutoff:
            raise ValueError("sector must satisfy 0 <= sector <= cutoff")
        if cutoff >= dim:
            raise ValueError("dim must be at least cutoff + 1")
        degree = (cutoff - remainder) // order
    else:
        assert degree is not None
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        if remainder + degree * order >= dim:
            raise ValueError("dim must exceed remainder + degree * order")

    k = np.arange(degree + 1, dtype=np.int64)
    n = remainder + k * order

    log_comb = gammaln(degree + 1) - gammaln(k + 1) - gammaln(degree - k + 1)
    log_amp = 0.5 * log_comb
    coeff = np.exp(log_amp - log_amp.max())
    coeff /= np.linalg.norm(coeff)

    if logical % 2:
        # Alternating signs give the orthogonal logical codeword.
        coeff[1::2] *= -1

    data = csr_array((coeff, (n, np.zeros(n.size, dtype=np.int64))), shape=(dim, 1))
    return Qobj(data, dtype=dtype, tol=tol)


rs_binomial = rotational_binomial
