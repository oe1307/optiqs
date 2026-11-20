"""Optional, bounded-memory CSR photon-loss transition kernel.

Imported only when ``photon_loss(..., accelerator='numba')`` is requested.
Numba is an optional dependency; ordinary OptiQS imports do not require it.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np

# Numba is optional. Keep import behavior consistent with _wigner_numba:
# static checking must also work without the optional JIT dependency.
njit = import_module("numba").njit


@njit(cache=True)
def sparse_loss_transitions(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    row_photons: np.ndarray,
    col_photons: np.ndarray,
    counts: np.ndarray,
    log_factorials: np.ndarray,
    log_eta: float,
    log_loss: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one bounded COO batch, avoiding vectorized temporary arrays."""
    size = 0
    for index in range(counts.size):
        size += counts[index]
    out_rows = np.empty(size, dtype=np.int64)
    out_cols = np.empty(size, dtype=np.int64)
    out_values = np.empty(size, dtype=values.dtype)
    position = 0
    for index in range(rows.size):
        nr = row_photons[index]
        nc = col_photons[index]
        for lost in range(counts[index]):
            log_weight = 0.5 * (
                log_factorials[nr]
                + log_factorials[nc]
                - 2.0 * log_factorials[lost]
                - log_factorials[nr - lost]
                - log_factorials[nc - lost]
                + (nr + nc - 2 * lost) * log_eta
                + 2 * lost * log_loss
            )
            contribution = values[index] * np.exp(log_weight)
            if contribution != 0:
                shift = lost * stride
                out_rows[position] = rows[index] - shift
                out_cols[position] = cols[index] - shift
                out_values[position] = contribution
                position += 1
    return out_rows[:position], out_cols[:position], out_values[:position]


@njit(cache=True)
def sparse_loss_transitions_table(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    row_photons: np.ndarray,
    col_photons: np.ndarray,
    counts: np.ndarray,
    amplitudes: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use precomputed <n-lost|K_lost|n> when many transitions reuse them."""
    size = 0
    for index in range(counts.size):
        size += counts[index]
    out_rows = np.empty(size, dtype=np.int64)
    out_cols = np.empty(size, dtype=np.int64)
    out_values = np.empty(size, dtype=values.dtype)
    position = 0
    for index in range(rows.size):
        nr = row_photons[index]
        nc = col_photons[index]
        for lost in range(counts[index]):
            contribution = values[index] * amplitudes[lost, nr] * amplitudes[lost, nc]
            if contribution != 0:
                shift = lost * stride
                out_rows[position] = rows[index] - shift
                out_cols[position] = cols[index] - shift
                out_values[position] = contribution
                position += 1
    return out_rows[:position], out_cols[:position], out_values[:position]
