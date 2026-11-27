"""Shared input validation for state constructors."""

import numpy as np


def validate_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def validate_dim(dim: int) -> int:
    dim = validate_integer("dim", dim)
    if dim < 1:
        raise ValueError("dim must be positive")
    return dim


def validate_real(name: str, value: float) -> None:
    if isinstance(value, (bool, np.bool_, np.timedelta64)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def validate_alpha(alpha: complex) -> None:
    if isinstance(alpha, (bool, np.bool_, np.timedelta64)) or not isinstance(
        alpha, (int, float, complex, np.integer, np.floating, np.complexfloating)
    ):
        raise TypeError("alpha must be a real or complex number")
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")


def validate_tol(tol: float) -> None:
    validate_real("tol", tol)
    if not 0 <= tol < 1:
        raise ValueError("tol must satisfy 0 <= tol < 1")
