"""Python-version-compatible aliases for annotations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import NotImplementedType
else:
    NotImplementedType = type(NotImplemented)
