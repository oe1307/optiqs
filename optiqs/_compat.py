"""Python-version-compatible aliases for annotations."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from types import NotImplementedType
    else:
        # Python 3.9 has no public name for the NotImplemented singleton type.
        from typing import Any as NotImplementedType
else:
    NotImplementedType = type(NotImplemented)
