"""
ssimulacra2
~~~~~~~~~~~
A Python binding for the SSIMULACRA2 image quality assessment algorithm rust re-implementation.

:copyright: (c) 2025 noaione
:license: BSD-3-Clause, see LICENSE for details.
"""

from __future__ import annotations

from ._ssimulacra2 import (  # type: ignore
    __version__,
    analyze,
)

__all__ = ("__version__", "analyze")

__name__ = "ssimulacra2"
__package__ = "ssimulacra2"
