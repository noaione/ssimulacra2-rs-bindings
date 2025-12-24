"""
ssimulacra2
~~~~~~~~~~~
A Python binding for the SSIMULACRA2 image quality assessment algorithm Zig re-implementation.

:copyright: (c) 2025 noaione
:license: BSD-3-Clause, see LICENSE for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._ssimulacra2 import (  # type: ignore
    __version__,
    analyze,
)

if TYPE_CHECKING:
    from PIL import Image  # type: ignore

__all__ = (
    "__version__",
    "analyze",
    "pil_analyze",
)

__name__ = "ssimulacra2"
__package__ = "ssimulacra2"


def pil_analyze(*, source: "Image.Image", degraded: "Image.Image") -> float:
    """Analyze the given source and degraded images from PIL Image objects.

    :param source: A PIL Image object representing the source image in RGB8/RGBA8/Luma8 format.
    :param degraded: A PIL Image object representing the degraded image in RGB8/RGBA8/Luma8 format.
    :return: The SSIMULACRA2 score as a float.
    """

    # support mode
    if source.mode not in ("RGB", "RGBA", "L", "F"):
        raise ValueError(f"Unsupported source image mode: {source.mode}")
    if degraded.mode not in ("RGB", "RGBA", "L", "F"):
        raise ValueError(f"Unsupported degraded image mode: {degraded.mode}")

    source_pixels = list(source.getdata())
    degraded_pixels = list(degraded.getdata())
    width, height = source.width, source.height
    return analyze(source=source_pixels, degraded=degraded_pixels, width=width, height=height)
