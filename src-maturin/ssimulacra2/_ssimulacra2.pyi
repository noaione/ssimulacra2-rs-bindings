"""
ssimulacra2._ssimulacra2
~~~~~~~~~~~~~~~~~~~~~~~~
A Python binding for the SSIMULACRA2 image quality assessment algorithm rust re-implementation.

:copyright: (c) 2025 noaione
:license: BSD-3-Clause, see LICENSE for details.
"""

from __future__ import annotations

from typing import TypeAlias

__version__: str
"""Current version of fastnomicon"""

FlatInt: TypeAlias = list[int]
"""The type alias for a flat list of integers representing pixel data."""
FlatFloat: TypeAlias = list[float]
"""The type alias for a flat list of floats representing pixel data."""
RGBInt: TypeAlias = list[tuple[int, int, int]]
"""The type alias for a list of RGB tuples with integer components."""
RGBFloat: TypeAlias = list[tuple[float, float, float]]
"""The type alias for a list of RGB tuples with float components."""
RGBAInt: TypeAlias = list[tuple[int, int, int, int]]
"""The type alias for a list of RGBA tuples with integer components."""
RGBAFloat: TypeAlias = list[tuple[float, float, float, float]]
"""The type alias for a list of RGBA tuples with float components."""
LumaInt: TypeAlias = list[tuple[int]]
"""The type alias for a list of Luma tuples with integer components."""
LumaFloat: TypeAlias = list[tuple[float]]
"""The type alias for a list of Luma tuples with float components."""

InputPixels: TypeAlias = (
    FlatInt | FlatFloat | RGBInt | RGBFloat | RGBAInt | RGBAFloat | LumaInt | LumaFloat
)
"""The type alias for all supported input pixel formats."""

def analyze(source: InputPixels, degraded: InputPixels, width: int, height: int) -> float:
    """Analyze the given source and degraded images.

    :param source: A list of numbers representing the source image pixels in RGB8/RGBA8/Luma8 format.
    :param degraded: A list of numbers representing the degraded image pixels in RGB8/RGBA8/Luma8 format.
    :param width: The width of the images.
    :param height: The height of the images.
    :return: The SSIMULACRA2 score as a float.
    """
    ...
