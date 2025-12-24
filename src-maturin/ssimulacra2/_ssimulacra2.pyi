"""
ssimulacra2._ssimulacra2
~~~~~~~~~~~~~~~~~~~~~~~~
A Python binding for the SSIMULACRA2 image quality assessment algorithm rust re-implementation.

:copyright: (c) 2025 noaione
:license: BSD-3-Clause, see LICENSE for details.
"""

from __future__ import annotations

__version__: str
"""Current version of fastnomicon"""

def analyze(source: list[int], degraded: list[int], width: int, height: int) -> float:
    """Analyze the given source and degraded images.

    :param source: A list of integers representing the source image pixels in RGB8 format.
    :param degraded: A list of integers representing the degraded image pixels in RGB8 format.
    :param width: The width of the images.
    :param height: The height of the images.
    :return: The SSIMULACRA2 score as a float.
    """
    ...
