"""
Navigation toolbox for coordinate transformations and Earth models
"""

from __future__ import annotations
from . import earth
from . import transform

__all__: tuple = ("earth", "say_hello", "transform")

def say_hello() -> None:
    """
    A function that prints says hello.
    """
