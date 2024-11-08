"""
Earth geometry and gravity models using WGS84 parameters
"""

from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import typing

__all__ = [
    "A",
    "E2",
    "F",
    "GE",
    "GP",
    "RATE",
    "curvature_matrix",
    "gravitation_ecef",
    "gravity",
    "gravity_n",
    "principal_radii",
    "rate_n",
    "say_hello",
]

def curvature_matrix(lat: float, alt: float) -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    Computes the Earth curvature matrix.
    """

def gravitation_ecef(
    lla: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)],
) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Computes the gravitational force vector in ECEF frame.
    """

def gravity(lat: float, alt: float) -> float:
    """
    Computes the gravity magnitude using the Somigliana model with linear altitude correction.
    """

def gravity_n(lat: float, alt: float) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Computes the gravity vector in the NED (North-East-Down) frame.
    """

def principal_radii(lat: float, alt: float) -> tuple[float, float, float]:
    """
    Computes the principal radii of curvature of Earth ellipsoid.
    """

def rate_n(lat: float) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Computes Earth rate in the NED frame.
    """

def say_hello() -> None:
    """
    A function that prints says hello.
    """

A: float = 6378137.0
E2: float = 0.0066943799901413
F: float = 0.0019318526463962815
GE: float = 9.7803253359
GP: float = 9.8321849378
RATE: float = 7.292115e-05
