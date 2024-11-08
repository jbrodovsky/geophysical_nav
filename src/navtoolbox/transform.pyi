"""
Coordinate transformations for navigation toolbox
"""

from __future__ import annotations
import numpy
import typing

__all__ = [
    "DEG_TO_RAD",
    "RAD_TO_DEG",
    "ecef_to_lla",
    "lla_to_ecef",
    "lla_to_ned",
    "mat_en_from_ll",
    "mat_from_rph",
    "ned_to_lla",
    "say_hello",
]

def ecef_to_lla(r_e: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Converts ECEF Cartesian coordinates to latitude, longitude, and altitude.
    """

def lla_to_ecef(lla: numpy.ndarray[numpy.float64[3, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Converts latitude, longitude, and altitude to ECEF Cartesian coordinates.
    """

def lla_to_ned(
    lla: numpy.ndarray[numpy.float64[3, 1]], ref_lla: numpy.ndarray[numpy.float64[3, 1]]
) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Converts latitude, longitude, and altitude to NED Cartesian coordinates.
    """

def mat_en_from_ll(lat: float, lon: float) -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    Computes the transformation matrix from ECEF to NED frame.
    """

@typing.overload
def mat_from_rph(roll: float, pitch: float, heading: float) -> numpy.ndarray[numpy.float64[3, 3]]:
    """
    Computes the transformation matrix from body to NED frame.
    """

@typing.overload
def mat_from_rph(rph: numpy.ndarray[numpy.float64[3, 3]]) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Converts a orientation vector of roll, pitch, and heading angles to a rotation matrix.
    """

def ned_to_lla(
    ned: numpy.ndarray[numpy.float64[3, 1]], ref_lla: numpy.ndarray[numpy.float64[3, 1]]
) -> numpy.ndarray[numpy.float64[3, 1]]:
    """
    Converts NED Cartesian coordinates to latitude, longitude, and altitude.
    """

def say_hello() -> None:
    """
    A function that prints says hello.
    """

DEG_TO_RAD: float = 0.017453292519943295
RAD_TO_DEG: float = 57.29577951308232
