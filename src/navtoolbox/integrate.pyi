"""
Strapdown inertial navigation integration
"""
from __future__ import annotations
import numpy
__all__ = ['StrapdownIntegrator']
class StrapdownIntegrator:
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, arg0: float, arg1: float, arg2: float, arg3: float, arg4: float, arg5: float, arg6: float, arg7: float, arg8: float) -> None:
        ...
    def integrate(self, dt: float, gyro: numpy.ndarray[numpy.float64[3, 1]], accel: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
