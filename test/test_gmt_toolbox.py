"""
Test the GMT Toolbox module.
"""

import unittest


from src.geophysical import gmt_toolbox as tbx


def test_map_type() -> None:
    """
    Test the MapType enum class.
    """
    relief = tbx.MapType.RELIEF
    gravity = tbx.MapType.GRAVITY
    magnetic = tbx.MapType.MAGNETIC
    unknown = tbx.MapType.UNKNOWN
    assert str(relief) == "elevation"
    assert str(gravity) == "free_air_anomaly"
    assert str(magnetic) == "magnetic_anomaly"
    assert str(unknown) == "unknown"

def test_measurement_type() -> None:
    """
    Test the MeasurementType enumeration.
    """
    assert tbx.MeasurementType.GRAVITY.value == 2
    assert tbx.MeasurementType.MAGNETIC.value == 3
    assert tbx.MeasurementType.RELIEF.value == 1
    assert tbx.MeasurementType.BATHYMETRY.value == 0
    assert str(tbx.MeasurementType.BATHYMETRY) == "BATHYMETRY"
    assert str(tbx.MeasurementType.RELIEF) == "RELIEF"
    assert str(tbx.MeasurementType.GRAVITY) == "GRAVITY"
    assert str(tbx.MeasurementType.MAGNETIC) == "MAGNETIC"


def test_relief_resolution() -> None:
    """
    Test the relief_resolution function.
    """
    assert tbx.ReliefResolution.ONE_DEGREE.value == "01d"
    assert tbx.ReliefResolution.THIRTY_MINUTES.value == "30m"
    assert tbx.ReliefResolution.TWENTY_MINUTES.value == "20m"
    assert tbx.ReliefResolution.FIFTEEN_MINUTES.value == "15m"
    assert tbx.ReliefResolution.TEN_MINUTES.value == "10m"
    assert tbx.ReliefResolution.SIX_MINUTES.value == "06m"
    assert tbx.ReliefResolution.FIVE_MINUTES.value == "05m"
    assert tbx.ReliefResolution.FOUR_MINUTES.value == "04m"
    assert tbx.ReliefResolution.THREE_MINUTES.value == "03m"
    assert tbx.ReliefResolution.TWO_MINUTES.value == "02m"
    assert tbx.ReliefResolution.ONE_MINUTE.value == "01m"
    assert tbx.ReliefResolution.THIRTY_SECONDS.value == "30s"
    assert tbx.ReliefResolution.FIFTEEN_SECONDS.value == "15s"
    assert tbx.ReliefResolution.THREE_SECONDS.value == "03s"
    assert tbx.ReliefResolution.ONE_SECOND.value == "01s"

def test_gravity_resolution() -> None:
    """
    Test the gravity resolution enum.
    """
    assert tbx.GravityResolution.ONE_DEGREE.value == "01d"
    assert tbx.GravityResolution.THIRTY_MINUTES.value == "30m"
    assert tbx.GravityResolution.TWENTY_MINUTES.value == "20m"
    assert tbx.GravityResolution.FIFTEEN_MINUTES.value == "15m"
    assert tbx.GravityResolution.TEN_MINUTES.value == "10m"
    assert tbx.GravityResolution.SIX_MINUTES.value == "06m"
    assert tbx.GravityResolution.FIVE_MINUTES.value == "05m"
    assert tbx.GravityResolution.FOUR_MINUTES.value == "04m"
    assert tbx.GravityResolution.THREE_MINUTES.value == "03m"
    assert tbx.GravityResolution.TWO_MINUTES.value == "02m"
    assert tbx.GravityResolution.ONE_MINUTE.value == "01m"

def test_magnetic_resolution() -> None:
    """
    Test the magnetic resolution enum.
    """
    assert tbx.MagneticResolution.ONE_DEGREE.value == "01d"
    assert tbx.MagneticResolution.THIRTY_MINUTES.value == "30m"
    assert tbx.MagneticResolution.TWENTY_MINUTES.value == "20m"
    assert tbx.MagneticResolution.FIFTEEN_MINUTES.value == "15m"
    assert tbx.MagneticResolution.TEN_MINUTES.value == "10m"
    assert tbx.MagneticResolution.SIX_MINUTES.value == "06m"
    assert tbx.MagneticResolution.FIVE_MINUTES.value == "05m"
    assert tbx.MagneticResolution.FOUR_MINUTES.value == "04m"
    assert tbx.MagneticResolution.THREE_MINUTES.value == "03m"
    assert tbx.MagneticResolution.TWO_MINUTES.value == "02m"


if __name__ == "__main__":
    unittest.main()
