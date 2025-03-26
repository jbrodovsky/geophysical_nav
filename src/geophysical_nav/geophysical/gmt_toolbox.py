"""
Toolbox module. Contains utility functions for accessing and manipulating geophysical map data.
"""

from enum import Enum

from numpy.typing import NDArray
from numpy import float64, int64

# import xarray as xr
from xarray import DataArray
from anglewrapper import wrap
from pygmt.datasets import (
    load_earth_free_air_anomaly,
    load_earth_magnetic_anomaly,
    load_earth_relief,
)


class MapType(Enum):
    """
    Enum class for PyGMT map names. Used to validate the map type and resolution.
    """

    RELIEF = "elevation"
    GRAVITY = "free_air_anomaly"
    MAGNETIC = "magnetic_anomaly"
    UNKNOWN = "unknown"

    def __str__(self):
        if self == MapType.RELIEF:
            return "elevation"
        elif self == MapType.GRAVITY:
            return "free_air_anomaly"
        elif self == MapType.MAGNETIC:
            return "magnetic_anomaly"
        else:
            return "unknown"

    def __repr__(self):
        return f"MapType<{str(self)}>"


class MeasurementType(Enum):
    BATHYMETRY = 0
    RELIEF = 1
    GRAVITY = 2
    MAGNETIC = 3

    def __str__(self):
        if self == MeasurementType.BATHYMETRY:
            return "BATHYMETRY"
        elif self == MeasurementType.RELIEF:
            return "RELIEF"
        elif self == MeasurementType.GRAVITY:
            return "GRAVITY"
        elif self == MeasurementType.MAGNETIC:
            return "MAGNETIC"
        else:
            return "Unknown"

    def __repr__(self):
        return f"MeasurementType<{str(self)}>"


class ReliefResolution(Enum):
    ONE_DEGREE = "01d"
    THIRTY_MINUTES = "30m"
    TWENTY_MINUTES = "20m"
    FIFTEEN_MINUTES = "15m"
    TEN_MINUTES = "10m"
    SIX_MINUTES = "06m"
    FIVE_MINUTES = "05m"
    FOUR_MINUTES = "04m"
    THREE_MINUTES = "03m"
    TWO_MINUTES = "02m"
    ONE_MINUTE = "01m"
    THIRTY_SECONDS = "30s"
    FIFTEEN_SECONDS = "15s"
    THREE_SECONDS = "03s"
    ONE_SECOND = "01s"


class GravityResolution(Enum):
    ONE_DEGREE = "01d"
    THIRTY_MINUTES = "30m"
    TWENTY_MINUTES = "20m"
    FIFTEEN_MINUTES = "15m"
    TEN_MINUTES = "10m"
    SIX_MINUTES = "06m"
    FIVE_MINUTES = "05m"
    FOUR_MINUTES = "04m"
    THREE_MINUTES = "03m"
    TWO_MINUTES = "02m"
    ONE_MINUTE = "01m"


class MagneticResolution(Enum):
    ONE_DEGREE = "01d"
    THIRTY_MINUTES = "30m"
    TWENTY_MINUTES = "20m"
    FIFTEEN_MINUTES = "15m"
    TEN_MINUTES = "10m"
    SIX_MINUTES = "06m"
    FIVE_MINUTES = "05m"
    FOUR_MINUTES = "04m"
    THREE_MINUTES = "03m"
    TWO_MINUTES = "02m"


class GeophysicalMap:
    """
    Class for storing and validating geophysical map data. Combination of a dataclass and a
    thin wrapper around xarray.DataArray. The class is used to store the map data and query
    the map data for specific points. The class is also used to validate the map type and
    resolution and to construct the map data from the GMT databases.
    """

    map_type: MeasurementType
    map_resolution: ReliefResolution | GravityResolution | MagneticResolution
    west_lon: float
    east_lon: float
    south_lat: float
    north_lat: float
    map_data: DataArray

    def __init__(
        self,
        map_type: MeasurementType,
        map_resolution: ReliefResolution | GravityResolution | MagneticResolution,
        west_lon: float,
        east_lon: float,
        south_lat: float,
        north_lat: float,
        inflate_bounds: float = 0.0,
    ):
        assert isinstance(map_type, MeasurementType), "map_type must be a MeasurementType"
        self.map_type = map_type
        assert isinstance(map_resolution, (ReliefResolution, GravityResolution, MagneticResolution)), (
            "map_resolution must be a valid ReliefResolution, GravityResolution, or MagneticResolution"
        )
        self.map_resolution = map_resolution
        west_lon = wrap.to_180(west_lon)
        east_lon = wrap.to_180(east_lon)
        # assert that the west longitude is less than the east longitude
        assert west_lon < east_lon, "West longitude must be less than east longitude."
        assert south_lat < north_lat, "South latitude must be less than north latitude."
        # Inflate the bounds to ensure that the map section is large enough to interpolate
        assert 0 <= inflate_bounds, "Inflation percentage must be greater than or equal to zero."
        if inflate_bounds > 0:
            west_lon, south_lat, east_lon, north_lat = self._inflate_bounds(
                west_lon, south_lat, east_lon, north_lat, inflate_bounds
            )
        west_lon = wrap.to_180(west_lon)
        east_lon = wrap.to_180(east_lon)
        if east_lon < west_lon:
            self.west_lon = east_lon
            self.east_lon = west_lon
        else:
            self.west_lon = west_lon
            self.east_lon = east_lon
        self.south_lat = south_lat if south_lat >= -90 else -90
        self.north_lat = north_lat if north_lat <= 90 else 90
        # Validate map type and construct GMT map name to call via grdcut
        if map_type == MeasurementType.RELIEF or map_type == MeasurementType.BATHYMETRY:
            assert isinstance(map_resolution, ReliefResolution), "map_resolution must be a ReliefResolution"
            self.map_data = load_earth_relief(
                resolution=map_resolution.value,
                region=[west_lon, east_lon, south_lat, north_lat],
            )
        elif map_type == MeasurementType.GRAVITY:
            assert isinstance(map_resolution, GravityResolution), "map_resolution must be a GravityResolution"
            self.map_data = load_earth_free_air_anomaly(
                resolution=map_resolution.value,
                region=[west_lon, east_lon, south_lat, north_lat],
            )
        elif map_type == MeasurementType.MAGNETIC:
            assert isinstance(map_resolution, MagneticResolution), "map_resolution must be a MagneticResolution"
            self.map_data = load_earth_magnetic_anomaly(
                resolution=map_resolution.value,
                region=[west_lon, east_lon, south_lat, north_lat],
            )
        else:
            raise ValueError("Map type not recognized")

    def get_map_point(
        self,
        longitudes: list[int | float | int64 | float64] | NDArray[int64 | float64] | DataArray,
        latitudes: list[int | float | int64 | float64] | NDArray[int64 | float64] | DataArray,
    ) -> NDArray[float64]:
        """
        Wrapper on DataArray.interp() to query the map and simply get the returned values.
        Converting the lists of longitudes and latitudes to xarray.DataArray objects speeds
        up the query.

        Parameters
        -----------
        :param geo_map: the map data to query.
        :type geo_map: xarray.DataArray
        :param longitudes: list-like of longitudes to query.
        :type longitudes: list-like
        :param latitudes: list-like of latitudes to query.
        :type latitudes: list-like

        :returns: numpy.ndarray

        """
        longitudes = DataArray(longitudes)
        latitudes = DataArray(latitudes)
        vals = self.map_data.interp(lon=longitudes, lat=latitudes)
        return vals.data

    def _inflate_bounds(self, min_x, min_y, max_x, max_y, inflation_percent):
        """
        Used to inflate the cropping bounds for the map section
        """

        # Calculate the width and height of the original bounds
        width = max_x - min_x
        height = max_y - min_y

        # Check if the width or height is near zero and add a small amount
        if width <= 1e-6:
            width = 0.1
        if height <= 1e-6:
            height = 0.1

        # Calculate the amount to inflate based on the percentage
        inflate_x = width * inflation_percent
        inflate_y = height * inflation_percent

        # Calculate the new minimum and maximum coordinates
        new_min_x = min_x - inflate_x
        new_min_y = min_y - inflate_y
        new_max_x = max_x + inflate_x
        new_max_y = max_y + inflate_y

        return new_min_x, new_min_y, new_max_x, new_max_y

    def __str__(self):
        return f"GeophysicalMap<{self.map_type}, {self.map_resolution}, {self.west_lon}, {self.east_lon}, {self.south_lat}, {self.north_lat}>"

    def __repr__(self):
        return f"GeophysicalMap<{self.map_type}, {self.map_resolution}, {self.west_lon}, {self.east_lon}, {self.south_lat}, {self.north_lat}>"
