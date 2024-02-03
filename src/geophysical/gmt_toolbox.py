"""
Toolbox module. Contains utility functions for accessing and manipulating geophysical map data.
"""

import os
from argparse import ArgumentParser
from datetime import timedelta

import numpy as np
import xarray as xr
from anglewrapper import wrap
from haversine import Unit, haversine
from pandas import read_csv, to_timedelta
from pygmt.datasets import (
    load_earth_free_air_anomaly,
    load_earth_magnetic_anomaly,
    load_earth_relief,
)


def get_map_section(
    west_lon: float,
    east_lon: float,
    south_lat: float,
    north_lat: float,
    map_type: str = "relief",
    map_res: str = "02m",
) -> xr.DataArray:
    """
    Function for querying the raw map to get map segments. This is the publicly facing function
    that should be used in other modules for reading in maps from raw data. If you don't need to
    query the GMT database and instead need to load a local map file use load_map_file() instead.
    This function will query the remote GMT databases and return the map data as an xarray.DataArray.

    Parameters
    ----------
    :param west_lon: West longitude value in degrees.
    :type west_lon: float
    :param east_lon: East longitude value in degrees.
    :type east_lon: float
    :param south_lat: South latitude value in degrees.
    :type south_lat: float
    :param north_lat: North latitude value in degrees.
    :type north_lat: float
    :param map_type: Geophysical map type (relief, gravity, magnetic)
    :type map_type: string
    :param map_res: map resolution of output, all maps have 01d, 30m, 20m, 15m, 10m, 06m, 05m,
    04m, 03m, and 02m; additionally gravity and relief have 01m; additionally, relief has 30s,
    15s, 03s, 01s
    :type map_res: string

    Returns
    -------
    :returns: xarray.DataArray

    """
    west_lon = wrap.to_180(west_lon)
    east_lon = wrap.to_180(east_lon)
    # assert that the west longitude is less than the east longitude
    assert west_lon < east_lon, "West longitude must be less than east longitude."
    # Validate map type and construct GMT map name to call via grdcut
    if map_type == "gravity" and _validate_gravity_resolution(map_res):
        out = load_earth_free_air_anomaly(
            resolution=map_res,
            region=[west_lon, east_lon, south_lat, north_lat],
        )
    elif map_type == "magnetic" and _validate_magentic_resolution(map_res):
        out = load_earth_magnetic_anomaly(
            resolution=map_res,
            region=[west_lon, east_lon, south_lat, north_lat],
        )
    elif map_type == "relief" and _validate_relief_resoltion(map_res):
        out = load_earth_relief(
            resolution=map_res,
            region=[west_lon, east_lon, south_lat, north_lat],
        )
    else:
        # print("Map type not recognized")
        raise ValueError("Map type not recognized")

    return out


def load_map_file(filepath: str) -> xr.DataArray:
    """
    Used to load the local .nc (netCDF4) map files in to a Python xarray DataArray structure.

    Parameters
    -----------
    :param filepath: the filepath to the map file.
    :type filepath: string

    :returns: xarray.DataArray
    """
    # Add in error handling for file not found
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        return xr.load_dataarray(filepath)
    except Exception as e:
        print(e)
        raise e


def save_map_file(map_data: xr.DataArray, filepath: str) -> None:
    """
    Used to save the map data to a local .nc (netCDF4) file.

    Parameters
    -----------
    :param map_data: the map data to save.
    :type map_data: xarray.DataArray
    :param filepath: the filepath to save the map file.
    :type filepath: string

    :returns: None
    """
    # Add some control flow to validate the inputs and file paths
    assert isinstance(map_data, xr.DataArray), "map_data must be an xarray.DataArray"
    # Check to make sure the file path is valid, if the filepath contains a directory that
    # doesn't exist, create it
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    map_data.to_netcdf(filepath)
    return None


def get_map_point(geo_map: xr.DataArray, longitudes, latitudes) -> np.ndarray:
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
    longitudes = xr.DataArray(longitudes)
    latitudes = xr.DataArray(latitudes)
    vals = geo_map.interp(lon=longitudes, lat=latitudes)
    return vals.data


def _validate_gravity_resolution(res: str) -> bool:
    valid = [
        "01d",
        "30m",
        "20m",
        "15m",
        "10m",
        "06m",
        "05m",
        "04m",
        "03m",
        "02m",
        "01m",
    ]
    if any(res == R for R in valid):
        return True
    # else:
    #     print("Invalid resolution for map type: GRAVITY")
    #     return False
    raise ValueError(f"Resolution {res} invalid for map type: GRAVITY. Valid resolutions are: {valid}")


def _validate_magentic_resolution(res: str) -> bool:
    valid = ["01d", "30m", "20m", "15m", "10m", "06m", "05m", "04m", "03m", "02m"]
    if any(res == R for R in valid):
        return True
    raise ValueError(f"Resolution {res} invalid for map type: MAGNETIC. Valid resolutions are: {valid}")


def _validate_relief_resoltion(res: str) -> bool:
    valid = [
        "1d",
        "30m",
        "20m",
        "15m",
        "10m",
        "06m",
        "05m",
        "04m",
        "03m",
        "02m",
        "01m",
        "30s",
        "15s",
        "03s",
        "01s",
    ]
    if any(res == R for R in valid):
        return True

    raise ValueError(f"Resolution {res} invalid for map type: RELIEF. Valid resolutions are: {valid}")


def inflate_bounds(min_x, min_y, max_x, max_y, inflation_percent):
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


def main() -> None:
    """
    Command line tool for accessing GMT maps.
    """

    parser = ArgumentParser(
        prog="GMT Map Access Tool",
        description="A light weight wrapper for accesssing GMT maps via Python.",
    )
    parser.add_argument(
        "--type",
        default="relief",
        choices=["relief", "gravity", "grav", "magnetic", "mag"],
        required=True,
        help="Map type to load.",
    )
    parser.add_argument(
        "--res",
        default="02m",
        required=False,
        help=(
            "Map resolution code. Available resolutions depend on the map selected.\nGravity:"
            "\t01d, 30m, 20m, 15m, 10m, 06m, 05m, 04m, 03m, 02m, 01m\nMagnetic:\t01d, 30m, 20m, "
            "15m, 10m, 06m, 05m, 04m, 03m, 02m\nRelief:\t01d, 30m, 20m, 15m, 10m, 06m, 05m, 04m, "
            "03m, 02m, 01m, 30s, 15s, 03s, 01s"
        ),
    )
    parser.add_argument("--location", default="./", required=False, help="File location to save output.")
    parser.add_argument("--name", default="map", required=False, help="Output file name.")
    # add arguements to the parser for west longitude, east longitude, south latitude,
    # and north latitude
    parser.add_argument(
        "--west",
        default=-180,
        type=float,
        required=True,
        help="West longitude in degrees +/-180.",
    )
    parser.add_argument(
        "--east",
        default=180,
        type=float,
        required=True,
        help="East longitude in degrees +/-180.",
    )
    parser.add_argument(
        "--south",
        default=-90,
        type=float,
        required=True,
        help="South latitude in degrees +/-90.",
    )
    parser.add_argument(
        "--north",
        default=90,
        type=float,
        required=True,
        help="North latitude in degrees +/-90.",
    )

    args = parser.parse_args()
    # _get_map_section(
    #     args.west,
    #     args.east,
    #     args.south,
    #     args.north,
    #     args.type,
    #     args.res,
    #     f"{args.location}/{args.name}",
    # )
    return None


if __name__ == "__main__":
    main()
