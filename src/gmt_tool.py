"""
Python module to access and save geophysical maps using GMT.
"""

import subprocess
import os

from argparse import ArgumentParser
import xarray as xr
from numpy import ndarray

from tools import wrap_to_180


def get_map_section(
    west_lon: float,
    east_lon: float,
    south_lat: float,
    north_lat: float,
    map_type: str = "relief",
    map_res: str = "02m",
    output_location: str = "./output.nc",
    save: bool = False,
) -> xr.DataArray:
    """
    Function for querying the raw map to get map segments. This is the publicly facing function
    that should be used in other modules for reading in maps from raw data. If you don't need to
    query the database and instead need to load a local map file use load_map_file() instead.
    This function will query the remote GMT databases and save a local copy of the map section to
    the location specified by output_location.

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
    :param output_location: filepath and filename to output location.
    :type output_location: string

    Returns
    -------
    :returns: xarray.DataArray

    """
    _get_map_section(west_lon, east_lon, south_lat, north_lat, map_type, map_res, output_location)
    out = load_map_file(f"{output_location}.nc")
    if not save:
        os.remove(f"{output_location}.nc")
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
    return xr.load_dataarray(filepath)


def get_map_point(geo_map: xr.DataArray, longitudes, latitudes) -> ndarray:
    """
    Wrapper on DataArray.interp() to query the map and simply get the returned values
    """
    vals = geo_map.interp(lon=longitudes, lat=latitudes)
    if longitudes.shape == latitudes.shape and longitudes.shape > (1,):
        return vals.data.diagonal()
    else:
        return vals.data


def _get_map_section(
    west_lon: float,
    east_lon: float,
    south_lat: float,
    north_lat: float,
    map_type: str = "relief",
    map_res: str = "02m",
    output_location: str = "./",
) -> None:
    """
    Function for querying the raw map source and saving it to a local file. This is the function
    that is called when running the module. The public method with the same name calls this function
    to act as the low-level interface.
    """
    west_lon = wrap_to_180(west_lon)
    east_lon = wrap_to_180(east_lon)
    # assert that the west longitude is less than the east longitude
    assert west_lon < east_lon, "West longitude must be less than east longitude."
    # Validate map type and construct GMT map name to call via grdcut
    map_name = "earth_"
    if map_type == "gravity" and _validate_gravity_resolution(map_res):
        map_name += f"faa_{map_res}"
    elif map_type == "magnetic" and _validate_magentic_resolution(map_res):
        map_name += f"mag_{map_res}"
    elif map_type == "relief" and _validate_relief_resoltion(map_res):
        map_name += f"{map_type}_{map_res}"
    else:
        print("Map type not recognized")
        return

    if map_type == "relief" and (map_res == "03s" or map_res == "01s"):
        map_name += "_g"
    else:
        map_name += "_p"

    cmd = f"gmt grdcut @{map_name} -Rd{west_lon}/{east_lon}/{south_lat}/{north_lat} " f"-G{output_location}.nc"
    print(cmd)
    out = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        shell=True,
    )
    print(out)
    return None


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
    else:
        print("Invalid resolution for map type: GRAVITY")
        return False


def _validate_magentic_resolution(res: str) -> bool:
    valid = ["01d", "30m", "20m", "15m", "10m", "06m", "05m", "04m", "03m", "02m"]
    if any(res == R for R in valid):
        return True
    else:
        print("Invalid resolution for map type: MAGNETIC")
        return False


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
    else:
        print("Invalid resolution for map type: RELIEF")
        return False


def inflate_bounds(min_x, min_y, max_x, max_y, inflation_percent):
    """
    Used to inflate the cropping bounds for the map section
    """

    # Calculate the width and height of the original bounds
    width = max_x - min_x
    height = max_y - min_y

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
    _get_map_section(
        args.west,
        args.east,
        args.south,
        args.north,
        args.type,
        args.res,
        f"{args.location}/{args.name}",
    )
    return None


if __name__ == "__main__":
    main()
