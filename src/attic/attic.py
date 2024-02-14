"""
Attic module for functions that are no longer in use but may be useful in the future.
"""

import numpy as np


def haversine_angle(origin: tuple, destination: tuple) -> float:
    """
    Computes the Haversine calcution between two (latitude, longitude) tuples to find the
    relative bearing between points.
    https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/

    Points are assumed to be (latitude, longitude) pairs in e NED degrees. Bearing angle
    is returned in degrees from North.
    """
    destination = np.deg2rad(destination)
    origin = np.deg2rad(origin)
    d_lon = destination[1] - origin[1]
    x = np.cos(destination[0]) * np.sin(d_lon)
    y = np.cos(origin[0]) * np.sin(destination[0]) - np.sin(origin[0]) * np.cos(destination[0]) * np.cos(d_lon)
    heading = np.rad2deg(np.arctan2(x, y))
    return heading


def process_mgd77(location: str) -> None:
    """
    Processes the raw .m77t file(s) from NOAA. May be a single file or a folder.
    If a folder is specified, the function will recursively search through the
    folder to find all .m77t files.

    Parameters
    ----------
    :param location: The file path to the root folder to search.
    :type location: STRING

    Returns
    -------
    :returns: data: list of dataframes containing the processed data
    :returns: names: list of names of the files
    """
    data = []
    names = []

    for root, _, files in os.walk(location):
        for file in files:
            if file.endswith(".m77t"):
                df = pd.read_csv(os.path.join(root, file), delimiter="\t", header=0)
                df = m77t_to_df(df)
                data.append(df)
                names.append(file.split(".m77t")[0])

    return data, names
