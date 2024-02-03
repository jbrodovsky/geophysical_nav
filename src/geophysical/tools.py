"""
Utilities toolbox
"""

import math
from datetime import timedelta
import numpy as np
from pandas import read_csv, to_timedelta
from haversine import haversine, Unit


# Load trackline data file
def load_trackline_data(filepath: str, filtering_window=30, filtering_period=1):
    """
    Loads and formats a post-processed NOAA trackline dataset
    """
    data = read_csv(
        filepath,
        header=0,
        index_col=0,
        parse_dates=True,
        dtype={
            "LAT": float,
            "LON": float,
            "BAT_TTIME": float,
            "CORR_DEPTH": float,
            "MAG_TOT": float,
            "MAG_RES": float,
            "DT": str,
        },
    )
    data["DT"] = to_timedelta(data["DT"])

    dist = np.zeros_like(data.LON)
    head = np.zeros_like(data.LON)

    for i in range(1, len(data)):
        dist[i] = haversine(
            (data.iloc[i - 1]["LAT"], data.iloc[i - 1]["LON"]),
            (data.iloc[i]["LAT"], data.iloc[i]["LON"]),
            Unit.METERS,
        )
        head[i] = haversine_angle(
            (data.iloc[i - 1]["LAT"], data.iloc[i - 1]["LON"]),
            (data.iloc[i]["LAT"], data.iloc[i]["LON"]),
        )

    data["distance"] = dist
    data["heading"] = head
    data["vel"] = data["distance"] / (data["DT"] / timedelta(seconds=1))
    data["vel_filt"] = (
        data["vel"]
        .rolling(window=filtering_window, min_periods=filtering_period)
        .median()
    )
    data["vN"] = np.cos(np.deg2rad(head)) * data["vel_filt"]
    data["vE"] = np.sin(np.deg2rad(head)) * data["vel_filt"]
    return data


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
    y = np.cos(origin[0]) * np.sin(destination[0]) - np.sin(origin[0]) * np.cos(
        destination[0]
    ) * np.cos(d_lon)
    heading = np.rad2deg(np.arctan2(x, y))
    return heading


def wrap_to_pi(angle):
    """
    Wraps the given angle(s) to +/- pi.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    elif isinstance(angle, list):
        return [wrap_to_pi(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_pi(a) for a in angle)
    elif hasattr(
        angle, "__iter__"
    ):  # Check if it's an iterable (including numpy array)
        return [wrap_to_pi(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_2pi(angle):
    """
    Wraps the given angle(s) to 2 pi.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return angle % (2 * math.pi)
    elif isinstance(angle, list):
        return [wrap_to_2pi(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_2pi(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_2pi(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_180(angle):
    """
    Wraps the given angle(s) to 180.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return (angle + 180) % 360 - 180
    elif isinstance(angle, list):
        return [wrap_to_180(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_180(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_180(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")


def wrap_to_360(angle):
    """
    Wraps the given angle(s) to 360.

    Parameters
    ----------
    :param angle: The values to wrap.
    :type angle: int, float, list[int], list[float], iterable[int], iterable[float]

    Returns
    --------
    :return: The wrapped values
    :rtype: int, float, list[int], list[float], iterable[int], iterable[float]

    Raises
    -------
    :raise TypeError: If the input data type is unsupported.
    """
    if isinstance(angle, (int, float)):
        return angle % 360
    elif isinstance(angle, list):
        return [wrap_to_360(a) for a in angle]
    elif isinstance(angle, tuple):
        return tuple(wrap_to_360(a) for a in angle)
    elif hasattr(angle, "__iter__"):
        return [wrap_to_360(a) for a in angle]
    else:
        raise TypeError("Unsupported data type for angle")
