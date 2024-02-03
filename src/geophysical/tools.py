"""
Utilities toolbox
"""


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
