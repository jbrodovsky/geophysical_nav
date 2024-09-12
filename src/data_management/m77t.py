"""
Library for interacting with the M77T data format.
"""

from typing import List, Tuple

from haversine import Unit, haversine_vector
from numpy import arctan2, column_stack, cos, deg2rad, float64, rad2deg, sin, zeros_like, asarray, nan_to_num
from numpy.typing import NDArray
from pandas import DataFrame, Series, concat, read_csv, to_datetime, Index, Timedelta
from pyins.sim import generate_imu


# --- MGD77T Processing ------------------------------------------------------
def read_m77t(filepath: str) -> DataFrame:
    """
    Read in a .m77t file and return the data as a Pandas DataFrame.
    """
    try:
        data: DataFrame = read_csv(filepath, delimiter="\t", header=0)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {filepath} not found.") from exc
    return data


def m77t_to_df(data: DataFrame) -> DataFrame:
    """
    Formats a data frame from the raw .m77t input into a more useful representation.
    The time data is foramtted to a Python `datetime` object and used as the new
    index of the DataFrame. Rows containing N/A values are dropped.

    Parameters
    -----------
    :param data: the raw input data from the .m77t read in via a Pandas DataFrame
    :type data: Pandas DataFrame

    :returns: the time indexed and down sampled data frame.
    """
    data = data.dropna(subset=["TIME"])
    # Reformat date, time, and timezone data from dataframe to propoer Python datetime
    dates: Series = data["DATE"].astype(int)
    times: Series = (data["TIME"].astype(float)).apply(int)
    timezones: Series = data["TIMEZONE"].astype(int)
    timezones = timezones.apply(lambda tz: f"+{tz:02}00" if tz >= 0 else f"{tz:02}00")
    times = times.apply(lambda time_int: f"{time_int // 100:02d}{time_int % 100:02d}")
    datetimes: Series = dates.astype(str) + times.astype(str)
    timezones.index = datetimes.index
    datetimes += timezones.astype(str)
    datetimes = to_datetime(datetimes, format="%Y%m%d%H%M%z")
    data.index = Index(datetimes)
    data = data[["LAT", "LON", "CORR_DEPTH", "MAG_TOT", "MAG_RES", "GRA_OBS", "FREEAIR"]]
    # Rename "CORR_DEPTH" to "DEPTH"
    data = data.rename(columns={"CORR_DEPTH": "DEPTH"})
    # Sort the DataFrame by the index
    data = data.sort_index()
    # Remove duplicate index values
    data = data.loc[~data.index.duplicated(keep="last")]

    return data


def find_periods(mask: list[int | bool]) -> list:
    """
    Find the start and stop indecies from a boolean mask.
    """
    # Calculate the starting and ending indices for each period
    periods = []
    start_index = None

    for idx, is_true in enumerate(mask):
        if not is_true and start_index is None:
            start_index = idx
        elif is_true and start_index is not None:
            end_index = idx - 1
            periods.append((start_index, end_index))
            start_index = None

    # If the last period extends until the end of the mask, add it
    if start_index is not None:
        end_index = len(mask) - 1
        periods.append((start_index, end_index))

    return periods


def split_dataset(df: DataFrame, periods: list) -> list:
    """
    Split a dataframe into subsections based on the given periods.
    """
    subsections = []
    for start, end in periods:
        subsection = df.iloc[start : end + 1]  # Add 1 to include the end index
        subsections.append(subsection)
    return subsections


def create_trajectory(df: DataFrame) -> DataFrame:
    """
    Simulates the closed loop trajectory information from the trackline data.
    """
    ins: DataFrame = _populate_imu(df)
    depths: NDArray[float64] = zeros_like(ins.index)
    grav: NDArray[float64] = zeros_like(ins.index)
    grav_anom: NDArray[float64] = zeros_like(ins.index)
    mags: NDArray[float64] = zeros_like(ins.index)
    mag_anom: NDArray[float64] = zeros_like(ins.index)

    # Get measurement values from the data frame
    if "DEPTH" in df.columns:
        depths = df["DEPTH"].to_numpy()
        depths = depths[:-1]
    if "MAG_TOT" in df.columns:
        mags = df["MAG_TOT"].to_numpy()
        mags = mags[:-1]
    if "MAG_RES" in df.columns:
        mag_anom = df["MAG_RES"].to_numpy()
        mag_anom = mag_anom[:-1]
    if "GRA_OBS" in df.columns:
        grav = df["GRA_OBS"].to_numpy()
        grav = grav[:-1]
    if "FREEAIR" in df.columns:
        grav_anom = df["FREEAIR"].to_numpy()
        grav_anom = grav_anom[:-1]

    ins = ins.assign(depth=depths, gra_obs=grav, freeair=grav_anom, mag_tot=mags, mag_res=mag_anom)
    # shift all column names of INS to lowercase
    ins.columns = Index(data=[col.lower() for col in ins.columns])
    return ins


def calculate_bearing_vector(coords1: NDArray[float64], coords2: NDArray[float64]) -> NDArray[float64]:
    """
    Calculate the bearing between two sets of coordinates. Vectors are row-wise.
    """
    # lat1, lon1 = deg2rad(coords1)
    # lat2, lon2 = deg2rad(coords2)
    coords1 = asarray(coords1)
    coords2 = asarray(coords2)

    lat1: NDArray[float64] = deg2rad(coords1[:, 0])
    lon1: NDArray[float64] = deg2rad(coords1[:, 1])
    lat2: NDArray[float64] = deg2rad(coords2[:, 0])
    lon2: NDArray[float64] = deg2rad(coords2[:, 1])

    lon_diff: NDArray[float64] = lon2 - lon1

    x: NDArray[float64] = cos(lat2) * sin(lon_diff)
    y: NDArray[float64] = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon_diff)

    bearing: NDArray[float64] = arctan2(x, y)

    # Convert to degrees
    bearing = rad2deg(bearing)

    # Normalize to 0-360
    bearing = (bearing + 360) % 360

    return bearing


def calculate_bearing(coords1: Tuple[float, float], coords2: Tuple[float, float]) -> float:
    """
    Calculate the bearing between two sets of coordinates.
    """
    point1: Tuple[float, float] = deg2rad(coords1)
    point2: Tuple[float, float] = deg2rad(coords2)

    lat1: float = point1[0]
    lon1: float = point1[1]
    lat2: float = point2[0]
    lon2: float = point2[1]

    lon_diff: float = lon2 - lon1

    x: float = cos(lat2) * sin(lon_diff)
    y: float = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon_diff)

    bearing: float = arctan2(x, y)

    # Convert to degrees
    bearing = rad2deg(bearing)

    # Normalize to 0-360
    bearing = (bearing + 360) % 360

    return bearing


def _populate_imu(data: DataFrame) -> DataFrame:
    """
    Populate the inertial measurement unit columns in the dataframe using the
    lat, lon, and dt columns.
    """
    timestamp: Series = data.index.to_series()
    lat: NDArray[float64] = data["LAT"].to_numpy()
    lon: NDArray[float64] = data["LON"].to_numpy()
    points: NDArray[float64] = column_stack(tup=(lat, lon))
    dists: NDArray[float64] = haversine_vector(array1=points[:-1, :], array2=points[1:, :], unit=Unit.METERS)
    dt: NDArray[float64] = data.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
    vel: Series = Series(dists / dt[1:], index=timestamp[1:])

    lat_rad: NDArray[float64] = deg2rad(lat)
    lon_rad: NDArray[float64] = deg2rad(lon)
    points_rad: NDArray[float64] = column_stack(tup=(lat_rad, lon_rad))
    bearings: Series = Series(
        calculate_bearing_vector(coords1=points_rad[:-1, :], coords2=points_rad[1:, :]), index=timestamp[1:]
    )

    # smooth the velocity and bearing data using a moving median window
    vel = vel.rolling(window=Timedelta(minutes=60), min_periods=1).mean()
    bearings = bearings.rolling(window=Timedelta(minutes=60), min_periods=1).mean()

    vel_north: NDArray[float64] = vel * cos(deg2rad(bearings))
    vel_east: NDArray[float64] = vel * sin(deg2rad(bearings))
    vel_down: NDArray[float64] = zeros_like(vel)

    vel_ned: NDArray[float64] = column_stack(tup=(vel_north, vel_east, vel_down))

    lla: NDArray[float64] = column_stack((lat, lon, zeros_like(lat)))
    rph: NDArray[float64] = column_stack(tup=(zeros_like(bearings), zeros_like(bearings), bearings))

    time: NDArray[float64] = data.index.to_series().diff().dt.total_seconds().fillna(value=0).to_numpy()
    time = time.cumsum()

    out: Tuple[DataFrame, DataFrame] = generate_imu(time=time[:-1], lla=lla[:-1, :], rph=rph, velocity_n=vel_ned)
    traj: DataFrame = out[0]
    imu: DataFrame = out[1]

    out_traj: DataFrame = concat(objs=[traj, imu], axis=1)
    out_traj.index = Index(timestamp[:-1])
    out_traj = out_traj.assign(speed=vel)
    out_traj = out_traj.drop(columns=["VN", "VE", "VD"])

    out_traj["speed"][0] = out_traj["speed"][1]

    return out_traj


def process_m77t_file(filepath: str, max_time_delta: float = 60) -> List[DataFrame]:
    """
    Process a single M77T file and return the trajectory data. This function
    provides the complete processing pipeline for converting M77T data to
    a list of continuous trajectory dataframes for each trackline. The data
    is broken up and an considered to be "non-continuous" if the time delta
    between data points is greater than the `max_time_delta` parameter.

    Parameters
    -----------
    :param filepath: the path to the M77T file
    :type filepath: str

    :param max_time_delta: the maximum time delta between data points in seconds
    :type max_time_delta: float

    :returns: List[DataFrame]
    """
    data: DataFrame = read_m77t(filepath)
    data = m77t_to_df(data)
    trajectory: DataFrame = create_trajectory(data)
    # Split the data into continuous sections
    dt: Series = trajectory.index.to_series().diff().dt.total_seconds().fillna(0)
    periods: List[Tuple[int, int]] = find_periods(mask=(dt > max_time_delta).to_list())
    continuous: List[DataFrame] = split_dataset(df=trajectory, periods=periods)
    return continuous
