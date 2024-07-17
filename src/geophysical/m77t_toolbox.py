"""
Library for interacting with the M77T data format.
"""

import os
from datetime import timedelta
from typing import List

from haversine import haversine_vector, Unit
from numpy import zeros_like, column_stack, rad2deg, deg2rad, cos, sin, arctan2
from numpy.typing import NDArray
from numpy import float64
from pandas import DataFrame, Series, Timedelta, read_csv, to_datetime, concat
from pyins.sim import generate_imu


# --- MGD77T Processing ------------------------------------------------------
def read_m77t(filepath: str) -> DataFrame:
    """
    Read in a .m77t file and return the data as a Pandas DataFrame.
    """
    try:
        data = read_csv(filepath, delimiter="\t", header=0)
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
    # Reformate date, time, and timezone data from dataframe to propoer Python datetime
    dates = data["DATE"].astype(int)
    times = (data["TIME"].astype(float)).apply(int)
    timezones = data["TIMEZONE"].astype(int)
    timezones = timezones.apply(lambda tz: f"+{tz:02}00" if tz >= 0 else f"{tz:02}00")
    times = times.apply(lambda time_int: f"{time_int // 100:02d}{time_int % 100:02d}")
    datetimes = dates.astype(str) + times.astype(str)
    timezones.index = datetimes.index
    datetimes += timezones.astype(str)
    datetimes = to_datetime(datetimes, format="%Y%m%d%H%M%z")
    data.index = datetimes
    data = data[["LAT", "LON", "CORR_DEPTH", "MAG_TOT", "MAG_RES", "GRA_OBS", "FREEAIR"]]
    # Clean up the rest of the data frame
    # data = data.dropna(axis=1, how="all")
    # data = data.dropna(axis=0, how="any")
    # Sort the DataFrame by the index
    data = data.sort_index()
    # Remove duplicate index values
    data = data.loc[~data.index.duplicated(keep="last")]

    return data


def parse_trackline_from_file(
    filepath: str,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    data_types: List[str] = None,
) -> tuple[list:DataFrame, list:str]:
    """
    Parse a single trackline dataset csv into periods of continuous data.
    """
    data = read_m77t(filepath)
    data = m77t_to_df(data)
    # get the filename without the extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    validated_subsections, names = parse_tracklines(
        data,
        max_time=max_time,
        max_delta_t=max_delta_t,
        min_duration=min_duration,
        track_name=file_name,
        data_types=data_types,
    )
    return validated_subsections, names


def parse_tracklines(
    # db_path: str,
    data: DataFrame,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    data_types: List[str] = None,
    track_name: str = "parsed",
) -> tuple[list:DataFrame, list:str]:
    """
    Parse a trackline into periods of continuous data.
    """
    # tables = get_tables(db_path)
    parsed = []
    parsed_names = []
    data_types = validate_data_type_string(data_types)
    for type_string in data_types:
        validated_subsections = _split_and_validate_dataset(
            data,
            max_time=max_time,
            max_delta_t=max_delta_t,
            min_duration=min_duration,
            data_types=type_string,
        )
        new_names = [f"{track_name}_{type_string}_{i}" for i in range(len(validated_subsections))]
        parsed.extend(validated_subsections)
        parsed_names.extend(new_names)

    return parsed, parsed_names


def validate_data_type_string(data_types: List[str]) -> list[str]:
    """
    Checks for valid data type strings and standardizes the input.
    `data_types` should be a string or list of strings where each string is
    one of the following: "relief", "depth", "bathy", "mag", "magnetic", "grav", "gravity"

    Return behavior is dependent on the input type:
    - If a string is passed in, the function will return a string
    - If a list of lists is passed in, the function will return a list

    Parameters
    -----------
    :param data_types: the data types to validate
    :type data_types: List[str]

    :returns: List[str]
    """
    out_d_type = type(data_types)
    if out_d_type == str:
        return _validate_data_type_string(data_types)
    if out_d_type == list:
        types = []
        for data_type in data_types:
            types.append(validate_data_type_string(data_type))
        types = ["".join(dtype) for dtype in types]
        return types
    raise NotImplementedError(f"Data type {out_d_type} not supported.")


def _validate_data_type_string(data_type: str) -> str:
    """
    Checks for valid data type strings and standardizes the input
    """
    valid_string = ""
    data_type = data_type.lower()
    if data_type == "all":
        return "DGM"
    if data_type in ("relief", "depth", "bathy"):
        valid_string += "D"
    if data_type in ("grav", "gravity"):
        valid_string += "G"
    if data_type in ("mag", "magnetic"):
        valid_string += "M"
    if data_type not in (
        "relief",
        "depth",
        "bathy",
        "mag",
        "magnetic",
        "grav",
        "gravity",
    ):
        raise NotImplementedError(
            f"Data type {data_type} not recognized. Please choose from the following: relief, depth, bathy, mag,"
            + "magnetic, grav, gravity"
        )
    return valid_string


def get_parsed_data_summary(data: list[DataFrame], names: list[str]) -> DataFrame:
    """
    For each dataframe in data, calculate the following: start time, end time, duration,
    starting latitude and longitude, ending latitude and longitude, and number of data
    points.
    """
    summary = DataFrame(
        columns=[
            "num_points",
            "start_time",
            "end_time",
            "duration",
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "depth_mean",
            "depth_std",
            "depth_range",
            "grav_mean",
            "grav_std",
            "grav_range",
            "mag_mean",
            "mag_std",
            "mag_range",
        ]
    )
    for df, name in zip(data, names):
        start_time = df.index[0]
        end_time = df.index[-1]
        duration = end_time - start_time
        start_lat = df["LAT"].iloc[0]
        start_lon = df["LON"].iloc[0]
        end_lat = df["LAT"].iloc[-1]
        end_lon = df["LON"].iloc[-1]
        num_points = len(df)
        try:
            depth_mean, depth_std, depth_range = _get_measurement_statistics(df["DEPTH"])
        except KeyError:
            depth_mean = None
            depth_std = None
            depth_range = None
        try:
            grav_mean, grav_std, grav_range = _get_measurement_statistics(df["GRAV_ANOM"])
        except KeyError:
            grav_mean = None
            grav_std = None
            grav_range = None
        try:
            mag_mean, mag_std, mag_range = _get_measurement_statistics(df["MAG_RES"])
        except KeyError:
            mag_mean = None
            mag_std = None
            mag_range = None

        summary.loc[name] = [
            num_points,
            start_time,
            end_time,
            duration,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            depth_mean,
            depth_std,
            depth_range,
            grav_mean,
            grav_std,
            grav_range,
            mag_mean,
            mag_std,
            mag_range,
        ]

    return summary


def _get_measurement_statistics(measurement: Series) -> tuple:
    """
    Calculate the mean, standard deviation, and number of data points for a given measurement.
    """
    mean = measurement.mean()
    std = measurement.std()
    meas_range = measurement.max() - measurement.min()
    return mean, std, meas_range


# General  parsing
def _split_and_validate_dataset(
    data: DataFrame,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    data_types: List[str] = None,
) -> list:
    """
    Split the dataset into periods of continuous data and validate the subsections.
    data_types should be a validated string of the form "DGM" where D is depth, G is gravity,
    and M is magnetic.
    """
    data_columns = []

    if data_types is None:
        data_types = ["DGM"]

    if "D" in data_types:
        data_columns.append("CORR_DEPTH")
    if "G" in data_types:
        data_columns.append("FREEAIR")
    if "M" in data_types:
        data_columns.append("MAG_RES")

    columns_to_copy = ["LAT", "LON"]
    columns_to_copy.extend(data_columns)

    data = data[columns_to_copy].copy()

    # Drop the rows that contain N/A values for all of CorrDepth, MagTot, MagRes, GraObs, and FreeAir
    # data = data.dropna(axis=1, how="all").copy()
    data = data.dropna(subset=data_columns, how="any", axis=0).copy()
    if len(data) == 0:
        return []
    # Rename the columns to be more descriptive
    data = data.rename(columns={"CORR_DEPTH": "DEPTH", "FREEAIR": "GRAV_ANOM"})

    if not isinstance(max_time, timedelta):
        max_time = timedelta(minutes=max_time)
    if not isinstance(max_delta_t, timedelta):
        max_delta_t = timedelta(minutes=max_delta_t)
    if not isinstance(min_duration, timedelta):
        min_duration = timedelta(minutes=min_duration)

    # Split the dataset into periods of continuous data collection
    data["DT"] = data.index.to_series().diff().fillna(Timedelta(seconds=0))
    data.loc[data.index[0], "DT"] = timedelta(seconds=0)
    inds = (data["DT"] > max_time).to_list()
    subsets = find_periods(inds)
    subsections = split_dataset(data, subsets)

    # Validate the subsections
    validated_subsections = []
    for df in subsections:
        # Check that the time between each data point is less than the max delta t
        if not (df["DT"] < max_delta_t).mean():
            continue
        # Check that the subsection meets the minimum duration
        if len(df) < 3 or df.index[-1] - df.index[0] < min_duration:
            continue
        # Check that the subsection has at least 2 unique timestamps
        if len(df.index.unique()) < 2:
            continue
        df = df.drop(columns="DT")
        validated_subsections.append(df)

    return validated_subsections


def find_periods(mask) -> list:
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
    ins = _populate_imu(df)
    # Get measurement values from the data frame
    if df.columns.contains("DEPTH"):
        ins.assign(depth=df["DEPTH"])
    else:
        ins.assign(depth=zeros_like(ins.index))
    if df.columns.contains("MAG_TOT"):
        ins.assign(mag_tot=df["MAG_TOT"])
    else:
        ins.assign(mag_tot=zeros_like(ins.index))
    if df.columns.contains("MAG_RES"):
        ins.assign(mag_res=df["MAG_RES"])
    else:
        ins.assign(mag_res=zeros_like(ins.index))
    if df.columns.contains("GRA_OBS"):
        ins.assign(grav_obs=df["GRA_OBS"])
    else:
        ins.assign(grav_obs=zeros_like(ins.index))
    if df.columns.contains("GRAV_ANOM"):
        ins.assign(freeair=df["GRAV_ANOM"])
    else:
        ins.assign(freeair=zeros_like(ins.index))

    return ins


def calculate_bearing_vector(coords1: NDArray[float64], coords2: NDArray[float64]) -> NDArray[float64]:
    """
    Calculate the bearing between two sets of coordinates. Vectors are row-wise.
    """
    # lat1, lon1 = deg2rad(coords1)
    # lat2, lon2 = deg2rad(coords2)

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


def calculate_bearing(coords1: tuple[float, float], coords2: tuple[float, float]) -> float:
    """
    Calculate the bearing between two sets of coordinates.
    """
    point1: tuple[float, float] = deg2rad(coords1)
    point2: tuple[float, float] = deg2rad(coords2)

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
    timestamp: Series = data.index
    lat: NDArray[float64] = data["LAT"].to_numpy()
    lon: NDArray[float64] = data["LON"].to_numpy()
    points: NDArray[float64] = column_stack(tup=(lat, lon))
    dists: NDArray[float64] = haversine_vector(array1=points[:-1, :], array2=points[1:, :], unit=Unit.METERS)
    dt: NDArray[float64] = data.index.to_series().diff().dt.total_seconds().fillna(0).to_numpy()
    vel: NDArray[float64] = dists / dt[1:]

    lat_rad: NDArray[float64] = deg2rad(lat)
    lon_rad: NDArray[float64] = deg2rad(lon)
    points_rad: NDArray[float64] = column_stack(tup=(lat_rad, lon_rad))
    bearings: NDArray[float64] = calculate_bearing_vector(coords1=points_rad[:-1, :], coords2=points_rad[1:, :])

    lla: NDArray[float64] = column_stack((lat, lon, zeros_like(lat)))
    rph: NDArray[float64] = column_stack(tup=(zeros_like(bearings), zeros_like(bearings), bearings))

    time: NDArray[float64] = data.index.to_series().diff().dt.total_seconds().fillna(value=0).to_numpy()
    time = time.cumsum()

    out: tuple[DataFrame, DataFrame] = generate_imu(time=time[:-1], lla=lla[:-1, :], rph=rph)
    traj: DataFrame = out[0]
    imu: DataFrame = out[1]
    traj = traj.drop(columns=["VN", "VE", "VD"])
    traj = traj.assign(speed=vel)

    out_traj: DataFrame = concat(objs=[traj, imu], axis=1)
    out_traj.index = timestamp[:-1]
    return out_traj
