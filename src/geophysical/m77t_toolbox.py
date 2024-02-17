"""
Library for interacting with the M77T data format.
"""

import os
import sqlite3
from datetime import timedelta
from typing import List

import pandas as pd

from db_tools import get_tables, table_to_df


# --- MGD77T Processing ------------------------------------------------------
def m77t_to_df(data: pd.DataFrame) -> pd.DataFrame:
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
    datetimes = pd.to_datetime(datetimes, format="%Y%m%d%H%M%z")
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


def save_mgd77_dataset(
    data: list[pd.DataFrame],
    names: list[str],
    output_location: str,
    output_format: str = "db",
    dataset_name: str = "tracklines",
) -> None:
    """
    Used to save the processed MGD77T data. Data is either saved to a folder as
    .csv or to a single .db file. Default is .db.

    Parameters
    ----------
    :param data: list of dataframes containing the processed data
    :type data: list of pandas.DataFrame
    :param names: list of names of the files
    :type names: list of strings
    :param output_location: The file path to the root folder to search.
    :type output_location: STRING
    :param output_format: The format for the output (db or csv).
    :type output_format: STRING
    :param dataset_name: The name of the dataset to be saved.
    :type dataset_name: STRING

    Returns
    -------
    :returns: none
    """
    if not os.path.isdir(output_location):
        os.makedirs(output_location)

    if output_format == "db":
        conn = sqlite3.connect(os.path.join(output_location, f"{dataset_name}.db"))
        with sqlite3.connect(os.path.join(output_location, f"{dataset_name}.db")) as conn:
            for i, df in enumerate(data):
                df.to_sql(
                    names[i],
                    conn,
                    if_exists="replace",
                    index=True,
                    index_label="TIME",
                    dtype={
                        "TIME": "TIMESTAMP",
                        "LAT": "FLOAT",
                        "LON": "FLOAT",
                        "CORR_DEPTH": "FLOAT",
                        "MAG_TOT": "FLOAT",
                        "MAG_RES": "FLOAT",
                        "GRA_OBS": "FLOAT",
                        "FREEAIR": "FLOAT",
                    },
                )
    elif output_format == "csv":
        for i, df in enumerate(data):
            df.to_csv(os.path.join(output_location, names[i] + ".csv"))

    else:
        raise NotImplementedError(
            f"Output format {output_format} not recognized. Please choose from the " + "following: db, csv"
        )


##############################################################################
# Dataset Parsing ############################################################
##############################################################################
# MGD77T parsing from a folder of .csv
# def parse_dataset_from_folder(args):
#     """
#     Recursively search through a given folder to find .csv files. When found,
#     read them into memory using parse_trackline, processes them, and then save as a
#     .csv to the location specified by `output_path`.
#     """
#     if args.format == "csv":
#         file_paths = _search_folder(args.location, "*.csv")
#         print("Found the following source files:")
#         print("\n".join(file_paths))
#         for file_path in file_paths:
#             filename = os.path.split(file_path)[-1]
#             print(f"Processing: {filename}")
#             parse_trackline_from_file(
#                 file_path,
#                 save=True,
#                 output_dir=args.output,
#                 max_time=timedelta(minutes=args.max_time),
#                 max_delta_t=timedelta(minutes=args.max_delta_t),
#                 min_duration=timedelta(minutes=args.min_duration),
#             )
#         # data_out.to_csv(f"{output_path}/{name}.csv")
#         # data_out.to_csv(os.path.join(args.output, f"{name}.csv"))


def parse_trackline_from_file(
    filepath: str,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
) -> list:
    """
    Parse a single trackline dataset csv into periods of continuous data.
    """
    data = pd.read_csv(
        filepath,
        header=0,
        index_col=0,
        parse_dates=True,
        dtype={
            "LAT": float,
            "LON": float,
            "CORR_DEPTH": float,
            "MAG_TOT": float,
            "MAG_RES": float,
            "GRA_OBS": float,
            "FREEAIR": float,
        },
    )
    # get the filename without the extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    validated_subsections = _split_and_validate_dataset(
        data,
        max_time=max_time,
        max_delta_t=max_delta_t,
        min_duration=min_duration,
    )
    names = [f"{file_name}_{i}" for i in range(len(validated_subsections))]
    return validated_subsections, names


# MGD77T parsing from a database
def parse_tracklines_from_db(
    db_path: str,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    data_types: List[str] = None,
) -> tuple[list : pd.DataFrame, list:str]:
    """
    Parse a trackline database into periods of continuous data.
    """
    tables = get_tables(db_path)
    parsed = []
    parsed_names = []
    for table in tables:
        print(f"Processing: {table}")
        data = table_to_df(db_path, table)
        data_types = validate_data_type_string(data_types)
        for type_string in data_types:
            validated_subsections = _split_and_validate_dataset(
                data,
                max_time=max_time,
                max_delta_t=max_delta_t,
                min_duration=min_duration,
                data_types=type_string,
            )
            new_names = [f"{table}_{type_string}_{i}" for i in range(len(validated_subsections))]
            parsed.extend(validated_subsections)
            parsed_names.extend(new_names)

    return parsed, parsed_names


def validate_data_type_string(data_types: List[str]) -> List[str]:
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


def get_parsed_data_summary(data: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """
    For each dataframe in data, calculate the following: start time, end time, duration,
    starting latitude and longitude, ending latitude and longitude, and number of data
    points.
    """
    summary = pd.DataFrame(
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


def _get_measurement_statistics(measurement: pd.Series) -> tuple:
    """
    Calculate the mean, standard deviation, and number of data points for a given measurement.
    """
    mean = measurement.mean()
    std = measurement.std()
    meas_range = measurement.max() - measurement.min()
    return mean, std, meas_range


# General  parsing
def _split_and_validate_dataset(
    data: pd.DataFrame,
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
    data["DT"] = data.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
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


def split_dataset(df: pd.DataFrame, periods: list) -> list:
    """
    Split a dataframe into subsections based on the given periods.
    """
    subsections = []
    for start, end in periods:
        subsection = df.iloc[start : end + 1]  # Add 1 to include the end index
        subsections.append(subsection)
    return subsections


# def mgd77_to_sql(source_data_location: str, output_location: str):
#     """
#     Convert MGD77T data to a SQLite database.
#     """
#     # Check and see if the output_location directory exists
#     if not os.path.exists(output_location):
#         os.makedirs(output_location)
#
#     # Check to see if the database exists
#     if not os.path.exists(f"{output_location}/tracklines.db"):
#         tables = []
#     else:
#         tables = get_tables(f"{output_location}/tracklines.db")
#
#     for root, _, files in os.walk(source_data_location):
#         for file in files:
#             if file.endswith(".m77t"):
#                 # check to see if the file has already been processed
#                 filename = os.path.splitext(file)[0]
#                 if filename not in tables:
#                     print("Processing file: " + file)
#                     data = pd.read_csv(os.path.join(root, file), delimiter="\t", header=0)
#                     data = m77t_to_df(data)
#                     save_dataset(
#                         [data],
#                         [filename],
#                         output_location=output_location,
#                         output_format="db",
#                         dataset_name="tracklines",
#                     )
#                 else:
#                     print("Skipping file: " + file + " (already processed)")
#
