"""
Toolbox for processing raw data collected from the SensorLogger app and the MGD77T format
as well as tools for saving and organizing the data into a database in either a SQLite
based .db formate or a folder of .csv files.
"""

import os
import argparse
import fnmatch
from typing import List
from datetime import timedelta
import sqlite3

import pytz
import pandas as pd


##############################################################################
### Raw Data Processing ######################################################
##############################################################################
# --- Sensor Logger Processing -----------------------------------------------
def process_sensorlogger(args):
    """
    Process the raw .csv files recorded from the SensorLogger app. This function will correct
    and rectify the coordinate frame as well as rename the recorded variables.
    """
    assert os.path.exists(args.location) or os.path.isdir(
        args.location
    ), "Error: invalid location for input data. Please verify file path."
    imu, magnetic_anomaly, barometer, gps = process_sensor_logger_dataset(args.location)
    output_folder = os.path.join(args.output, "processed")
    save_sensor_logger_dataset(output_folder, imu, magnetic_anomaly, barometer, gps)


def process_sensor_logger_dataset(folder: str):
    """
    Process the raw .csv files recorded from the SensorLogger app. This function will correct
    and rectify the coordinate frame as well as rename the recorded variables.

    Parameters
    ----------
    :param folder: the filepath to the folder containing the raw data values. This folder should
    contain TotalAcceleration.csv, Gyroscope.csv, Magnetometer.csv, Barometer.csv, and
    LocationGps.csv
    :type folder: string

    :returns: Pandas dataframes corresponding to the processed and cleaned imu, magnetometer,
    barometer, and GPS data.
    """

    accel = pd.read_csv(
        f"{folder}/TotalAcceleration.csv", sep=",", header=0, index_col=0, dtype=float
    )
    accel = accel.rename(columns={"z": "a_z", "y": "a_y", "x": "a_x"})
    accel["a_z"] = -accel["a_z"]
    accel = accel.drop(columns="seconds_elapsed")

    gyros = pd.read_csv(
        f"{folder}/Gyroscope.csv", sep=",", header=0, index_col=0, dtype=float
    )
    gyros["y"] = -gyros["y"]
    gyros = gyros.rename(columns={"z": "w_z", "y": "w_y", "x": "w_x"})
    gyros = gyros.drop(columns="seconds_elapsed")

    imu = accel.merge(gyros, how="outer", left_index=True, right_index=True)
    imu = imu.fillna(value=pd.NA)
    imu = _convert_datetime(imu)

    magnetometer = pd.read_csv(
        f"{folder}/Magnetometer.csv", sep=",", header=0, index_col=0, dtype=float
    )
    magnetometer["z"] = -magnetometer["z"]
    magnetometer = magnetometer.rename(
        columns={"z": "mag_z", "y": "mag_y", "x": "mag_x"}
    )
    magnetometer = magnetometer.drop(columns="seconds_elapsed")
    magnetometer = _convert_datetime(magnetometer)

    barometer = pd.read_csv(
        f"{folder}/Barometer.csv", sep=",", header=0, index_col=0, dtype=float
    )
    barometer = barometer.drop(columns="seconds_elapsed")
    barometer = _convert_datetime(barometer)

    gps = pd.read_csv(
        f"{folder}/LocationGps.csv", sep=",", header=0, index_col=0, dtype=float
    )
    gps = gps.drop(columns="seconds_elapsed")
    gps = _convert_datetime(gps)

    return imu, magnetometer, barometer, gps


def save_sensor_logger_dataset(
    imu: pd.DataFrame,
    magnetometer: pd.DataFrame,
    barometer: pd.DataFrame,
    gps: pd.DataFrame,
    output_format: str = "csv",
    output_folder: str = "./",
) -> None:
    """
    Saves the processed sensor logger data. Data is saved to a folder.

    Parameters
    ----------
    :param imu: IMU data.
    :type imu: pandas.DataFrame
    :param magnetometer: Magnetometer data.
    :type magnetometer: pandas.DataFrame
    :param barometer: barometer data.
    :type barometer: pandas.DataFrame
    :param gps: GPS data.
    :type gps: pandas.DataFrame
    :param output_format: file extension and format for output files. Optional.
    :type output_format: string
    :output_folder: filepath and/or folder name for output. Optional.
    :type output_folder: string

    :returns: none
    """
    if output_format == "csv":
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        imu.to_csv(f"{output_folder}/imu.csv")
        magnetometer.to_csv(f"{output_folder}/magnetometer.csv")
        barometer.to_csv(f"{output_folder}/barometer.csv")
        gps.to_csv(f"{output_folder}/gps.csv")
    else:
        print("Other file formats not implemented yet.")


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
    data = data[
        ["LAT", "LON", "CORR_DEPTH", "MAG_TOT", "MAG_RES", "GRA_OBS", "FREEAIR"]
    ]
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

    if os.path.isdir(location):
        data, names = _process_mgd77_dataset(location)
    else:
        filename = location.split("\\")[-1]
        names = [filename.split(".m77t")[0] + ".csv"]
        data = pd.read_csv(location, sep="\t", header=0)
        data = m77t_to_df(data)
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

    if output_format == "db":
        conn = sqlite3.connect(os.path.join(output_location, f"{dataset_name}.db"))
        with sqlite3.connect(
            os.path.join(output_location, f"{dataset_name}.db")
        ) as conn:
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
            f"Output format {output_format} not recognized. Please choose from the "
            + "following: db, csv"
        )


##############################################################################
### Private Utitilty Functions ###############################################
##############################################################################
def _process_mgd77_dataset(folder_path: str) -> (list, list):
    """
    Recursively search through a given folder to find .m77t files. When found,
    read them into memory using Pandas.

    Parameters
    ----------
    :param folder_path: The file path to the root folder to search.
    :type folder_path: STRING

    Returns
    -------
    :returns: data_out: list of dataframes containing the processed data
    :returns: names: list of names of the files

    """
    file_paths = _search_folder(folder_path, "*.m77t")
    print("Found the following source files:")
    print("\n".join(file_paths))
    data_out = []
    names = []
    for file_path in file_paths:
        filename = os.path.split(file_path)[-1]
        print(f"Processing: {filename}")
        name = filename.split(".m77t")[0]
        # print(f"Saving as: {name}.csv")
        data_in = pd.read_csv(
            file_path,
            sep="\t",
            header=0,
        )
        data_out.append(m77t_to_df(data=data_in))
        names.append(name)
    return data_out, names
    # data_out.to_csv(os.path.join(output_path, f"{name}.csv"))


def _search_folder(folder_path: str, extension: str) -> list:
    """
    Recursively search through a given folder to find files of a given file's
    extension. File extension must be formatted as: .ext with an astericks.

    Parameters
    ----------
    :param folder_path: The file path to the root folder to search.
    :type folder_path: STRING
    :param extension: The file extension formatted as .ext to search for.
    :type extension: STRING

    Returns
    -------
    :returns: list of filepaths to the file types of interest.

    """
    new_file_paths = []
    for root, _, files in os.walk(folder_path):
        print(f"Searching: {root}")
        for filename in fnmatch.filter(files, extension):
            print(f"Adding: {os.path.join(root, filename)}")
            new_file_paths.append(os.path.join(root, filename))
    return new_file_paths


def _convert_datetime(
    df: pd.DataFrame, timezone: str = "America/New_York"
) -> pd.DataFrame:
    """ """
    dates = pd.to_datetime(df.index / 1e9, unit="s").tz_localize("UTC")
    df.index = dates.tz_convert(pytz.timezone(timezone))
    df = df.resample("1s").mean()
    return df


##############################################################################
### Dataset Parsing ##########################################################
##############################################################################
# MGD77T parsing from a folder of .csv
def parse_dataset_from_folder(args):
    """
    Recursively search through a given folder to find .csv files. When found,
    read them into memory using parse_trackline, processes them, and then save as a
    .csv to the location specified by `output_path`.
    """
    if args.format == "csv":
        file_paths = _search_folder(args.location, "*.csv")
        print("Found the following source files:")
        print("\n".join(file_paths))
        for file_path in file_paths:
            filename = os.path.split(file_path)[-1]
            print(f"Processing: {filename}")
            parse_trackline_from_file(
                file_path,
                save=True,
                output_dir=args.output,
                max_time=timedelta(minutes=args.max_time),
                max_delta_t=timedelta(minutes=args.max_delta_t),
                min_duration=timedelta(minutes=args.min_duration),
            )
        # data_out.to_csv(f"{output_path}/{name}.csv")
        # data_out.to_csv(os.path.join(args.output, f"{name}.csv"))


def parse_trackline_from_file(
    filepath: str,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    save: bool = False,
    output_dir: str = None,
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
    validated_subsections = split_and_validate_dataset(
        data,
        max_time=max_time,
        max_delta_t=max_delta_t,
        min_duration=min_duration,
    )
    names = [f"{file_name}_{i}" for i in range(len(validated_subsections))]
    # Save off the subsections to CSV files
    if save:
        if output_dir is not None and not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if output_dir is None:
            output_dir = ""
        for i, df in enumerate(validated_subsections):
            df.to_csv(os.path.join(output_dir, f"{file_name}_{i}.csv"))

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
        for data_type in data_types:
            if isinstance(data_type, str):
                type_string = validate_data_type_string(data_type)
            elif isinstance(data_type, list):
                type_string = ""
                for dt in data_type:
                    type_string += validate_data_type_string(dt)
            else:
                raise NotImplementedError(f"Data type {type(data_type)} not supported.")
            validated_subsections = split_and_validate_dataset(
                data,
                max_time=max_time,
                max_delta_t=max_delta_t,
                min_duration=min_duration,
                data_types=type_string,
            )
            new_names = [
                f"{table}_{type_string}_{i}" for i in range(len(validated_subsections))
            ]
            parsed.extend(validated_subsections)
            parsed_names.extend(new_names)

    return parsed, parsed_names


def validate_data_type_string(data_type: str) -> str:
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
            f"Data type {data_type} not recognized. Please choose from the following: "
            + "relief, depth, bathy, mag, magnetic, grav, gravity"
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
            depth_mean, depth_std, depth_range = _get_measurement_statistics(
                df["DEPTH"]
            )
        except KeyError:
            depth_mean = None
            depth_std = None
            depth_range = None
        try:
            grav_mean, grav_std, grav_range = _get_measurement_statistics(
                df["GRAV_ANOM"]
            )
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
def split_and_validate_dataset(
    data: pd.DataFrame,
    max_time: timedelta = timedelta(minutes=10),
    max_delta_t: timedelta = timedelta(minutes=2),
    min_duration: timedelta = timedelta(minutes=60),
    data_types: List[str] = None,
) -> list:
    """
    Split the dataset into periods of continuous data and validate the subsections.
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


##############################################################################
### Database Utility Functions ###############################################
##############################################################################
def get_tables(db_path: str):
    """
    Get the names of all tables in a database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    conn.close()

    # The result is a list of tuples. Convert it to a list of strings.
    tables = [table[0] for table in tables]

    return tables


def table_to_df(db_path: str, table_name: str):
    """
    Load and convert a table in a database to a pandas data frame. Thin wrapper
    around pandas.read_sql_query that specifies the default data format for
    each column.
    """
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            conn,
            index_col="TIME",
        )
        data.index = pd.to_datetime(data.index)

    return data


def df_to_table(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Write a pandas data frame to a table in a database. Thin wrapper around
    pandas.DataFrame.to_sql that specifies the default data format for each
    column.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=True,
                index_label="TIME",
            )
    except sqlite3.OperationalError as e:
        print(e)

    return None


def save_dataset(
    data: list[pd.DataFrame],
    names: list[str],
    output_location: str,
    output_format: str = "db",
    dataset_name: str = "parsed",
) -> None:
    """
    Used to save the parsed MGD77T data. Data is either saved to a folder as
    .csv or to a single .db file. Default is .db.

    Parameters
    ----------
    :param data: list of dataframes containing the parsed data
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

    if output_format == "db":
        for df, name in zip(data, names):
            df_to_table(df, os.path.join(output_location, f"{dataset_name}.db"), name)
    elif output_format == "csv":
        for i, df in enumerate(data):
            df.to_csv(os.path.join(output_location, f"{names[i]}.csv"))

    else:
        raise NotImplementedError(
            f"Output format {output_format} not recognized. Please choose from the "
            + "following: db, csv"
        )


#######################################################################
# Command Line Interface
def parse_args():
    """
    Command line interface argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="SensorLoggerProcessor",
        description="Post-process the raw datasets collected by the Sensor Logger App",
    )

    parser.add_argument(
        "--mode",
        choices=["sensorlogger", "mgd77", "parser"],
        required=True,
        help="Type of sensor recording to process. Parser is used to parse NOAA datasets "
        + "in the mgd77t format into continuous datasets.",
    )
    parser.add_argument(
        "--location",
        default="./",
        help="Path to the data. Can either be a direct file path to the .m77t file, "
        + "a folder containing such file(s), or the folder containing the raw .csvs from "
        + "the sensor logger. If a folder is given, each subfolder is searched for files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="./",
        help="Output filepath to save off processed data",
        required=False,
    )
    parser.add_argument(
        "--format",
        choices=["csv", "db"],
        default="db",
        required=False,
        help="Output format for processed data. Default is .db",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=10,
        required=False,
        help="Maximum time between data points to be considered a continuous dataset. "
        + "Default is 10 minutes. Input units are minutes.",
    )
    parser.add_argument(
        "--max_delta_t",
        type=float,
        default=2,
        required=False,
        help="Maximum time between data points to be considered a continuous dataset. "
        + "Default is 2 minutes. Input units are minutes.",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=60,
        required=False,
        help="Minimum duration of a continuous dataset. Default is 60 minutes. Input "
        + "units are minutes.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command line interface for processing the raw datasets collected by the Sensor Logger App or
    NOAA datasets in the mgd77t format.
    """
    args = parse_args()

    process_map = {
        "sensorlogger": process_sensorlogger,
        "mgd77": process_mgd77,
        "parser": parse_dataset_from_folder,
    }
    if args.mode in process_map:
        process_map[args.mode](args)
    else:
        raise NotImplementedError(
            f"Parser mode type {args.mode} not recognized. Please choose from the following: "
            + "sensorlogger, mgd77, parser"
        )


if __name__ == "__main__":
    main()
