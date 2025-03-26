"""
Toolbox used for interacting and processing data from the SensorLogger app and the MGD77T format.
Contains tools for processing raw data collected from the SensorLogger app and the MGD77T format
as well as tools for saving and organizing the data into a database in either a SQLite
based .db format or a folder of .csv files.
"""

import argparse
import fnmatch
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz
from haversine import Unit, haversine


##############################################################################
### Raw Data Processing ######################################################
##############################################################################
# --- Sensor Logger Processing -----------------------------------------------
def process_sensorlogger(args):
    """
    Process the raw .csv files recorded from the SensorLogger app. This function will correct
    and rectify the coordinate frame as well as rename the recorded variables.
    """
    assert os.path.exists(args.location) or os.path.isdir(args.location), (
        "Error: invalid location for input data. Please verify file path."
    )
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

    accel = pd.read_csv(f"{folder}/TotalAcceleration.csv", sep=",", header=0, index_col=0, dtype=float)
    accel = accel.rename(columns={"z": "a_z", "y": "a_y", "x": "a_x"})
    accel["a_z"] = -accel["a_z"]
    accel = accel.drop(columns="seconds_elapsed")

    gyros = pd.read_csv(f"{folder}/Gyroscope.csv", sep=",", header=0, index_col=0, dtype=float)
    gyros["y"] = -gyros["y"]
    gyros = gyros.rename(columns={"z": "w_z", "y": "w_y", "x": "w_x"})
    gyros = gyros.drop(columns="seconds_elapsed")

    imu = accel.merge(gyros, how="outer", left_index=True, right_index=True)
    imu = imu.fillna(value=pd.NA)
    imu = _convert_datetime(imu)

    magnetometer = pd.read_csv(f"{folder}/Magnetometer.csv", sep=",", header=0, index_col=0, dtype=float)
    magnetometer["z"] = -magnetometer["z"]
    magnetometer = magnetometer.rename(columns={"z": "mag_z", "y": "mag_y", "x": "mag_x"})
    magnetometer = magnetometer.drop(columns="seconds_elapsed")
    magnetometer = _convert_datetime(magnetometer)

    barometer = pd.read_csv(f"{folder}/Barometer.csv", sep=",", header=0, index_col=0, dtype=float)
    barometer = barometer.drop(columns="seconds_elapsed")
    barometer = _convert_datetime(barometer)

    gps = pd.read_csv(f"{folder}/LocationGps.csv", sep=",", header=0, index_col=0, dtype=float)
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


##############################################################################
### Private Utitilty Functions ###############################################
##############################################################################
# def _process_mgd77_dataset(folder_path: str) -> (list, list):
#     """
#     Recursively search through a given folder to find .m77t files. When found,
#     read them into memory using Pandas.
#
#     Parameters
#     ----------
#     :param folder_path: The file path to the root folder to search.
#     :type folder_path: STRING
#
#     Returns
#     -------
#     :returns: data_out: list of dataframes containing the processed data
#     :returns: names: list of names of the files
#
#     """
#     file_paths = _search_folder(folder_path, "*.m77t")
#     print("Found the following source files:")
#     print("\n".join(file_paths))
#     data_out = []
#     names = []
#     for file_path in file_paths:
#         filename = os.path.split(file_path)[-1]
#         print(f"Processing: {filename}")
#         name = filename.split(".m77t")[0]
#         # print(f"Saving as: {name}.csv")
#         data_in = pd.read_csv(
#             file_path,
#             sep="\t",
#             header=0,
#         )
#         data_out.append(m77t_to_df(data=data_in))
#         names.append(name)
#     return data_out, names
#     # data_out.to_csv(os.path.join(output_path, f"{name}.csv"))


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


def _convert_datetime(df: pd.DataFrame, timezone: str = "America/New_York") -> pd.DataFrame:
    """ """
    dates = pd.to_datetime(df.index / 1e9, unit="s").tz_localize("UTC")
    df.index = dates.tz_convert(pytz.timezone(timezone))
    df = df.resample("1s").mean()
    return df


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


# Load trackline data file
def load_trackline_data(filepath: str, filtering_window=30, filtering_period=1):
    """
    Loads and formats a post-processed NOAA trackline dataset
    """
    data = pd.read_csv(
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
    data["DT"] = pd.to_timedelta(data["DT"])

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
    data["vel_filt"] = data["vel"].rolling(window=filtering_window, min_periods=filtering_period).median()
    data["vN"] = np.cos(np.deg2rad(head)) * data["vel_filt"]
    data["vE"] = np.sin(np.deg2rad(head)) * data["vel_filt"]
    return data


#######################################################################
# Command Line Interface
def parse_args():
    """
    Command line interface argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="Dataset Processing Tool",
        description="Pre-process the raw datasets",
    )
    parser.add_argument("--mdg77", action="store_true", help="Process MGD77T datasets")
    parser.add_argument("--parse", action="store_true", help="Parse datasets mode")
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
        help="Minimum duration of a continuous dataset. Default is 60 minutes. Input " + "units are minutes.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Command line interface for processing the raw datasets collected by the Sensor Logger App or
    NOAA datasets in the mgd77t format.
    """
    # args = parse_args()

    # process_map = {
    #     "sensorlogger": process_sensorlogger,
    #     "mgd77": process_mgd77,
    #     "parser": parse_dataset_from_folder,
    # }
    # if args.mode in process_map:
    #     process_map[args.mode](args)
    # else:
    #     raise NotImplementedError(
    #         f"Parser mode type {args.mode} not recognized. Please choose from the following: "
    #         + "sensorlogger, mgd77, parser"
    #     )


if __name__ == "__main__":
    main()
