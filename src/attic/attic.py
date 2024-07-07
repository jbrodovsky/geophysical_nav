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


def save_mgd77_dataset(
    data: list[DataFrame],
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
