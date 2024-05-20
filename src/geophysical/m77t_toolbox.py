"""
Library for interacting with the M77T data format.
"""

import os
from datetime import timedelta
from typing import List, Optional

from sqlalchemy import DateTime, Float, Integer, MetaData, String, ForeignKey
from numpy import zeros_like, hstack, rad2deg, deg2rad, cos, sin, arctan2
from pandas import DataFrame, Series, Timedelta, read_csv, to_datetime, concat, read_sql_table
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


def calculate_trajectory(df: DataFrame) -> DataFrame:
    """
    Simulates the closed loop trajectory information from the trackline data.
    """
    time = df.index.to_series().diff().dt.total_seconds().fillna(0)
    time = time.cumsum().values
    ll = df[["LAT", "LON"]].values
    a = zeros_like(time, dtype=float)
    a = a.reshape(-1, 1)
    lla = hstack([ll, a])
    rph = zeros_like(lla)
    h = [calculate_bearing(df[["LAT", "LON"]].iloc[i], df[["LAT", "LON"]].iloc[i + 1]) for i in range(len(df) - 1)]
    h.append(h[-1])
    rph[:, 2] = h
    traj, imu = generate_imu(time, lla, rph)
    traj[["VN", "VE", "VD", "roll", "pitch", "heading"]] = (
        traj[["VN", "VE", "VD", "roll", "pitch", "heading"]].rolling(50, min_periods=1).mean()
    )
    imu = imu.rolling(50, min_periods=1).mean()
    return concat([traj, imu], axis=1)


def calculate_bearing(coords1: tuple[float, float], coords2: tuple[float, float]) -> float:
    """
    Calculate the bearing between two sets of coordinates.
    """
    lat1, lon1 = deg2rad(coords1)
    lat2, lon2 = deg2rad(coords2)

    lon_diff = lon2 - lon1

    x = cos(lat2) * sin(lon_diff)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon_diff)

    bearing = arctan2(x, y)

    # Convert to degrees
    bearing = rad2deg(bearing)

    # Normalize to 0-360
    bearing = (bearing + 360) % 360

    return bearing


# --- Database Management ----------------------------------------------------


class DatabaseManager:
    """
    Database manager for interacting with the M77T data format.
    """

    def __init__(self, uri):
        self.engine = self._create_engine(uri)
        self.metadata = MetaData()

        # Define table schemas within SQLAlchemy's MetaData context
        # Individual trackline data collections meta data
        self.collections = Table(
            "Collections",
            self.metadata,
            Column("CollectionID", Integer, primary_key=True, autoincrement=True),
            Column("CollectionName", String(255), nullable=False),
            Column("Description", String),
        )
        # Individual trackline data
        self.raw_data = Table(
            "RawTracklines",
            self.metadata,
            Column("DataID", Integer, primary_key=True, autoincrement=True),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("Timestamp", DateTime, nullable=False),
            Column("LAT", Float),
            Column("LON", Float),
            Column("CORR_DEPTH", Float),
            Column("MAG_TOT", Float),
            Column("MAG_RES", Float),
            Column("GRA_OBS", Float),
            Column("FREEAIR", Float),
        )
        # Table containing the parsed tracklines of continuous data segments
        self.source_tracklines = Table(
            "SourceTracklines",
            self.metadata,
            Column("TrajectoryID", Integer, primary_key=True, autoincrement=True),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("StartTime", DateTime, nullable=False),
            Column("EndTime", DateTime, nullable=False),
            Column("Duration", Float),
            Column("StartLAT", Float),
            Column("StartLON", Float),
            Column("EndLAT", Float),
            Column("EndLON", Float),
            Column("NumPoints", Integer),
        )
        # Open loop (IMU integrated velocities) trajectories
        self.trajectories = Table(
            "TrajectoryData",
            self.metadata,
            Column("DataPoint", Integer, primary_key=True, autoincrement=True),
            Column("TrajectoryID", Integer, ForeignKey("SourceTracklines.TrajectoryID")),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("Timestamp", DateTime, nullable=False),
            Column("LAT", Float),
            Column("LON", Float),
            Column("VE", Float),
            Column("VN", Float),
            Column("VD", Float),
            Column("CORR_DEPTH", Float),
            Column("MAG_TOT", Float),
            Column("MAG_RES", Float),
            Column("GRA_OBS", Float),
            Column("FREEAIR", Float),
        )
        # Configurations
        self.configurations = Table(
            "Configurations",
            self.metadata,
            Column("ConfigurationID", Integer, primary_key=True, autoincrement=True),
            Column("ConfigurationName", String(255), nullable=False),
            Column("Particles", Integer, nullable=False),
            Column("VelocityNoise", Float, nullable=False),
            Column("PositionCovariance", Float, nullable=False),
            Column("AltitudeCovariance", Float, nullable=False),
            Column("VelocityCovariance", Float, nullable=False),
            Column("BathymetrySigma", Float, nullable=True),
            Column("GravitySigma", Float, nullable=True),
            Column("MagneticSigma", Float, nullable=True),
        )
        # Results
        self.results = Table(
            "ResultsSummary",
            self.metadata,
            Column("ResultID", Integer, primary_key=True, autoincrement=True),
            Column("TrajectoryID", Integer, ForeignKey("SourceTracklines.TrajectoryID")),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("ConfigurationID", Integer, ForeignKey("Configurations.ConfigurationID")),
            Column("StartTime", DateTime, nullable=False),
            Column("EndTime", DateTime, nullable=False),
            Column("Duration", Float),
            Column("StartLAT", Float),
            Column("StartLON", Float),
            Column("EndLAT", Float),
            Column("EndLON", Float),
            Column("NumRecoveries", Integer, nullable=True),
        )
        # Results trajectory data
        self.results_data = Table(
            "ResultsData",
            self.metadata,
            Column("DataPoint", Integer, primary_key=True, autoincrement=True),
            Column("ResultID", Integer, ForeignKey("ResultsSummary.ResultID")),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("ConfigurationID", Integer, ForeignKey("Configurations.ConfigurationID")),
            Column("Timestamp", DateTime, nullable=False),
            Column("PF_LAT", Float, nullable=False),
            Column("PF_LON", Float, nullable=False),
            Column("PF_DEPTH", Float, nullable=False),
            Column("PF_VE", Float, nullable=False),
            Column("PF_VN", Float, nullable=False),
            Column("PF_VD", Float, nullable=False),
            Column("RMSE", Float, nullable=False),
            Column("ERROR", Float, nullable=False),
        )
        # Recoveries data
        self.recoveries = Table(
            "Recoveries",
            self.metadata,
            Column("RecoveryID", Integer, primary_key=True, autoincrement=True),
            Column("ResultID", Integer, ForeignKey("ResultsSummary.ResultID")),
            Column("CollectionID", Integer, ForeignKey("Collections.CollectionID")),
            Column("Start", DateTime, nullable=False),
            Column("End", DateTime, nullable=False),
            Column("Duration", Float, nullable=False),
            Column("AverageError", Float, nullable=False),
            Column("MinError", Float, nullable=False),
            Column("MaxError", Float, nullable=False),
        )

    def _create_engine(self, uri):
        """Create and return an SQLAlchemy engine."""
        engine = create_engine(uri)
        return engine

    # --- Internal interface methods for each table --------------------------------
    def create_tables(self):
        """Create the tables in the database based on the defined metadata."""
        self.metadata.create_all(self.engine)

    def add_collection(self, collection_name, description=""):
        """Insert a new collection into the Collections table."""
        insert_stmt = self.collections.insert().values(CollectionName=collection_name, Description=description)
        with self.engine.connect() as conn:
            result = conn.execute(insert_stmt)
            conn.commit()
            return result.inserted_primary_key[0]

    def remove_collection(self, collection_name):
        """Remove a collection from the Collections table."""
        delete_stmt = self.collections.delete().where(self.collections.c.CollectionName == collection_name)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def add_raw_data(self, collection_id, df: DataFrame):
        """Insert raw data into the RawTracklines table."""
        df["CollectionID"] = collection_id
        df["DataID"] = df.index
        df.to_sql(name="RawTracklines", con=self.engine, if_exists="replace", index=True, index_label="Timestamp")

    def remove_raw_data(self, collection_id):
        """Remove raw data from the RawTracklines table."""
        delete_stmt = self.raw_data.delete().where(self.raw_data.c.CollectionID == collection_id)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def add_source_trackline(
        self, collection_id, start_time, end_time, duration, start_lat, start_lon, end_lat, end_lon
    ):
        """Insert a source trackline into the SourceTracklines table."""
        insert_stmt = self.source_tracklines.insert().values(
            CollectionID=collection_id,
            StartTime=start_time,
            EndTime=end_time,
            Duration=duration,
            StartLAT=start_lat,
            StartLON=start_lon,
            EndLAT=end_lat,
            EndLON=end_lon,
        )
        with self.engine.connect() as conn:
            result = conn.execute(insert_stmt)
            conn.commit()
            return result.inserted_primary_key[0]

    def remove_source_trackline(self, collection_id, trajectory_id):
        """Remove a source trackline from the SourceTracklines table."""
        delete_stmt = (
            self.source_tracklines.delete()
            .where(self.source_tracklines.c.CollectionID == collection_id)
            .where(self.source_tracklines.c.TrajectoryID == trajectory_id)
        )
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def add_trajectory_data(self, collection_id, trajectory_id, df: DataFrame):
        """Insert trajectory data into the TrajectoryData table."""
        df["CollectionID"] = collection_id
        df["TrajectoryID"] = trajectory_id
        df.to_sql(name="TrajectoryData", con=self.engine, if_exists="replace", index=False)

    def remove_trajectory_data(self, collection_id, trajectory_id):
        """Remove trajectory data from the TrajectoryData table."""
        delete_stmt = (
            self.trajectories.delete()
            .where(self.trajectories.c.CollectionID == collection_id)
            .where(self.trajectories.c.TrajectoryID == trajectory_id)
        )
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def add_result(
        self,
        collection_id,
        trajectory_id,
        configuration_id,
        start_time,
        end_time,
        duration,
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        num_recoveries=None,
    ):
        """Insert a results summary into the ResultsSummary table."""
        insert_stmt = self.results.insert().values(
            CollectionID=collection_id,
            TrajectoryID=trajectory_id,
            ConfigurationID=configuration_id,
            StartTime=start_time,
            EndTime=end_time,
            Duration=duration,
            StartLAT=start_lat,
            StartLON=start_lon,
            EndLAT=end_lat,
            EndLON=end_lon,
            NumRecoveries=num_recoveries,
        )
        with self.engine.connect() as conn:
            result = conn.execute(insert_stmt)
            conn.commit()
            return result.inserted_primary_key[0]

    def remove_result(self, result_id):
        """Remove a results summary from the ResultsSummary table."""
        delete_stmt = self.results.delete().where(self.results.c.ResultID == result_id)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def add_results_data(self, collection_id, result_id, config_id, df: DataFrame):
        """Insert results data into the ResultsData table."""
        df["CollectionID"] = collection_id
        df["ResultID"] = result_id
        df["ConfigurationID"] = config_id
        df.to_sql(name="ResultsData", con=self.engine, if_exists="replace", index=False)

    def remove_results_data(self, result_id):
        """Remove results data from the ResultsData table."""
        delete_stmt = self.results_data.delete().where(self.results_data.c.ResultID == result_id)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    # --- External interface methods --------------------------------------------
    def create_raw_data_collection(self, collection_name: str, data: DataFrame):
        """Upload raw data to the database."""
        collection_id = self.add_collection(collection_name)
        self.add_raw_data(collection_id, data)

    def delete_raw_data_collection(self, collection_id):
        """Remove raw data from the database."""
        self.remove_collection(collection_id)
        self.remove_raw_data(collection_id)

    def create_source_trajectory(self, collection_id, data: DataFrame) -> None:
        """Create a source trajectory in the database."""
        trajectory_id = self.add_source_trackline(
            collection_id,
            data.index[0],
            data.index[-1],
            (data.index[-1] - data.index[0]).total_seconds(),
            data["LAT"].iloc[0],
            data["LON"].iloc[0],
            data["LAT"].iloc[-1],
            data["LON"].iloc[-1],
        )
        self.add_trajectory_data(collection_id, trajectory_id, data)

    def delete_source_trajectory(self, collection_id, trajectory_id):
        """Remove a source trajectory from the database."""
        self.remove_source_trackline(collection_id, trajectory_id)
        self.remove_trajectory_data(collection_id, trajectory_id)

    def create_configuration(
        self,
        configuration_name,
        particles,
        velocity_noise,
        position_covariance,
        altitude_covariance,
        velocity_covariance,
        bathymetry_sigma=None,
        gravity_sigma=None,
        magnetic_sigma=None,
    ):
        """Insert a configuration into the Configurations table."""
        insert_stmt = self.configurations.insert().values(
            ConfigurationName=configuration_name,
            Particles=particles,
            VelocityNoise=velocity_noise,
            PositionCovariance=position_covariance,
            AltitudeCovariance=altitude_covariance,
            VelocityCovariance=velocity_covariance,
            BathymetrySigma=bathymetry_sigma,
            GravitySigma=gravity_sigma,
            MagneticSigma=magnetic_sigma,
        )
        with self.engine.connect() as conn:
            result = conn.execute(insert_stmt)
            conn.commit()
            return result.inserted_primary_key[0]

    def delete_configuration(self, configuration_id):
        """Remove a configuration from the Configurations table."""
        delete_stmt = self.configurations.delete().where(self.configurations.c.ConfigurationID == configuration_id)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def create_result(self, collection_id, trajectory_id, configuration_id, data: DataFrame):
        """Insert a result into the ResultsSummary table."""
        result_id = self.add_result(
            collection_id,
            trajectory_id,
            configuration_id,
            data.index[0],
            data.index[-1],
            (data.index[-1] - data.index[0]).total_seconds(),
            data["LAT"].iloc[0],
            data["LON"].iloc[0],
            data["LAT"].iloc[-1],
            data["LON"].iloc[-1],
        )
        self.add_results_data(collection_id, result_id, configuration_id, data)

    def delete_result(self, result_id):
        """Remove a result from the ResultsSummary table."""
        self.remove_result(result_id)
        self.remove_results_data(result_id)

    def create_recovery(
        self, result_id, collection_id, start_time, end_time, duration, average_error, min_error, max_error
    ):
        """Insert a recovery into the Recoveries table."""
        insert_stmt = self.recoveries.insert().values(
            ResultID=result_id,
            CollectionID=collection_id,
            Start=start_time,
            End=end_time,
            Duration=duration,
            AverageError=average_error,
            MinError=min_error,
            MaxError=max_error,
        )
        with self.engine.connect() as conn:
            result = conn.execute(insert_stmt)
            conn.commit()
            return result.inserted_primary_key[0]

    def delete_recovery(self, recovery_id):
        """Remove a recovery from the Recoveries table."""
        delete_stmt = self.recoveries.delete().where(self.recoveries.c.RecoveryID == recovery_id)
        with self.engine.connect() as conn:
            conn.execute(delete_stmt)
            conn.commit()

    def load_table_to_df(self, table_name):
        """Load a table from the database into a DataFrame."""
        return read_sql_table(table_name, self.engine)

    def load_raw_data(self, collection_id):
        """Load raw data from the database into a DataFrame."""
        return read_sql_table("RawTracklines", self.engine, index_col="Timestamp", parse_dates=True).query(
            f"CollectionID == {collection_id}"
        )

    def load_source_trackline(self, tracjectory_id):
        """Load a source trackline from the database into a DataFrame and
        select only the data corresponding to the given collection and trajectory."""
        return read_sql_table("TrajectoryData", self.engine, index_col="Timestamp", parse_dates=True).query(
            f"TrajectoryID == {tracjectory_id}"
        )

    def load_configuration(self, configuration_id):
        """Load a configuration from the database into a DataFrame."""
        return read_sql_table("Configurations", self.engine).query(f"ConfigurationID == {configuration_id}")

    def load_result(self, result_id):
        """Load a result from the database into a DataFrame."""
        return read_sql_table("ResultsData", self.engine, index_col="Timestamp", parse_dates=True).query(
            f"ResultID == {result_id}"
        )
