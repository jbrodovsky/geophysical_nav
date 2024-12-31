"""
Module used to interact with the source database. Note that this is purely for SQL operations
of pre-formatted and processed data (.m77t or otherwise).

Data should be stored in a database with the following schema:
    - trajectories (summary table containing the high level information of the trajectories)
        - id: int (primary key, unique identifier)
        - source: str (name of the source file)
        - start: datetime of the start of the trajectory
        - stop: datetime of the end of the trajectory
        - duration: float (duration of the trajectory in seconds)
        - distance: float (distance of the trajectory in meters)
        - measurements: str (available measurements in the trajectory)
        - points: int (number of points in the trajectory)

    - data (table containing the true INS position and orientation, IMU output, and
    geophysical measurement data of the trajectories)
        - id: int (primary key, unique identifier)
        - trajectory_id: int (foreign key to the trajectories table)
        - timestamp: datetime (timestamp of the data point)
        - lat: float (lat of the data point)
        - lon: float (lon of the data point)
        - altitude: float (altitude of the data point)
        - VN: float (velocity north of the data point)
        - VE: float (velocity east of the data point)
        - VD: float (velocity down of the data point)
        - roll: float (roll of the data point)
        - pitch: float (pitch of the data point)
        - heading: float (heading of the data point)
        - gyro_x: float (gyroscope x of the data point)
        - gyro_y: float (gyroscope y of the data point)
        - gyro_z: float (gyroscope z of the data point)
        - accel_x: float (accelerometer x of the data point)
        - accel_y: float (accelerometer y of the data point)
        - accel_z: float (accelerometer z of the data point)
        - depth: float (depth measurement of the data point)
        - mag_tot: float (total magnetic field measurement of the data point)
        - mag_res: float (residual magnetic field measurement of the data point)
        - gra_obs: float (observed gravity measurement of the data point)
        - freeair: float (free air gravity anomaly measurement of the data point)
"""

import argparse
import os
from datetime import datetime
from typing import Any

import h5py
from haversine import Unit, haversine_vector
from numpy import column_stack, nan_to_num, ndarray
from pandas import DataFrame, read_hdf, read_sql
from sqlalchemy import (
    Boolean,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.orm.query import Query
from tqdm import tqdm

from data_management.m77t import process_m77t_file


class Base(DeclarativeBase):
    """Base class for all ORM classes"""


class Trajectory(Base):
    """Class representing the Trajectory table in the database"""

    __tablename__: str = "trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    start: Mapped[datetime] = mapped_column(DateTime)
    stop: Mapped[datetime] = mapped_column(DateTime)
    duration: Mapped[float] = mapped_column(Float)
    distance: Mapped[float] = mapped_column(Float)
    depth: Mapped[bool] = mapped_column(Boolean)
    mag_tot: Mapped[bool] = mapped_column(Boolean)
    mag_res: Mapped[bool] = mapped_column(Boolean)
    gra_obs: Mapped[bool] = mapped_column(Boolean)
    freeair: Mapped[bool] = mapped_column(Boolean)
    points: Mapped[int] = mapped_column(Integer)

    def to_table_row(self) -> dict[str, Any]:
        """Converts the Trajectory object to a table row for use with a Pandas DataFrame"""
        return {
            "ID": self.id,
            "Source": self.source,
            "Start": self.start,
            "Stop": self.stop,
            "Duration": self.duration,
            "Distance": self.distance,
            "Depth": self.depth,
            "Mag Tot": self.mag_tot,
            "Mag Res": self.mag_res,
            "Gra Obs": self.gra_obs,
            "Freeair": self.freeair,
            "Points": self.points,
        }

    def __repr__(self) -> str:
        return f"<Trajectory {self.id}>: {self.source}"


class Data(Base):
    """Class representing the Data table in the database"""

    __tablename__: str = "data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trajectory_id: Mapped[int] = mapped_column(Integer, ForeignKey("trajectories.id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    alt: Mapped[float] = mapped_column(Float)
    vn: Mapped[float] = mapped_column(Float)
    ve: Mapped[float] = mapped_column(Float)
    vd: Mapped[float] = mapped_column(Float)
    roll: Mapped[float] = mapped_column(Float)
    pitch: Mapped[float] = mapped_column(Float)
    heading: Mapped[float] = mapped_column(Float)
    # speed: Mapped[float] = mapped_column(Float)
    gyro_x: Mapped[float] = mapped_column(Float)
    gyro_y: Mapped[float] = mapped_column(Float)
    gyro_z: Mapped[float] = mapped_column(Float)
    accel_x: Mapped[float] = mapped_column(Float)
    accel_y: Mapped[float] = mapped_column(Float)
    accel_z: Mapped[float] = mapped_column(Float)
    distance: Mapped[float] = mapped_column(Float)
    depth: Mapped[float] = mapped_column(Float, nullable=True)
    mag_tot: Mapped[float] = mapped_column(Float, nullable=True)
    mag_res: Mapped[float] = mapped_column(Float, nullable=True)
    gra_obs: Mapped[float] = mapped_column(Float, nullable=True)
    freeair: Mapped[float] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<Data {self.id}>: {self.timestamp}"


class DatabaseManager:
    """
    class to manage the sqlite database of source trajectory data
    """

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.engine: Engine = create_engine(url=f"sqlite:///{source}")
        self.create_tables()

    def create_tables(self) -> None:
        """Create the tables in the database"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop the tables in the database"""
        Base.metadata.drop_all(self.engine)

    def get_all_tables(self) -> list[str]:
        """Get the names of the tables in the database"""
        return Base.metadata.tables.keys()

    def get_table(self, table_name: str) -> DataFrame:
        """Get the contents of a table in the database"""
        with Session(bind=self.engine) as session:
            query: Query = session.query(Base.metadata.tables[table_name])
            return read_sql(sql=query.statement, con=self.engine)

    def insert_trajectory(self, trajectory: DataFrame, name: str) -> int:
        """Insert a trajectory into the database"""
        distances: ndarray[float] = haversine_vector(
            array1=column_stack(tup=(trajectory["lat"], trajectory["lon"])),
            array2=column_stack(
                tup=(
                    trajectory["lat"].shift(periods=1),
                    trajectory["lon"].shift(periods=1),
                )
            ),
            unit=Unit.METERS,
        )

        # replace nan values in distances with zero
        distances = nan_to_num(x=distances)

        # Check db/sources if the measurement columns are present in the DataFrame by validating that over half of the
        # rows are not nan
        has_depth = False
        has_mag_tot = False
        has_mag_res = False
        has_gra_obs = False
        has_freeair = False

        if trajectory["depth"].count() > len(trajectory) / 2:
            has_depth = True
        if trajectory["mag_tot"].count() > len(trajectory) / 2:
            has_mag_tot = True
        if trajectory["mag_res"].count() > len(trajectory) / 2:
            has_mag_res = True
        if trajectory["gra_obs"].count() > len(trajectory) / 2:
            has_gra_obs = True
        if trajectory["freeair"].count() > len(trajectory) / 2:
            has_freeair = True

        with Session(bind=self.engine) as session:
            trajectory_entry = Trajectory(
                source=name,
                start=trajectory.index.min(),
                stop=trajectory.index.max(),
                duration=(trajectory.index.max() - trajectory.index.min()).seconds,
                distance=distances.sum(),
                depth=has_depth,
                mag_tot=has_mag_tot,
                mag_res=has_mag_res,
                gra_obs=has_gra_obs,
                freeair=has_freeair,
                points=len(trajectory),
            )
            session.add(instance=trajectory_entry)
            session.commit()
            trajectory_id: int = trajectory_entry.id

            print(f"Inserting trajectory with ID: {trajectory_id}")
            for _, row in tqdm(trajectory.iterrows()):
                data_entry = Data(
                    trajectory_id=trajectory_id,
                    timestamp=row.name,
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["alt"],
                    vn=row["VN"],
                    ve=row["VE"],
                    vd=row["VE"],
                    roll=row["roll"],
                    pitch=row["pitch"],
                    heading=row["heading"],
                    gyro_x=row["gyro_x"],
                    gyro_y=row["gyro_y"],
                    gyro_z=row["gyro_z"],
                    accel_x=row["accel_x"],
                    accel_y=row["accel_y"],
                    accel_z=row["accel_z"],
                    distance=row["distance"],
                    depth=row["depth"],
                    mag_tot=row["mag_tot"],
                    mag_res=row["mag_res"],
                    gra_obs=row["gra_obs"],
                    freeair=row["freeair"],
                )
                session.add(instance=data_entry)
                session.commit()

        return trajectory_id

    def get_trajectory(self, trajectory_id: int) -> DataFrame:
        """Get a trajectory from the database"""
        with Session(bind=self.engine) as session:
            query: Query[Data] = session.query(Data).filter(Data.trajectory_id == trajectory_id)
            return read_sql(sql=query.statement, con=self.engine)

    def get_all_trajectories(self) -> DataFrame:
        """Get all trajectories from the database."""
        with Session(bind=self.engine) as session:
            query: Query[Trajectory] = session.query(Trajectory)
            return read_sql(sql=query.statement, con=self.engine)


def write_results_to_file(filename: str, configuration: dict, summary: DataFrame, results: list[DataFrame]) -> None:
    """Writes the results of a simulation to a hdf5 file"""

    # Check if the filepath specified by filename exists
    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])
    # Check if .hdf5 extension is present in the filename
    if filename.split(".")[-1] != "hdf5":
        filename = f"{filename}.hdf5"
    # Create an HDF5 file
    with h5py.File(name=filename, mode="w") as f:
        # Store the dictionary as attributes of a group
        config_group: h5py.Group = f.create_group(name="config")
        for key, value in configuration.items():
            config_group.attrs[key] = value

        # Store each DataFrame in the list of DataFrames in separate groups
        f.create_group(name="results")

    for i, df in enumerate(iterable=results):
        df.to_hdf(path_or_buf=filename, key=f"results/result_{i}", mode="a")

        # Store the main results DataFrame
        summary.to_hdf(path_or_buf=filename, key="summary", mode="a")


def read_results_file(filename: str) -> tuple[dict, DataFrame, list[DataFrame]]:
    """Reads the results of a simulation from a hdf5 file"""

    # Open the HDF5 file
    with h5py.File(name=filename, mode="r") as f:
        # Read the configuration from the attributes of the group
        config: dict = dict(f["config"].attrs.items())

        # Read the main results DataFrame
        summary: DataFrame = read_hdf(path_or_buf=f.filename, key="summary")

        # Read each DataFrame in the list of DataFrames from separate groups
        results: list[DataFrame] = []
        for i in range(len(f["results"])):
            results.append(read_hdf(path_or_buf=f.filename, key=f"results/result_{i}"))

        return config, summary, results


def main() -> None:
    """
    Command line tool interface for the database manager. Finds, reads, parses, and stores .m77t files into a
    SQLite database.
    Takes three arguments:
        1. Source file path; should either be a .m77t file or a folder containing .m77t file(s).
        2. Output file path to where the database file will be saved.
        3. Time interval to parse for continuous data collections should be a number in seconds.
    """

    parser = argparse.ArgumentParser(
        prog="Database Manager",
        description="This is a tool to convert .m77t files into continuous source INS trajectories and measurements.",
    )
    parser.add_argument("--source", type=str, help="Source file path", required=True)
    parser.add_argument("--output", type=str, help="Output file path", required=True)
    parser.add_argument(
        "--interval",
        type=int,
        help="Time interval in seconds to parse for continuous data collections",
        required=True,
    )

    args: argparse.Namespace = parser.parse_args()

    print("Confirming parameters:")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Interval: {args.interval}")
    # Confirm the parameters: press y to continue
    if input("Continue? (y/n): ") != "y":
        return
    # Parse the filepath and filename from args.output
    filepath: str = os.path.split(args.output)[0]
    filename: str = os.path.split(args.output)[1]

    # If folder does not exist, create it
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    dbmgr = DatabaseManager(source=os.path.join(filepath, filename))

    # Get a list of .m77t files as specified by the source path
    # (names:list[str], trajectories: list[DataFrame])
    data: tuple[list[str], list[DataFrame]] = _get_m77t_files(source=args.source, interval=args.interval)
    names: list[str] = data[0]
    trajectories: list[DataFrame] = data[1]
    # data = zip(names, trajectories)
    i: int = -1
    if len(names) == 1:
        print(f"Inserting trajectories from {names[0]} into database")
        for trajectory in tqdm(trajectories[0]):
            i = dbmgr.insert_trajectory(trajectory=trajectory, name=names[0])
            # print(f"Inserted trajectory with ID: {i}")

    else:
        for i, name in enumerate(names):
            print(f"Inserting trajectories from {name} into database")
            for traj in tqdm(trajectories[i]):
                i = dbmgr.insert_trajectory(trajectory=traj, name=name)
                # print(f"Inserted trajectory with ID: {i}")


def _get_m77t_files(source: str, interval: int) -> tuple[list[str], list[DataFrame]]:
    # If the source is a folder, get all .m77t files in the folder
    names: list[str] = []
    trajectories: list[DataFrame] = []
    if os.path.isdir(source):
        for root, dir, files in os.walk(source):
            for file in files:
                if file.endswith(".m77t"):
                    print(f"Found: {file} at {os.path.join(root, file)}")
                    # dbmgr.add_data(os.path.join(root, file))
                    names.append(file.split(".")[0])
                    parsed: list[DataFrame] = process_m77t_file(
                        filepath=os.path.join(root, file), max_time_delta=interval
                    )
                    trajectories.append(parsed)
    else:
        # get file name from the filepath
        assert source.split(".")[-1] == "m77t", "Source file must be a .m77t file"
        rootfile: tuple[str, str] = os.path.split(source)
        filepath: str = rootfile[0]
        filename: str = rootfile[1]
        print(f"Found: {filename} at {os.path.join(filepath, filename)}")
        names.append(os.path.split(source)[1].split(".")[0])
        trajectories.append(process_m77t_file(filepath=source, max_time_delta=interval))
    return names, trajectories


if __name__ == "__main__":
    main()
