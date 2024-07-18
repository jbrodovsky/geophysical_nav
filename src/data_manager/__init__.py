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
        - roll: float (roll of the data point)
        - pitch: float (pitch of the data point)
        - heading: float (heading of the data point)
        - speed: float (speed of the data point)
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

    - estimates (table containing the estimated state values from simulating the trajectories)
        - id: int (primary key, unique identifier)
        - trajectory_id: int (foreign key to the trajectories table)
        - timestamp: datetime (timestamp of the data point)
        - lat: float (lat of the data point)
        - lon: float (lon of the data point)
        - altitude: float (altitude of the data point)
        - roll: float (roll of the data point)
        - pitch: float (pitch of the data point)
        - heading: float (heading of the data point)
        - position_error_2d: float (2D position error of the data point)
        - position_error_3d: float (3D position error of the data point)
        - position_confidence_2d: float (2D position confidence of the data point)
        - position_confidence_3d: float (3D position confidence of the data point)
        - roll_error: float (roll error of the data point)
        - pitch_error: float (pitch error of the data point)
        - heading_error: float (heading error of the data point)

    - Result (summary table containing the results of the trajectory simulation)
        - id: int (primary key, unique identifier)
        - trajectory_id: int (foreign key to the trajectories table)
        - minimum_position_error_2d: float (minimum 2D position error of the trajectory)
        - maximum_position_error_2d: float (maximum 2D position error of the trajectory)
        - average_position_error_2d: float (average 2D position error of the trajectory)
        - minimum_position_error_3d: float (minimum 3D position error of the trajectory)
        - maximum_position_error_3d: float (maximum 3D position error of the trajectory)
        - average_position_error_3d: float (average 3D position error of the trajectory)
        - minimum_position_confidence_2d: float (minimum 2D position confidence of the trajectory)
        - maximum_position_confidence_2d: float (maximum 2D position confidence of the trajectory)
        - average_position_confidence_2d: float (average 2D position confidence of the trajectory)
        - minimum_position_confidence_3d: float (minimum 3D position confidence of the trajectory)
        - maximum_position_confidence_3d: float (maximum 3D position confidence of the trajectory)
        - average_position_confidence_3d: float (average 3D position confidence of the trajectory)
        - minimum_roll_error: float (minimum roll error of the trajectory)
        - maximum_roll_error: float (maximum roll error of the trajectory)
        - average_roll_error: float (average roll error of the trajectory)
        - minimum_pitch_error: float (minimum pitch error of the trajectory)
        - maximum_pitch_error: float (maximum pitch error of the trajectory)
        - average_pitch_error: float (average pitch error of the trajectory)
        - minimum_heading_error: float (minimum heading error of the trajectory)
        - maximum_heading_error: float (maximum heading error of the trajectory)
        - average_heading_error: float (average heading error of the trajectory)
        - drift_2d: float (total accumulated error in 2D position per distance traveled)
        - drift_3d: float (total accumulated error in 3D position per distance traveled)

"""

from datetime import datetime
from typing import List

from haversine import haversine_vector, Unit
from numpy import column_stack, ndarray, nan_to_num
from pandas import DataFrame, read_sql
from sqlalchemy import Boolean, Engine, Float, ForeignKey, Integer, String, create_engine, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.orm.query import Query


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

    def to_table_row(self):
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
    roll: Mapped[float] = mapped_column(Float)
    pitch: Mapped[float] = mapped_column(Float)
    heading: Mapped[float] = mapped_column(Float)
    speed: Mapped[float] = mapped_column(Float)
    gyro_x: Mapped[float] = mapped_column(Float)
    gyro_y: Mapped[float] = mapped_column(Float)
    gyro_z: Mapped[float] = mapped_column(Float)
    accel_x: Mapped[float] = mapped_column(Float)
    accel_y: Mapped[float] = mapped_column(Float)
    accel_z: Mapped[float] = mapped_column(Float)
    depth: Mapped[float] = mapped_column(Float, nullable=True)
    mag_tot: Mapped[float] = mapped_column(Float, nullable=True)
    mag_res: Mapped[float] = mapped_column(Float, nullable=True)
    gra_obs: Mapped[float] = mapped_column(Float, nullable=True)
    freeair: Mapped[float] = mapped_column(Float, nullable=True)

    def __repr__(self):
        return f"<Data {self.id}>: {self.timestamp}"


class DatabaseManager:
    """
    class to manage the sqlite database of source trajectory data
    """

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.engine: Engine = create_engine(f"sqlite:///{source}")
        self.create_tables()

    def create_tables(self) -> None:
        """Create the tables in the database"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop the tables in the database"""
        Base.metadata.drop_all(self.engine)

    def insert_trajectory(self, trajectory: DataFrame, name: str) -> int:
        """Insert a trajectory into the database"""
        distances: ndarray[float] = haversine_vector(
            array1=column_stack(tup=[trajectory["lat"], trajectory["lon"]]),
            array2=column_stack(tup=[trajectory["lat"].shift(periods=1), trajectory["lon"].shift(periods=1)]),
            unit=Unit.METERS,
        )

        # replace nan values in distances with zero
        distances = nan_to_num(x=distances)

        # Check if the measurement columns are present in the DataFrame by validating that over half of the rows are
        # not nan
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
            trajectory_id = trajectory_entry.id

            for _, row in trajectory.iterrows():
                data_entry = Data(
                    trajectory_id=trajectory_id,
                    timestamp=row.name,
                    lat=row["lat"],
                    lon=row["lon"],
                    alt=row["alt"],
                    roll=row["roll"],
                    pitch=row["pitch"],
                    heading=row["heading"],
                    speed=row["speed"],
                    gyro_x=row["gyro_x"],
                    gyro_y=row["gyro_y"],
                    gyro_z=row["gyro_z"],
                    accel_x=row["accel_x"],
                    accel_y=row["accel_y"],
                    accel_z=row["accel_z"],
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
            query: Query[Data] = session.query(_entity=Data).filter(Data.trajectory_id == trajectory_id)
            return read_sql(sql=query.statement, con=self.engine)

    def get_all_trajectories(self) -> DataFrame:
        """Get all trajectories from the database."""
        # Read the trajectory table from the database and return it as a DataFrame
        with Session(bind=self.engine) as session:
            query: Query[Trajectory] = session.query(_entity=Trajectory)
            return read_sql(sql=query.statement, con=self.engine)
