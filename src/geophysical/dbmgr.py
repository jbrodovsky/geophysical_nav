"""
Module used to interact with the source database. Note that this is purely for SQL operations of pre-formatted and processed data (.m77t or otherwise). 

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

    - data (table containing the true INS position and orientation, IMU output, and geophysical measurement data of the trajectories)
        - id: int (primary key, unique identifier)
        - trajectory_id: int (foreign key to the trajectories table)
        - timestamp: datetime (timestamp of the data point)
        - latitude: float (latitude of the data point)
        - longitude: float (longitude of the data point)
        - altitude: float (altitude of the data point)
        - roll: float (roll of the data point)
        - pitch: float (pitch of the data point)
        - yaw: float (yaw of the data point)
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
        - latitude: float (latitude of the data point)
        - longitude: float (longitude of the data point)
        - altitude: float (altitude of the data point)
        - roll: float (roll of the data point)
        - pitch: float (pitch of the data point)
        - yaw: float (yaw of the data point)
        - position_error_2d: float (2D position error of the data point)
        - position_error_3d: float (3D position error of the data point)
        - position_confidence_2d: float (2D position confidence of the data point)
        - position_confidence_3d: float (3D position confidence of the data point)
        - roll_error: float (roll error of the data point)
        - pitch_error: float (pitch error of the data point)
        - yaw_error: float (yaw error of the data point)

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
        - minimum_yaw_error: float (minimum yaw error of the trajectory)
        - maximum_yaw_error: float (maximum yaw error of the trajectory)
        - average_yaw_error: float (average yaw error of the trajectory)
        - drift_2d: float (total accumulated error in 2D position per distance traveled)
        - drift_3d: float (total accumulated error in 3D position per distance traveled)

"""

from datetime import datetime

from pandas import DataFrame
from sqlalchemy import Engine, ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.orm.properties import MappedColumn


class Base(DeclarativeBase):
    """Base class for all ORM classes"""


class Trajectory(Base):
    """Class representing the Trajectory table in the database"""

    __tablename__: str = "trajectories"

    id: MappedColumn[int] = mapped_column(int, primary_key=True)
    source: MappedColumn[str] = mapped_column(String)
    start: MappedColumn[datetime] = mapped_column(datetime)
    stop: MappedColumn[datetime] = mapped_column(datetime)
    duration: MappedColumn[float] = mapped_column(float)
    distance: MappedColumn[float] = mapped_column(float)
    measurements: MappedColumn[String] = mapped_column(String)
    points: MappedColumn[int] = mapped_column(int)

    def __repr__(self) -> str:
        return f"<Trajectory {self.id}>: {self.source}"


class Data(Base):
    """Class representing the Data table in the database"""

    __tablename__: str = "data"

    id: MappedColumn[int] = mapped_column(int, primary_key=True)
    trajectory_id: MappedColumn[int] = mapped_column(int, ForeignKey("trajectories.id"))
    timestamp: MappedColumn[datetime] = mapped_column(datetime)
    latitude: MappedColumn[float] = mapped_column(float)
    longitude: MappedColumn[float] = mapped_column(float)
    altitude: MappedColumn[float] = mapped_column(float)
    roll: MappedColumn[float] = mapped_column(float)
    pitch: MappedColumn[float] = mapped_column(float)
    yaw: MappedColumn[float] = mapped_column(float)
    speed: MappedColumn[float] = mapped_column(float)
    gyro_x: MappedColumn[float] = mapped_column(float)
    gyro_y: MappedColumn[float] = mapped_column(float)
    gyro_z: MappedColumn[float] = mapped_column(float)
    accel_x: MappedColumn[float] = mapped_column(float)
    accel_y: MappedColumn[float] = mapped_column(float)
    accel_z: MappedColumn[float] = mapped_column(float)
    depth: MappedColumn[float] = mapped_column(float)
    mag_tot: MappedColumn[float] = mapped_column(float)
    mag_res: MappedColumn[float] = mapped_column(float)
    gra_obs: MappedColumn[float] = mapped_column(float)
    freeair: MappedColumn[float] = mapped_column(float)

    def __repr__(self):
        return f"<Data {self.id}>: {self.timestamp}"


class Estimate(Base):
    """Class representing the Estimates table in the database"""

    __tablename__: str = "estimates"
    id: MappedColumn[int] = mapped_column(int, primary_key=True)
    trajectory_id: MappedColumn[int] = mapped_column(int, ForeignKey("trajectories.id"))
    timestamp: MappedColumn[datetime] = mapped_column(datetime)
    latitude: MappedColumn[float] = mapped_column(float)
    longitude: MappedColumn[float] = mapped_column(float)
    altitude: MappedColumn[float] = mapped_column(float)
    roll: MappedColumn[float] = mapped_column(float)
    pitch: MappedColumn[float] = mapped_column(float)
    yaw: MappedColumn[float] = mapped_column(float)
    position_error_2d: MappedColumn[float] = mapped_column(float)
    position_error_3d: MappedColumn[float] = mapped_column(float)
    position_confidence_2d: MappedColumn[float] = mapped_column(float)
    position_confidence_3d: MappedColumn[float] = mapped_column(float)
    roll_error: MappedColumn[float] = mapped_column(float)
    pitch_error: MappedColumn[float] = mapped_column(float)
    yaw_error: MappedColumn[float] = mapped_column(float)


class Result(Base):
    """Class representing the Result table in the database"""

    __tablename__: str = "results"
    id: MappedColumn[int] = mapped_column(int, primary_key=True)
    trajectory_id: MappedColumn[int] = mapped_column(int, ForeignKey("trajectories.id"))
    minimum_position_error_2d: MappedColumn[float] = mapped_column(float)
    maximum_position_error_2d: MappedColumn[float] = mapped_column(float)
    average_position_error_2d: MappedColumn[float] = mapped_column(float)
    minimum_position_error_3d: MappedColumn[float] = mapped_column(float)
    maximum_position_error_3d: MappedColumn[float] = mapped_column(float)
    average_position_error_3d: MappedColumn[float] = mapped_column(float)
    minimum_position_confidence_2d: MappedColumn[float] = mapped_column(float)
    maximum_position_confidence_2d: MappedColumn[float] = mapped_column(float)
    average_position_confidence_2d: MappedColumn[float] = mapped_column(float)
    minimum_position_confidence_3d: MappedColumn[float] = mapped_column(float)
    maximum_position_confidence_3d: MappedColumn[float] = mapped_column(float)
    average_position_confidence_3d: MappedColumn[float] = mapped_column(float)
    minimum_roll_error: MappedColumn[float] = mapped_column(float)
    maximum_roll_error: MappedColumn[float] = mapped_column(float)
    average_roll_error: MappedColumn[float] = mapped_column(float)
    minimum_pitch_error: MappedColumn[float] = mapped_column(float)
    maximum_pitch_error: MappedColumn[float] = mapped_column(float)
    average_pitch_error: MappedColumn[float] = mapped_column(float)
    minimum_yaw_error: MappedColumn[float] = mapped_column(float)
    maximum_yaw_error: MappedColumn[float] = mapped_column(float)
    average_yaw_error: MappedColumn[float] = mapped_column(float)
    drift_2d: MappedColumn[float] = mapped_column(float)
    drift_3d: MappedColumn[float] = mapped_column(float)

class DatabaseManager:
    def __init__(self, source: str) -> None:
        self.source: str = source
        self.engine: Engine = create_engine(f"sqlite:///{source}")

    def create_tables(self) -> None:
        """Create the tables in the database"""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop the tables in the database"""
        Base.metadata.drop_all(self.engine)

    def insert_trajectory(self, trajectory: DataFrame, name: str) -> int:
        """Insert a trajectory into the database"""
        