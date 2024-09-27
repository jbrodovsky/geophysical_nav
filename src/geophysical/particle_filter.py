"""
Particle filter map-matching algorithm and simulation code.

The particle filter approach to map-matching is to use output from the IMU or INS to propagate a set of particles
with a reference map. Measurements are taken and compared to the map to update the weights of the particles. The
particles are then resampled to generate a new set of particles. This process is repeated for each time step.

There are two ways to formulate the particle filter for map-matching. The first is a simple six-state model that
only tracks position and velocity, and propagates based on linear accelerations from the IMU post coning and sculling
correction. This method needs to track the previous IMU output, but only for the correction calculation. The second
is a full strapdown INS model that attempts to effectively monitor the position, velocity, and attitude of the
vehicle as well as the gyro and accel biases.

While this module recreates some of the functionality present in PyINS, it is not a direct copy. The particle filter
fundamentally differs from the Kalman filter INS used in PyINS where numba is used to rapidly iterate through a
DataFrame's rows by converting it to a numpy array. The particle filter is a Monte Carlo method that requires
that we maintain a set of particles and weights. As such we need to maintain a large number of particles in a
numpy array as well as iterate through the data. As such, the function found in this module are designed to be use
with numpy arrays and not just generic Python data structures.

The primary functions needing jitting are the propagate and update functions. The propagate function is used to
"""

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
import json

# import numpy as np
from filterpy.monte_carlo import residual_resample
from haversine import Unit, haversine, haversine_vector
from matplotlib import pyplot as plt
from numba import njit
from numpy import (
    abs,
    array,
    asarray,
    append,
    column_stack,
    cos,
    cross,
    deg2rad,
    diag,
    empty,
    eye,
    float64,
    int64,
    isnan,
    mean,
    nan,
    rad2deg,
    sin,
    sum,
    sqrt,
    tan,
    tile,
    zeros,
    zeros_like,
    ones_like,
    ones,
    any,
)
from numpy.random import multivariate_normal as mvn
from numpy.typing import NDArray
from pandas import DataFrame
from pyins import earth, transform
from scipy.stats import norm
from xarray import DataArray

from ..data_management.m77t import find_periods
from .gmt_toolbox import get_map_point, get_map_section, inflate_bounds

OVERFLOW = 500
EARTH_RATE = earth.RATE

class MeasurementType(Enum):
    BATHYMETRY = 0
    RELIEF = 1
    GRAVITY = 2
    MAGNETIC = 3

    def __str__(self):
        if self == MeasurementType.BATHYMETRY:
            return "BATHYMETRY"
        elif self == MeasurementType.RELIEF:
            return "RELIEF"
        elif self == MeasurementType.GRAVITY:
            return "GRAVITY"
        elif self == MeasurementType.MAGNETIC:
            return "MAGNETIC"
        else:
            return "Unknown"
    
    def __repr__(self):
        return f"MeasurementType<{str(self)}>"
    

@dataclass
class GeophysicalMeasurement():
    name: MeasurementType
    std: int | float | int64 | float64

    @classmethod
    def from_dict(cls, config: dict):
        name = config['name'].upper()
        if name == "BATHYMETRY" or name == "BATHY":
            return cls(MeasurementType.BATHYMETRY, config["std"])
        elif name == "RELIEF" or name == "TERRAIN":
            return cls(MeasurementType.RELIEF, config["std"])
        elif name == "GRAVITY" or name == "GRAV":
            return cls(MeasurementType.GRAVITY, config["std"])
        elif name == "MAGNETIC" or name == "MAG":
            return cls(MeasurementType.MAGNETIC, config["std"])
        else:
            raise ValueError(f"Measurement type {name} not recognized.")
    
    @classmethod
    def to_dict(cls, config):
        return {
            "name": str(config.name),
            "std": config.std,
        }

    def __repr__(self):
        return f"{self.name}\n\tStandard Deviation={self.std}"

@dataclass
class ParticleFilterConfig():
    n: int
    cov: NDArray[float64 | int64] | list[float | int]
    noise: NDArray[float64 | int64] | list[float | int]
    measurement_config: list[GeophysicalMeasurement]

    @classmethod
    def from_dict(cls, config: dict) -> dict:
        n = config["n"]
        cov = array(config["cov"])
        noise = array(config["noise"])
        measurement_config = [GeophysicalMeasurement.from_dict(meas) for meas in config["measurement_config"]]
        return cls(n, cov, noise, measurement_config)
        
    
    @classmethod
    def to_dict(cls, config):
        return {
            "n": config.n,
            "cov": config.cov.tolist(),
            "noise": config.noise.tolist(),
            "measurement_config": [GeophysicalMeasurement.to_dict(meas) for meas in config.measurement_config]
        }

    @classmethod
    def load(cls, path: str) -> dict:
        with open(path, "r") as file:
            return cls.from_dict(json.load(file))
    

    @classmethod
    def save(self, path: str):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file)


    def __repr__(self) -> str:
        return f"Particle Filter Configuration:\n\tNumber of Particles={self.n}\n\tCovariance={self.cov}\n\tNoise={self.noise}\n\tMeasurement Configurations={self.measurement_config}"

def coning_and_sculling_correction(
    current_gyros: NDArray[float64 | int64] | list[float | int],
    previous_gyros: NDArray[float64 | int64] | list[float | int],
    current_accels: NDArray[float64 | int64] | list[float | int],
    previous_accels: NDArray[float64 | int64] | list[float | int],
    dt: float64 | int64,
) -> tuple[NDArray, NDArray]:
    """
    Compute the coning and sculling corrections for a strapdown INS and return the
    corrected angular increments and velocity increments.

    Parameters
    ----------
    current_gyros : array_like
        The current gyro measurements.
    previous_gyros : array_like
        The previous gyro measurements.
    current_accels : array_like
        The current accelerometer measurements.
    previous_accels : array_like
        The previous accelerometer measurements.

    Returns
    -------
    tuple
        A tuple containing the corrected angular increments and velocity increments.

    Example
    -------
    >>> current_gyros = np.array([0.1, 0.2, 0.3])
    >>> previous_gyros = np.array([0.05, 0.15, 0.25])
    >>> current_accels = np.array([0.1, 0.2, 0.3])
    >>> previous_accels = np.array([0.05, 0.15, 0.25])
    >>> coning_and_sculling_correction(current_gyros, previous_gyros, current_accels, previous_accels)
    (array([0.075, 0.175, 0.275]), array([0.075, 0.175, 0.275]))

    """
    current_gyros = array(current_gyros)
    previous_gyros = array(previous_gyros)
    current_accels = array(current_accels)
    previous_accels = array(previous_accels)

    current_accels = current_accels.squeeze()
    previous_accels = previous_accels.squeeze()
    current_gyros = current_gyros.squeeze()
    previous_gyros = previous_gyros.squeeze()

    # Insure the inputs are numpy arrays of the correct shape (3,)
    assert current_gyros.shape == (3,), "Current gyros must be a 3-element vector."
    assert previous_gyros.shape == (3,), "Previous gyros must be a 3-element vector."
    assert current_accels.shape == (3,), "Current accels must be a 3-element vector."
    assert previous_accels.shape == (3,), "Previous accels must be a 3-element vector."

    # Rate calibrations
    a_gyro = previous_gyros
    b_gyro = current_gyros - previous_gyros
    a_accel = previous_accels
    b_accel = current_accels - previous_accels

    # Coning and sculling corrections
    gyro_increment = (a_gyro + 0.5 * b_gyro) * dt
    accel_increment = (a_accel + 0.5 * b_accel) * dt
    coning = cross(a_gyro, b_gyro) * dt**2 / 12
    sculling = (cross(a_gyro, b_accel) + cross(a_accel, b_gyro)) * dt**2 / 12

    theta = gyro_increment + coning
    dv = accel_increment + sculling + 0.5 * cross(gyro_increment, accel_increment)

    return theta, dv


def vector_to_skew_symmetric(
    v: NDArray[int64 | float64] | list[float | int],
) -> NDArray[int64 | float64]:
    """
    Convert a 3-element vector to a skew-symmetric matrix.

    Parameters
    ----------
    v : array_like
        A 3-element vector.

    Returns
    -------
    array_like
        A 3x3 skew-symmetric matrix.
    """
    v = array(v).squeeze()
    assert v.shape == (3,), "Input must be a 3-element vector."
    return array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float64)


def skew_symmetric_to_vector(
    m: NDArray[float64 | int64] | list[float | int],
) -> NDArray[float64 | int64]:
    """
    Convert a 3x3 skew-symmetric matrix to a 3-element vector.

    Parameters
    ----------
    m : array_like
        A 3x3 skew-symmetric matrix.

    Returns
    -------
    NDArray
        A 3-element vector
    """
    m = array(m).squeeze()
    assert m.shape == (3, 3), "Input must be a 3x3 matrix."
    return array([m[2, 1], m[0, 2], m[1, 0]])


@njit
def _propagate_imu(
    particles: NDArray,
    C_: NDArray,
    gyros: NDArray,
    accels: NDArray,
    dt: float64,
    Rn,
    Re,
    gravity_vector,
):
    """
    NED strapdown INS equations using NumPy and Numba. Numba does not support matrix
    multiplication with the @ operator for three dimensional arrays (aka lists of matrices)
    so we need to bring in the loop over the particles to the Python level. This function
    is not intended for use outside of the propagate_imu function as it makes certain
    assumptions about the input and does not validate it as Numba has limited support
    data validation.

    Parameters
    ----------
    particles : NDArray, (n, 15)
        The particles to propagate. NED strapdown state vector with the following
        elements: [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]
    C_ : array_like
        The previous attitude matrix.
    """
    n = len(particles)
    # Prior values
    lat_ = deg2rad(particles[:, 0])
    lon_ = deg2rad(particles[:, 1])
    alt_ = particles[:, 2]
    vn_ = particles[:, 3]
    ve_ = particles[:, 4]
    vd_ = particles[:, 5]
    # Initializing output arrays
    C = empty((n, 3, 3), dtype=float64)
    f = empty((n, 3), dtype=float64)
    velocity = empty((n, 3), dtype=float64)
    # Compute the angular velocity of the Earth relative to the local-level frame
    omega = empty((n, 3), dtype=float64)
    omega[:, 0] = ve_ / (Re + alt_)
    omega[:, 1] = -vn_ / (Rn + alt_)
    omega[:, 2] = -vn_ * tan(lat_) / (Rn + alt_)
    # Loop over the particles, Numba doesn't support (n, 3, 3) @ (3, 3) so we need to do it via a loop over n
    for i in range(n):
        # Attitude update
        Omega_ie = EARTH_RATE * _vector_to_skew_symmetric([cos(lat_[i]), 0, -sin(lon_[i])])
        Omega_en = _vector_to_skew_symmetric(omega[i])
        Omega_ib = _vector_to_skew_symmetric(gyros - particles[i, 9:12])
        C[i] = C_[i] @ (eye(3) + Omega_ib * dt) - (Omega_ie + Omega_en) @ C_[i] * dt
        # Specific force update
        f[i] = 0.5 * (C[i] + C_[i]) @ (accels[i] - particles[i, 12:])
        # Velocity update
        q = Omega_en + 2 * Omega_ie
        transport_rate = q @ particles[i, 3:6]
        velocity[i] = particles[i, 3:6] + (f[i] + gravity_vector[i] - transport_rate) * dt
    # Position Update
    alt = alt_ - dt / 2 * (vd_ + velocity[:, 2])
    lat = rad2deg(lat_ + dt / 2 * (vn_ / (Rn + lat_) + velocity[:, 0] / (Rn + alt)))
    lon = rad2deg(lon_ + dt / 2 * (ve_ / ((Re + alt_) * cos(lat_)) + velocity[:, 1] / ((Re + alt) * cos(lat))))

    return column_stack((lat, lon, alt, velocity)), C


@njit
def _vector_to_skew_symmetric(v: NDArray[int64 | float64]) -> NDArray[int64 | float64]:
    """
    Convert a 3-element vector to a skew-symmetric matrix. Jitted function for use
    in _propagate_imu. This function is not intended for use outside of the
    propagate_imu function as it makes certain assumptions about the input and
    does not validate it as Numba has limited support data validation.

    Please use the vector_to_skew_symmetric function for general use.

    Parameters
    ----------
    v : array_like
        A 3-element vector.

    Returns
    -------
    array_like
        A 3x3 skew-symmetric matrix.
    """
    # assert v.shape == (3,), "Input must be a 3-element vector."
    return array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=float64)


def propagate_imu(
    particles: NDArray[float64 | int64] | list[float | int],
    gyros: NDArray[float64 | int64] | list[float | int],
    accels: NDArray[float64 | int64] | list[float | int],
    dt: float64 | int64 | float | int = 1.0,
    noise: NDArray[float64 | int64] | list[float | int] = zeros(15, dtype=int64),
) -> NDArray:
    """
    Propagate the particles according to the strapdown INS equations in the
    NED frame. The particles are propagated based on the IMU output and the
    previous state. The particles are assumed to be in the following format:
    [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]

    Parameters:
    -----------
    particles : array-like (n x 15; nine navigation states six IMU states)
        The particles to propagate
    gyros : array-like (3,)
        The current gyro output from the IMU in rad/s
    accels : array-like (3,)
        The current accelerometer output from the IMU in m/s^2
    dt : numeric, greater than zero (default=1.0)
        The time step to propagate the particles in seconds
    noise : array-like (15,) (default=zeros(15,))
        The noise vector for the system in terms of variance (standard deviation squared)

    Returns:
    --------
    particles : np.ndarray (n x 12)
        The propagated particles.

    References:
    -----------
    .. [1] Groves, P. D. Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems (2nd ed.).
    """
    # Input validation
    particles = array(particles)
    gyros = array(gyros)
    accels = array(accels)
    noise = array(noise)
    assert particles.shape[1] >= 15, "Please check dimensions of particles. Particles must have at least 15 elements corresponding to the strapdown INS states."
    assert gyros.shape == (3,), "Gyros must be a 3-element vector."
    assert accels.shape == (3,), "Accels must be a 3-element vector."
    assert noise.shape[0] == particles.shape[1], "Noise must either be a vector or square matrix of equal dimension to the state vector (>=15)."
    assert dt > 0, "Time step must be greater than zero."
    assert all(noise >= 0), "Noise must be greater than or equal to zero."
    # Getting some constants
    Rn, Re, _ = earth.principal_radii(particles[:, 0], particles[:, 2])
    gravity_vector = earth.gravity_n(particles[:, 0], particles[:, 2])
    C_ = transform.mat_from_rph(deg2rad(particles[:, 6:9]))
    new_particles, C = _propagate_imu(particles, C_, gyros, accels, dt, noise, Rn, Re, EARTH_RATE, gravity_vector)
    new_particles = column_stack([new_particles, transform.mat_to_rph(C), particles[:, 9:]])
    jitter = mvn(len(particles[0]), diag(noise), len(particles))
    return new_particles + jitter


# Measurement Functions

def update_relief(
    particles: NDArray[float64 | int64] | list[float | int],
    geo_map: DataArray,
    observation: float | int | float64 | int64,
    sigma: float,
    bias: NDArray[float64 | int64] | list[float | int] = zeros(1, dtype=float64),
) -> NDArray[float64]:
    """
    Measurement update for bathymetry or terrain relief. This measurement function assumes a single scalar
    observation of a depth-below-keel type measurement from a sonar or altimeter. The particles are assumed to be
    in the following format: [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]. The
    observation is corrected for the altitude of the particles and compared to the geophysical map. The weights
    are then updated based on the difference between the observation and the map.

    Parameters
    ----------
    particles : array_like
        The particles to update. The particles are assumed to be in the following format:
        [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz, ...] where the first
        15 elements are the navigation states and the remaining elements are extra measurement or particle
        filter states.
    geo_map : DataArray
        The geophysical map to compare the particles to. The map should be a DataArray with the following
        dimensions: lat, lon.
    observation : numeric
        The observation to compare the particles to. This is assumed to be a scalar value.
    sigma : numeric
        The standard deviation of the normal distribution.
    bias : array_like (default=zeros(1,))
        The bias of the observation. This is assumed to be a scalar value. Due to variability in the configuration
        of the particle filter state vector, the bias (and subsequently any other measurement corrections) are 
        included as separate arguments.
    
    Returns
    -------
    array_like
        The updated weights of the particles.
    """

    observation = asarray(observation)
    bias = asarray(bias)
    observation = tile(observation, (particles.shape[0],))
    if bias.shape[0] < particles.shape[0]:
        bias = tile(bias, (particles.shape[0],))
    observation -= particles[:, 2] + bias
    return _update(particles, geo_map, observation, sigma)

def update_anomaly(
    particles: NDArray[float64 | int64] | list[float | int],
    geo_map: DataArray,
    observation: float | int | float64 | int64,
    sigma: float | int | float64 | int64,
    bias: NDArray[float64 | int64] | list[float | int] = zeros(1, dtype=float64),
) -> NDArray[float64]:
    """
    Measurement update for magnetic or gravimetric anomaly. This measurement function assumes a single scalar
    observation of an anomaly type measurement. The particles are assumed to be in the following format: 
    [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz]. The weights are then updated 
    based on the difference between the observation and the map. Unlike with terrain relief and bathymetry, the
    observation is not corrected for the altitude of the particles.

    Parameters
    ----------
    particles : array_like
        The particles to update. The particles are assumed to be in the following format:
        [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz, ...] where the first
        15 elements are the navigation states and the remaining elements are extra measurement or particle
        filter states.
    geo_map : DataArray
        The geophysical map to compare the particles to. The map should be a DataArray with the following
        dimensions: lat, lon.
    observation : numeric
        The observation to compare the particles to. This is assumed to be a scalar value.
    sigma : numeric
        The standard deviation of the normal distribution.
    bias : array_like (default=zeros(1,))
        The bias of the observation. This is assumed to be a scalar value. Due to variability in the configuration
        of the particle filter state vector, the bias (and subsequently any other measurement corrections) are 
        included as separate arguments.
    
    Returns
    -------
    array_like
        The updated weights of the particles.
    """

    observation = asarray(observation)
    bias = asarray(bias)
    observation = tile(observation, (particles.shape[0],))
    if bias.shape[0] < particles.shape[0]:
        bias = tile(bias, (particles.shape[0],))
    observation -= bias
    return _update(particles, geo_map, observation, sigma)

def _update(particles: NDArray[float64 | int64] | list[float | int],
            geo_map: DataArray,
            observation: float | int | float64 | int64,
            sigma: float) -> NDArray[float64]:
    """
    Generic measurement update function for the particle filter that assumes a zero mean normal distribution.
    Intended as a helper function for the update functions so as to reduce code duplication. This function
    assumes that any measurement corrections have already occurred and simply calculates the probabilities
    of the particles assuming a zero mean normal distribution according to the difference between the 
    observation and the map.

    Parameters
    ----------
    particles : array_like
        The particles to update. The particles are assumed to be in the following format:
        [lat, lon, alt, vn, ve, vd, roll, pitch, yaw, bgx, bgy, bgz, bax, bay, baz, ...] where the first
        15 elements are the navigation states and the remaining elements are extra measurement or particle
        filter states.
    geo_map : DataArray
        The geophysical map to compare the particles to. The map should be a DataArray with the following
        dimensions: lat, lon.
    observation : numeric
        The observation to compare the particles to. This is assumed to be a scalar value.
    sigma : numeric
        The standard deviation of the normal distribution.

    Returns
    -------
    array_like
        The updated weights of the particles.
    """
    z_bar = -get_map_point(geo_map, particles[:, 1], particles[:, 0])
    dz = observation - z_bar
    w = norm(loc=0, scale=sigma).pdf(dz)
    w[isnan(w)] = 1e-16
    return w


def run_particle_filter(
    trajectory: DataFrame,
    geomaps: dict[MeasurementType, DataArray],
    config: ParticleFilterConfig,
):
    """
    Run through an instance of the particle filter give a trajectory, map, and a configuration for the particle filter.

    Parameters
    ----------
    trajectory : DataFrame
        Trajectory should contain ground truth data ("lat", "lon", "alt", "VN", "VE", "VD", "roll", "pitch", "heading"), IMU
        data ("gyro_x", "gyro_y", "gyro_z", "accel_x", "accel_y", "accel_z"), observational data ("depth", "gra_obs", "freeair",
        "mat_tot", "mag_res"), and indexed to a datetime. The geomap should be a DataArray containing the geophysical data with 
        the axis corresponding to "lon" and "lat".
    geomap : dict[MeasurementType, DataArray]
        The geophysical map to compare the particles to. The map should be a DataArray with the following
        dimensions: lat, lon.
    config : ParticleFilterConfig
        The configuration for the particle filter, containing the following values:
        n: int
            The number of particles to use in the filter.
        cov: array_like
            The covariance matrix or diagonal for the initial state vector.
        noise: array_like
            The noise matrix or diagonal for the system.
        measurement_config: list[GeophysicalMeasurement]
            The list of geophysical measurements to use in the filter.
    
    """
    # To do: condense the input of this function so that the simulation can be compartmentalized, suggest using a data class for particle
    # filter configuration. 
    mu = trajectory.loc[trajectory.index[0], ["lat", "lon", "alt", "VN", "VE", "VD", "roll", "pitch", "heading"]].to_numpy()
    mu = append(mu, zeros(6 + len(config.measurement_config))) # add size IMU biases and other measurement biases
    assert mu.shape[0] == config.cov.shape[0], f"Initial state vector and covariance matrix do not match dimensions. {mu.shape[0]} != {config.cov.shape[0]}"
    assert mu.shape[0] == config.noise.shape[0], f"Initial state vector and noise matrix do not match dimensions. {mu.shape[0]} != {config.noise.shape[0]}"
    assert isinstance(config.n, int), "Number of particles must be an integer."
    assert config.n > 0, "Number of particles must be greater than zero."
    # Initialization
    particles = mvn(mu, diag(config.cov), (config.n,))
    weights = ones((config.n,)) / config.n
    rms_error_2d = zeros(len(trajectory))
    rms_error_3d = zeros_like(rms_error_2d)
    estimate = zeros((len(trajectory), particles.shape[1]))
    trajectory['dt'] = trajectory.index.to_series().diff().dt.seconds.fillna(0)

    # Jit the loop?
    for i, item in enumerate(trajectory.iterrows()):
        row = item[1]
        # Error calculations
        estimate[i] = weights @ particles
        rms_error_2d[i] = rmse(particles, row[['lat', 'lon']].to_numpy(), include_altitude=False, weights=weights)
        rms_error_3d[i] = rmse(particles, row[['lat', 'lon', 'alt']].to_numpy(), include_altitude=True, weights=weights)        
        # Propagate particles
        particles = propagate_imu(particles, row[["gyro_x", "gyro_y", "gyro_z"]], row[["accel_x", "accel_y", "accel_z"]], row["dt"], config.noise)
        # Update weights
        # To a certain extent the below is a measurement model itself, however a sensor fusion model hasn't 
        # been investigated yet that would allow for a more informed measurement model. For now, each measurement
        # will be treated as independent and equally weighted.
        new_weights = zeros_like(weights)
        for measurement in config.measurement_config:
            if measurement.name == MeasurementType.BATHYMETRY:
                new_weights += update_relief(particles, geomaps[measurement.name], row.depth, measurement.std)
            elif measurement.name == MeasurementType.RELIEF:
                new_weights += update_relief(particles, geomaps[measurement.name], row.depth, measurement.std)
            elif measurement.name == MeasurementType.GRAVITY:
                new_weights += update_anomaly(particles, geomaps[measurement.name], row.freeair, measurement.std)
            elif measurement.name == MeasurementType.MAGNETIC:
                new_weights += update_anomaly(particles, geomaps[measurement.name], row.mag_res, measurement.std)
            else:
                raise ValueError(f"Measurement type {measurement.name} not recognized.")
        weights = new_weights / sum(new_weights)
        # Resample
        inds = residual_resample(weights)
        particles[:] = particles[inds]
    # Final error calculations
    i += 1
    estimate[i] = weights @ particles
    rms_error_2d[i] = rmse(particles, row[['lat', 'lon']].to_numpy(), include_altitude=False, weights=weights)
    rms_error_3d[i] = rmse(particles, row[['lat', 'lon', 'alt']].to_numpy(), include_altitude=True, weights=weights)       

    return estimate, rms_error_2d, rms_error_3d


# Simulation functions
def process_particle_filter(
    trajectory: DataFrame,
    config: ParticleFilterConfig,
    map_type: str = "relief",
    map_resolution: str = "15s",
) -> DataFrame:
    """
    Process a single run of the particle filter. This method provides the neccessary
    pre-processing to load the map(s) and setup the initial conditions for the particle filter
    that are then given to run_particle_filter().
    """

    # trajectory: DataFrame,
    # geomap: DataArray,
    # config: ParticleFilterConfig,


    min_lon = data["LON"].min()
    max_lon = data["LON"].max()
    min_lat = data["LAT"].min()
    max_lat = data["LAT"].max()
    min_lon, min_lat, max_lon, max_lat = inflate_bounds(min_lon, min_lat, max_lon, max_lat, 0.25)
    geo_map = get_map_section(
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        map_type,
        map_resolution,
    )
    # Load initial conditions
    mu = np.asarray([data.iloc[0].LAT, data.iloc[0].LON, 0, 0, 0, 0])
    cov = np.asarray(configurations["cov"])
    cov = np.diag(cov)
    noise = np.asarray(configurations["velocity_noise"])
    noise = np.diag(noise)
    if map_type == "relief":
        measurement_sigma = configurations["bathy_std"]
        measurment_type = "DEPTH"
        measurement_bias = configurations["bathy_mean_d"]
    elif map_type == "gravity":
        measurement_sigma = configurations["gravity_std"]
        measurement_bias = configurations["gravity_mean_d"]
        measurment_type = "GRAV_ANOM"
    elif map_type == "magnetic":
        measurement_sigma = configurations["magnetic_std"]
        measurement_bias = configurations["magnetic_mean_d"]
        measurment_type = "MAG_RES"
    else:
        raise ValueError("Map type not recognized")

    n = configurations["n"]
    data[measurment_type] = data[measurment_type] - measurement_bias

    estimate, rms_error, error = run_particle_filter(
        mu,
        cov,
        n,
        data,
        geo_map,
        noise,
        measurement_sigma,
        measurment_type=measurment_type,
    )

    data[["PF_LAT", "PF_LON", "PF_DEPTH", "PF_VN", "PF_VE", "PF_VD"]] = estimate
    data["RMSE"] = rms_error
    data["ERROR"] = error
    return data, geo_map


# Error functions
def rmse(particles: NDArray[int64 | float64], truth: NDArray[int64 | float64], include_altitude:bool=False, weights: NDArray[float64]=ones(1, dtype=float64)) -> float64:
    """
    Root mean square error calculation to calculate the error between the particles and the truth. If weights are provided, the
    squared error is weighted by the weights.

    Parameters
    ----------
    particles : array_like
        The particles to calculate the error for.
    truth : vector_like
        The truth point to compare the particles to.
    include_altitude : bool (default=False)
        Include the altitude in the error calculation.
    weights : array_like (default=ones(1,))
        The weights of the particles. Default value insures that the weights are all equal and the error is not weighted.
    
    Returns
    -------
    float
        The root mean square error between the particles and the truth.
    """
    n = particles.shape[0]
    if truth.shape[0] != n:
        truth = tile(truth, (n, 1))
    if weights.shape[0] != n:
        weights = ones((n,))
    if include_altitude:
        alts = (particles[:, 2] - truth[2])
    else:
        alts = zeros((n,))
    diffs = haversine_vector(truth[:, :2], particles[:, :2], Unit.METERS)**2 + alts**2

    return sqrt(mean(diffs * weights))


# Plotting functions
# Plot the map and the trajectory
def plot_map_and_trajectory(
    geo_map: DataArray,
    data: DataFrame,
    title_str: str = "Map and Trajectory",
    title_size: int = 20,
    xlabel_str: str = "Lon (deg)",
    xlabel_size: int = 14,
    ylabel_str: str = "Lat (deg)",
    ylabel_size: int = 14,
):
    """
    Plot the trajectory two dimensionally on the map

    Parameters
    ----------
    geo_map : DataArray
        The map to plot on
    data : DataFrame
        The data to plot
    title_str : str
        The title of the plot
    title_size : int
        The size of the title
    xlabel_str : str
        The x axis label
    xlabel_size : int
        The size of the x axis label
    ylabel_str : str
        The y axis label
    ylabel_size : int
        The size of the y axis label


    Returns
    -------
    fig : Figure
        The figure object
    """
    min_lon = data.LON.min()
    max_lon = data.LON.max()
    min_lat = data.LAT.min()
    max_lat = data.LAT.max()
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.contourf(geo_map.lon, geo_map.lat, geo_map.data)
    ax.plot(data.LON, data.LAT, ".r", label="Truth")
    ax.plot(data.iloc[0].LON, data.iloc[0].LAT, "xk", label="Start")
    ax.plot(data.iloc[-1].LON, data.iloc[-1].LAT, "bo", label="Stop")
    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])
    ax.set_xlabel(xlabel_str, fontsize=xlabel_size)
    ax.set_ylabel(ylabel_str, fontsize=ylabel_size)
    ax.set_title(title_str, fontsize=title_size)
    ax.axis("image")
    ax.legend()
    return fig, ax
    # plt.show()
    # plt.savefig(f'{name}.png')


# Plot the particle filter estimate
def plot_estimate(
    geo_map: DataArray,
    data: DataFrame,
    title_str: str = "Particle Filter Estimate",
    title_size: int = 20,
    xlabel_str: str = "Lon (deg)",
    xlabel_size: int = 14,
    ylabel_str: str = "Lat (deg)",
    ylabel_size: int = 14,
    measurment_type: str = "depth",
):
    """
    Plot the particle filter estimate and the trajectory two dimensionally on the map.

    Parameters
    ----------
    geo_map : DataArray
        The map to plot on
    data : DataFrame
        The data to plot
    estimate : np.ndarray
        The estimate to plot
    Returns
    -------
    fig : Figure
        The figure object
    """

    cmap = "ocean"
    clim = [-10000, 0]
    clabel = "Depth (m)"
    if measurment_type == "gravity":
        cmap = "coolwarm"
        clim = [-100, 100]
        clabel = "Gravity Anomaly (mGal)"
    elif measurment_type == "magnetic":
        cmap = "PiYG"
        clim = [-100, 100]
        clabel = "Magnetic Anomaly (nT)"

    min_lon = data.LON.min()
    max_lon = data.LON.max()
    min_lat = data.LAT.min()
    max_lat = data.LAT.max()
    fig, ax = plt.subplots(1, 1)  # , figsize=(16, 8))
    contour = ax.contourf(geo_map.lon, geo_map.lat, geo_map.data, cmap=cmap, levels=50)
    # Set the color map limits
    # ax.set_clim([-5000, 0])
    contour.set_clim(clim)
    # Plot the colorbar for the map. If the map is taller than wide plot it on the right, otherwise plot it on the
    # bottom
    aspect_ratio = (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    if aspect_ratio > 1:
        cbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="horizontal",
            # fraction=0.1,
            # pad=0.05,
            # aspect=50,
            # shrink=0.5,
        )
        cbar.set_label(clabel)

    else:
        cbar = fig.colorbar(
            contour,
            ax=ax,
            orientation="vertical",
            # fraction=0.1,
            # pad=0.05,
            # aspect=50,
            # shrink=0.75,
        )
        cbar.set_label(clabel)

    ax.plot(data.LON, data.LAT, ".r", label="Truth")
    ax.plot(data.iloc[0].LON, data.iloc[0].LAT, "xk", label="Start")
    ax.plot(data.iloc[-1].LON, data.iloc[-1].LAT, "bo", label="Stop")

    ax.set_xlim([min_lon, max_lon])
    ax.set_ylim([min_lat, max_lat])
    ax.set_xlabel(xlabel_str, fontsize=xlabel_size)
    ax.set_ylabel(ylabel_str, fontsize=ylabel_size)
    ax.set_title(title_str, fontsize=title_size)

    ax.plot(data.PF_LON, data.PF_LAT, "g.", label="PF Estimate")
    ax.axis("image")
    ax.legend()
    return fig, ax


# Plot the particle filter error characteristics
def plot_error(
    data: DataFrame,
    title_str: str = "Particle Filter Error",
    title_size: int = 20,
    xlabel_str: str = "Time (hours)",
    xlabel_size: int = 14,
    ylabel_str: str = "Error (m)",
    ylabel_size: int = 14,
    max_error: int = 5000,
    annotations: dict = None,
) -> tuple:
    """
    Plot the error characteristics of the particle filter with respect to
    truth and map pixel resolution

    Parameters
    ----------
    data : DataFrame
        The data to plot
    rms_error : np.ndarray
        The error values to plot with respect to time
    res : float
        The resolution of the map in meters
    title_str : str
        The title of the plot
    title_size : int
        The size of the title
    xlabel_str : str
        The x axis label
    xlabel_size : int
        The size of the x axis label
    ylabel_str : str
        The y axis label
    ylabel_size : int
        The size of the y axis label
    max_error : int
        The maximum error to plot

    Returns
    -------
    fig : Figure
        The figure object
    """
    # res = haversine((0, 0), (geo_map.lat[1] - geo_map.lat[0], 0), Unit.METERS)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    time = (data.index - data.index[0]) / timedelta(hours=1)

    ax.plot(time, data.RMSE, label="RMSE")
    # ax.plot(time, data.ERROR, "--", label="Error")
    if annotations is not None:
        if annotations["recovery"] is not None:
            ax.plot(
                time,
                np.ones_like(time) * annotations["recovery"],
                label="Recovery",
            )
            # highlight the area undereath the points where the error is less than the pixel resolution
            ax.fill_between(
                time,
                data.RMSE.values,
                np.ones_like(time) * annotations["recovery"],
                where=data.RMSE.values < annotations["recovery"],
                color="blue",
                alpha=0.25,
            )
        if annotations["res"] is not None:
            ax.plot(
                time,
                np.ones_like(time) * annotations["res"],
                label="Pixel Resolution",
            )
            # highlight the area undereath the points where the error is less than the pixel resolution
            ax.fill_between(
                time,
                data.RMSE.values,
                np.ones_like(time) * annotations["res"],
                where=data.RMSE.values < annotations["res"],
                color="green",
                alpha=0.25,
            )

    # ax.plot(data['TIME'] / timedelta(hours=1), weighted_rmse, label='Weighted RMSE')
    ax.set_xlabel(xlabel_str, fontsize=xlabel_size)
    ax.set_ylabel(ylabel_str, fontsize=ylabel_size)
    ax.set_title(title_str, fontsize=title_size)
    ax.set_ylim([0, max_error])
    ax.legend()
    return fig, ax


def summarize_results(name: str, results: DataFrame, threshold: float):
    """
    Used to generate a text file that summarizes the results.
    """
    under_threshold = results["RMSE"].to_numpy() <= threshold
    # Default behavior for find_periods is to find transitions from False to True
    recoveries = find_periods(~under_threshold)

    start = []
    end = []
    duration = []
    average_error = []
    min_error = []
    max_error = []
    names = []
    initial_time = results.index[0]
    for recovery in recoveries:
        start_time = results.index[recovery[0]] - initial_time
        end_time = results.index[recovery[1]] - initial_time
        if end_time > start_time:
            names.append(name)
            start.append(start_time)
            end.append(end_time)
            duration.append(end_time - start_time)
            average_error.append(results[recovery[0] : recovery[1]].RMSE.mean())
            min_error.append(results[recovery[0] : recovery[1]].RMSE.min())
            max_error.append(results[recovery[0] : recovery[1]].RMSE.max())

    summary = DataFrame(
        {
            "name": names,
            "start": start,
            "end": end,
            "duration": duration,
            "average_error": average_error,
            "min error": min_error,
            "max error": max_error,
        }
    )
    return summary
