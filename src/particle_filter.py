"""
Particle filter algorithim and simulation code
"""
from datetime import timedelta
import os
import json
import argparse
import multiprocessing

from scipy.stats import norm
from scipy.io import savemat
from pandas import DataFrame
from xarray import DataArray
from haversine import haversine, Unit
import numpy as np
from matplotlib import pyplot as plt

from filterpy.monte_carlo import residual_resample
from gmt_tool import get_map_point, get_map_section, inflate_bounds
from tools import load_trackline_data
from pyins.pyins import earth

OVERFLOW = 500


# Particle filter functions
def propagate(
    particles: np.ndarray,
    control: np.ndarray,
    dt: float = 1.0,
    noise=np.diag([0, 0, 0]),
    noise_calibration_mode=False,
) -> np.ndarray:
    """
    Process model according to Groves. State vector is in NED:
    [lat, lon, depth, vN, vE, vD]. Controls are currently none, dt is in seconds.

    """
    n, _ = particles.shape
    # Velocity update
    if not noise_calibration_mode:
        velocity = np.random.multivariate_normal(control, noise, (n,))
    else:
        velocity = control + np.abs(
            np.random.multivariate_normal(np.zeros(3), noise, (n,))
        )

    # Depth update
    previous_depth = particles[:, 2]
    new_depth = previous_depth + 0.5 * (particles[:, 5] + velocity[:, 2]) * dt
    # Latitude update
    previous_lat = particles[:, 0]
    lat_rad = np.deg2rad(previous_lat)
    r_n, r_e_0, _ = earth.principal_radii(
        previous_lat, np.zeros_like(previous_lat)
    )  # pricipal_radii expect degrees
    previous_lat = np.deg2rad(previous_lat)
    lat_rad = previous_lat + 0.5 * dt * (
        particles[:, 3] / (r_n - previous_depth) + velocity[:, 0] / (r_n - new_depth)
    )
    # Longitude update
    _, r_e_1, _ = earth.principal_radii(np.rad2deg(lat_rad), np.zeros_like(lat_rad))
    lon_rad = np.deg2rad(particles[:, 1]) + 0.5 * dt * (
        particles[:, 4] / ((r_e_0 - previous_depth) * np.cos(previous_lat))
        + velocity[:, 1] / ((r_e_1 - new_depth) * np.cos(lat_rad))
    )

    particles = np.array([np.rad2deg(lat_rad), np.rad2deg(lon_rad), new_depth]).T
    particles = np.hstack([particles, velocity])
    return particles


def update_relief(
    particles: np.ndarray,
    # weights: np.ndarray,
    geo_map: DataArray,
    observation,
    relief_sigma: float,
) -> np.ndarray:
    """
    Measurement update
    """

    n, _ = particles.shape
    observation = np.asarray(observation)
    observation = np.tile(observation, (n,))
    observation -= particles[:, 2]
    z_bar = -get_map_point(geo_map, particles[:, 1], particles[:, 0])
    dz = observation - z_bar
    w = np.zeros_like(dz)

    inds = np.abs(dz) < OVERFLOW
    w[inds] = norm(loc=0, scale=relief_sigma).pdf(dz[inds])

    w[np.isnan(w)] = 1e-16
    if np.any(np.isnan(w)):
        print("NAN elements found")
        w[np.isnana(w)] = 1e-16

    w_sum = np.nansum(w)
    try:
        new_weights = w / w_sum
    except ZeroDivisionError:
        return np.ones_like(w) / n
    return new_weights


def run_particle_filter(
    mu: np.ndarray,
    cov: np.ndarray,
    n: int,
    data: DataFrame,
    geo_map: DataArray,
    noise: np.ndarray = np.diag([0.1, 0.01, 0]),
    measurement_sigma: float = 15,
):
    """
    Run through an instance of the particle filter
    """
    particles = np.random.multivariate_normal(mu, cov, (n,))
    weights = np.ones((n,)) / n
    error = np.zeros(len(data))
    rms_error = np.zeros_like(error)
    # Initial values
    estimate = [weights @ particles]
    rms_error[0] = rmse(particles, (data.iloc[0].LAT, data.iloc[0].LON))

    for i, item in enumerate(data.iterrows()):
        if i > 0:
            row = item[1]
            # Propagate
            u = np.asarray([row["vN"], row["vE"], 0])
            particles = propagate(particles, u, row["DT"].seconds, noise)
            # Update
            obs = row["CORR_DEPTH"]
            weights = update_relief(particles, geo_map, obs, measurement_sigma)
            # Resample
            inds = residual_resample(weights)
            particles[:] = particles[inds]
            # Calculate estimate and error
            estimate.append(weights @ particles)
            rms_error[i] = rmse(particles, (row.LAT, row.LON))
    estimate = np.asarray(estimate)
    return estimate, rms_error


# Simulation functions
def process_particle_filter(
    path_to_data: str,
    configurations: dict,
    output_dir: str,
    map_type: str = "relief",
    map_resolution: str = "15s",
):
    """
    Process the particle filter for a given set of configurations
    """
    # Load data
    data = load_trackline_data(path_to_data)
    # Process filepath for name
    path, file = os.path.split(path_to_data)
    name, ext = os.path.splitext(file)
    # Load map
    min_lon = data.LON.min()
    max_lon = data.LON.max()
    min_lat = data.LAT.min()
    max_lat = data.LAT.max()
    min_lon, min_lat, max_lon, max_lat = inflate_bounds(
        min_lon, min_lat, max_lon, max_lat, 0.25
    )
    geo_map = get_map_section(
        min_lon, max_lon, min_lat, max_lat, map_type, map_resolution, name
    )
    # Load initial conditions
    mu = np.asarray([data.iloc[0].LAT, data.iloc[0].LON, 0, 0, 0, 0])
    cov = np.asarray(configurations["cov"])
    cov = np.diag(cov)
    noise = np.asarray(configurations["velocity_noise"])
    noise = np.diag(noise)
    if map_type == "relief":
        measurement_sigma = configurations["bathy_std"]
    elif map_type == "gravity":
        measurement_sigma = configurations["gravity_std"]
    elif map_type == "magnetic":
        measurement_sigma = configurations["magnetic_std"]
    else:
        raise ValueError("Map type not recognized")

    n = configurations["n"]
    # Run particle filter
    estimate, rms_error = run_particle_filter(
        mu, cov, n, data, geo_map, noise, measurement_sigma
    )
    # Validate output path
    if not os.path.exists(os.path.join(output_dir, name, map_type)):
        os.makedirs(os.path.join(output_dir, name, map_type))
    # Plot results
    fig, ax = plot_map_and_trajectory(geo_map, data)
    fig.savefig(os.path.join(output_dir, name, map_type, "map_and_trajectory.png"))
    fig, ax = plot_estimate(geo_map, data, estimate)
    fig.savefig(os.path.join(output_dir, name, map_type, "estimate.png"))
    fig, ax = plot_error(data, rms_error)
    fig.savefig(os.path.join(output_dir, name, map_type, "error.png"))
    plt.close("all")


# Error functions
def rmse(particles, truth):
    """
    root mean square error calculation
    """
    diffs = [haversine(truth, (p[0], p[1]), Unit.METERS) for p in particles]
    diffs = np.asarray(diffs)
    return np.sqrt(np.mean(diffs**2))


def weighted_rmse(particles, weights, truth):
    """
    Weighted root mean square error calculation
    """
    diffs = [haversine(truth, (p[0], p[1]), Unit.METERS) for p in particles]
    diffs = np.asarray(diffs) * weights
    return np.sqrt(diffs**2)


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
    estimate: np.array,
    title_str: str = "Particle Filter Estimate",
    title_size: int = 20,
    xlabel_str: str = "Lon (deg)",
    xlabel_size: int = 14,
    ylabel_str: str = "Lat (deg)",
    ylabel_size: int = 14,
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

    ax.plot(estimate[:, 1], estimate[:, 0], "g.", label="PF Estimate")
    ax.axis("image")
    ax.legend()
    return fig, ax


# Plot the particle filter error characteristics
def plot_error(
    data: DataFrame,
    rms_error: np.array,
    res: float = None,
    title_str: str = "Particle Filter Error",
    title_size: int = 20,
    xlabel_str: str = "Time (hours)",
    xlabel_size: int = 14,
    ylabel_str: str = "Error (m)",
    ylabel_size: int = 14,
    max_error: int = 5000,
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
    time = data.index - data.index[0]
    if res is not None:
        ax.plot(
            time / timedelta(hours=1),
            np.ones_like(time) * res,
            label="Pixel Resolution",
        )
    ax.plot(time / timedelta(hours=1), rms_error, label="RMSE")
    # ax.plot(data['TIME'] / timedelta(hours=1), weighted_rmse, label='Weighted RMSE')
    ax.set_xlabel(xlabel_str, fontsize=xlabel_size)
    ax.set_ylabel(ylabel_str, fontsize=ylabel_size)
    ax.set_title(title_str, fontsize=title_size)
    ax.set_ylim([0, max_error])
    ax.legend()
    return fig, ax


def parse_args():
    """
    Command line interface specifications
    """
    parser = argparse.ArgumentParser(description="Particle Filter")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the particle filter configuration file",
        default="./config.json",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        help="Path to the data file or folder containing data files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output directory",
        default="./results/",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of map to use",
        default="relief",
    )
    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()
    config = json.load(open(args.config, "r", encoding="utf-8"))
    # Check to see if the data is a file or a folder
    if os.path.isfile(args.data):
        process_particle_filter(args.data, config, "./results/", args.type)
    elif os.path.isdir(args.data):
        # Get a list of all CSV files in the directory
        file_list = [
            os.path.join(args.data, file)
            for file in os.listdir(args.data)
            if file.endswith(".csv")
        ]
        cores = multiprocessing.cpu_count()
        # Process particle filters in parallel
        with multiprocessing.Pool(processes=cores) as pool:
            pool.starmap(
                process_particle_filter,
                [(file, config, "./results/", args.type) for file in file_list],
            )
            pool.close()
            pool.join()
    else:
        raise ValueError("Data path not recognized")


if __name__ == "__main__":
    main()
