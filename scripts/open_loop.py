import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from geophysical import gmt_toolbox as gmt
from data_management import dbmgr, m77t
from geophysical import particle_filter as pf
from geophysical.gmt_toolbox import GeophysicalMap, MeasurementType, ReliefResolution
import haversine as hs
import h5py
from tqdm import tqdm
import logging
import multiprocessing as mp
from datetime import timedelta
import time

logger = logging.getLogger("bathy_pf")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("bathy_pf.log")
handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter("%(asctime)s || %(message)s")
handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)

pf_config_delocalized = pf.ParticleFilterConfig.from_dict(
    {
        "n": 100_000,
        "cov": [0.01, 0.01, 1, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        "noise": [0.0, 0.0, 0.0, 1.75, 1.75, 0.01, 0.001, 0.001, 0.001, 0.1],
        "input_config": pf.ParticleFilterInputConfig.VELOCITY,
        "measurement_config": [{"name": "bathymetry", "std": 100}],
    }
)

pf_config_localized = pf.ParticleFilterConfig.from_dict(
    {
        "n": 100_000,
        "cov": [15 / (1852 * 60), 15 / (1852 * 60), 1, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        "noise": [0.0, 0.0, 0.0, 1.75, 1.75, 0.01, 0.001, 0.001, 0.001, 0.1],
        "input_config": pf.ParticleFilterInputConfig.VELOCITY,
        "measurement_config": [{"name": "bathymetry", "std": 100}],
    }
)

db = dbmgr.DatabaseManager("../.db")


def save_simulation_results(
    filename: str, pf_result: pd.DataFrame, trajectory: pd.DataFrame, trajectory_sd: pd.DataFrame
):
    # Create an HDF5 file
    with h5py.File(f"{filename}.h5", "w") as f:
        # Save the result dataframe
        result_group = f.create_group("result")
        for column in pf_result.columns:
            result_group.create_dataset(column, data=pf_result[column].values)

        # Save the feedback.trajectory dataframe
        trajectory_group = f.create_group("trajectory")
        for column in trajectory.columns:
            trajectory_group.create_dataset(column, data=trajectory[column].values)

        # Save the feedback.trajectory_sd dataframe
        trajectory_sd_group = f.create_group("trajectory_sd")
        for column in trajectory_sd.columns:
            trajectory_sd_group.create_dataset(column, data=trajectory_sd[column].values)


def load_simulation_results(filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Open the HDF5 file
    with h5py.File(filename, "r") as f:
        # Load the result dataframe
        result_data = {column: f["result"][column][:] for column in f["result"]}
        result = pd.DataFrame(result_data)

        # Load the feedback.trajectory dataframe
        trajectory_data = {column: f["trajectory"][column][:] for column in f["trajectory"]}
        trajectory = pd.DataFrame(trajectory_data)

        # Load the feedback.trajectory_sd dataframe
        trajectory_sd_data = {column: f["trajectory_sd"][column][:] for column in f["trajectory_sd"]}
        trajectory_sd = pd.DataFrame(trajectory_sd_data)

    distance = hs.haversine_vector(
        trajectory[["lat", "lon"]].to_numpy(), trajectory[["lat", "lon"]].shift(1).to_numpy(), hs.Unit.METERS
    )
    distance = np.cumsum(np.nan_to_num(distance, nan=0))

    # time = trajectory.index.to_numpy()

    result["distance"] = distance
    trajectory["distance"] = distance

    lat_lon_var = result[["lat_var", "lon_var"]].to_numpy()
    planar_variance = hs.haversine_vector(np.zeros_like(lat_lon_var), lat_lon_var, hs.Unit.METERS)
    three_d_variance = np.sqrt(planar_variance**2 + result["alt_var"].to_numpy() ** 2)

    result["planar_variance"] = planar_variance
    result["three_d_variance"] = three_d_variance

    # trajectory_sd["distance"] = distance

    return result, trajectory, trajectory_sd


def process_trajectory(id: int, pf_config: dict, output_dir: str):
    name = db.get_all_trajectories()["source"][id] + "_" + str(id)
    logger.info(f"Processing trajectory {name}")
    try:
        truth = db.get_trajectory(id)
    except Exception as e:
        logger.info(f"Failed to retrieve trajectory {id}. Cause: {e}.")
        return
    try:
        _, feedback = pf.calculate_truth(truth)
        trajectory = feedback.trajectory
    except Exception as e:
        logger.info(f"Failed to calculate truth for trajectory {id}. Cause: {e}.")
        return

    min_lat = truth.lat.min()
    max_lat = truth.lat.max()
    min_lon = truth.lon.min()
    max_lon = truth.lon.max()

    try:
        bathy_map = GeophysicalMap(
            MeasurementType.BATHYMETRY, ReliefResolution.FIFTEEN_SECONDS, min_lon, max_lon, min_lat, max_lat, 0.25
        )
    except Exception as e:
        logger.info(f"Failed to get bathymetry map for trajectory {id}. Cause: {e}.")
        return
    geomaps = {gmt.MeasurementType.BATHYMETRY: bathy_map}
    try:
        result = pf.run_particle_filter(truth, trajectory, geomaps, pf_config)
    except Exception as e:
        logger.info(f"Failed to run particle filter for trajectory {id}. Cause: {e}.")
        return
    try:
        out_path = os.path.join(output_dir, f"{name}")
        save_simulation_results(f"{out_path}", result, trajectory, feedback.trajectory_sd)
    except Exception as e:
        logger.info(f"Failed to save results for trajectory {id}. Cause: {e}.")
        return
    logger.info(f"Finished processing trajectory {name}")


# Define a function to process each trajectory
def process_trajectory_wrapper(bathy_trajectories: pd.DataFrame, pf_config: dict, j: int, output_dir: str):
    try:
        if bathy_trajectories.loc[bathy_trajectories["id"] == j]["duration"] < 3600:
            logger.info(f"Trajectory {j} is too short: {bathy_trajectories.loc['id']['duration']}")
            return
        process_trajectory(j, pf_config, output_dir)
    except Exception as e:
        logger.warning(f"Failed to process trajectory {j} for some unknown reason.")
        logger.warning(e)


# Process all trajectories in the database
def plot_estimate_error(
    trajectory: pd.DataFrame,
    result: pd.DataFrame,
    drift_rate: np.ndarray,
    map_resolution: float = 452,
    figure_size: tuple[int, int] = (12, 6),
    title_str: str = "Particle Filter Estimate Error",
    recovery_offset: float = 0,
) -> plt.Figure:
    drift_rate += recovery_offset
    fig, axes = plt.subplots(2, 1, figsize=figure_size, gridspec_kw={"height_ratios": [3, 1]})

    duration = trajectory.index[-1] * 60
    axes[0].plot(trajectory["distance"] / 1000, drift_rate, "m", label="INS Drift Rate")
    axes[0].fill_between(
        trajectory["distance"] / 1000,
        result["estimate_error"] - result["planar_variance"] / 2,
        result["estimate_error"] + result["planar_variance"] / 2,
        color="k",
        alpha=0.2,
        label="PF planar certainty",
    )
    axes[0].plot(trajectory["distance"] / 1000, result["estimate_error"], "k", label="Estimate error")

    axes[0].fill_between(
        trajectory["distance"] / 1000,
        np.maximum(result["estimate_error"] + result["planar_variance"] / 2, map_resolution),
        drift_rate,
        where=result["estimate_error"] + result["planar_variance"] / 2 <= drift_rate,
        color="magenta",
        alpha=0.3,
        label="Estimate error less than drift",
    )

    axes[0].fill_between(
        trajectory["distance"] / 1000,
        result["estimate_error"] + result["planar_variance"] / 2,
        map_resolution,
        where=result["estimate_error"] + result["planar_variance"] / 2 <= map_resolution,
        color="green",
        alpha=0.3,
        label="Estimate Error less than map resolution",
    )
    axes[0].axhline(452, color="g", linestyle="--", label="Map resolution")
    axes[0].set_xlabel("Distance traveled (km)")
    axes[0].set_ylabel("Error (meters)")
    axes[0].set_xlim(left=0, right=trajectory["distance"].max() / 1000)
    axes[0].set_ylim(bottom=0)
    axes[0].legend()
    axes[0].set_title(f"{title_str} | Duration: {duration / 3600:0.2f} hours")

    # Compress the y-axis of the second plot
    axes[1].set_aspect(aspect="auto", adjustable="datalim")
    axes[1].plot(trajectory["distance"] / 1000, result["planar_variance"], "k", label="2D")
    axes[1].plot(trajectory["distance"] / 1000, result["three_d_variance"], "b", label="3D")
    axes[1].set_xlabel("Distance traveled (km)")
    axes[1].set_ylabel("Estimate Certainty (m)")
    axes[1].set_xlim(left=0, right=trajectory["distance"].max() / 1000)
    axes[1].set_ylim([0, 5000])
    axes[1].legend()
    return fig


def find_estimate_statistic(result: pd.DataFrame, threshold_mask: list | np.ndarray) -> pd.DataFrame:
    """
    Find the periods where the estimate error is below a certain threshold
    """
    position_recovery_periods = m77t.find_periods(threshold_mask)
    recoveries = pd.DataFrame()
    for inds in position_recovery_periods:
        start = result.index[inds[0]] * 60
        end = result.index[inds[1]] * 60
        duration = end - start
        distance = result["distance"].iloc[inds[1]] - result["distance"].iloc[inds[0]]
        start_distance = result["distance"].iloc[inds[0]]
        end_distance = result["distance"].iloc[inds[1]]
        mean_error = result["estimate_error"].iloc[inds[0] : inds[1]].mean()
        median_error = result["estimate_error"].iloc[inds[0] : inds[1]].median()
        recovery = pd.DataFrame(
            {
                "start": [start],
                "end": [end],
                "duration": [duration],
                "mean_error": [mean_error],
                "median_error": [median_error],
                "distance": [distance],
                "start_distance": [start_distance],
                "end_distance": [end_distance],
                "distance_traveled": end_distance - start_distance,
            }
        )
        if duration > 0:
            recoveries = pd.concat([recoveries, recovery])

    return recoveries


def run_post_processing(output_dir: str, title_str: str):
    estimate_error_summary = pd.DataFrame()
    recoveries = pd.DataFrame()
    below_pixel = pd.DataFrame()
    certain_recoveries = pd.DataFrame()
    certain_below_pixel = pd.DataFrame()

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".h5"):
                # filename = os.path.join(root, file).split('.')[0]
                filename = file.split(".")[0]
                result, trajectory, trajectory_std = load_simulation_results(os.path.join(root, f"{filename}.h5"))
                drift_rate = trajectory.index.to_numpy() * 60 * 1852 / (24 * 3600)

                estimate_error_plot = plot_estimate_error(
                    trajectory,
                    result,
                    drift_rate=drift_rate,
                    figure_size=(12, 4),
                    title_str=title_str,  # "De-localized Particle Filter Estimate Error",
                    recovery_offset=1852,
                )
                estimate_error_plot.savefig(os.path.join(root, f"{filename}.png"))
                # estimate_error_plot.close()
                plt.close(estimate_error_plot)
                estimate_error = result["estimate_error"].rename(filename)
                estimate_error.index = result["distance"]
                estimate_error_summary = pd.merge(
                    estimate_error_summary, estimate_error, left_index=True, right_index=True, how="outer"
                )

                recoveries = pd.concat(
                    [recoveries, find_estimate_statistic(result, result["estimate_error"] >= drift_rate)]
                )
                below_pixel = pd.concat([below_pixel, find_estimate_statistic(result, result["estimate_error"] >= 452)])

                certain_recoveries = pd.concat(
                    [certain_recoveries, find_estimate_statistic(result, result["planar_variance"] >= drift_rate)]
                )
                certain_below_pixel = pd.concat(
                    [certain_below_pixel, find_estimate_statistic(result, result["planar_variance"] >= 452)]
                )

    estimate_error_summary.to_csv(os.path.join(output_dir, "estimate_error_summary.csv"))
    recoveries.to_csv(os.path.join(output_dir, "recoveries.csv"))
    below_pixel.to_csv(os.path.join(output_dir, "below_pixel.csv"))
    certain_recoveries.to_csv(os.path.join(output_dir, "certain_recoveries.csv"))
    certain_below_pixel.to_csv(os.path.join(output_dir, "certain_below_pixel.csv"))


def print_summary_data(data: pd.DataFrame) -> str:
    mean_start = timedelta(seconds=data.mean()["start"])
    mean_end = timedelta(seconds=data.mean()["end"])
    mean_duration = timedelta(seconds=data.mean()["duration"])
    out = f"There were a total of {len(data)} position recoveries.\n"
    out += f"Mean recovery start: {mean_start}\n"
    out += f"Mean recovery end: {mean_end}\n"
    out += f"Mean recovery duration: {mean_duration}\n"
    return out


def main():
    logger.info("=========================================")
    logger.info("Beginning new run of bathy particle filter")
    all_trajs = db.get_all_trajectories()
    bathy_trajectories = all_trajs[all_trajs["depth"]]
    # Create the output directory for saving results
    output_dir = "bathy_pf_results"
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, "delocalized"))
        os.makedirs(os.path.join(output_dir, "localized"))
    # Use multiprocessing to process trajectories in parallel
    print("Processing delocalized initialization")
    # Find the number of CPUs available
    cpu_count = mp.cpu_count()
    logger.info(f"Number of CPUs available: {cpu_count}")
    # Find the number of trajectories to process that are longer than 1 hour
    valid_trajectories = bathy_trajectories[bathy_trajectories["duration"] >= 3600]
    logger.info(f"Number of trajectories to process: {len(valid_trajectories)}")
    start = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(
            process_trajectory,
            [
                (id, pf_config_delocalized, os.path.join(output_dir, "delocalized"))
                for id in tqdm(bathy_trajectories["id"])
            ],
        )
    logger.info(f"Finished delocalized run of bathy particle filter. Elapsed time: {time.time() - start}")
    logger.info("=========================================")
    print("Processing localized initialization")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(
            process_trajectory,
            [(id, pf_config_localized, os.path.join(output_dir, "localized")) for id in tqdm(bathy_trajectories["id"])],
        )
    logger.info(f"Finished localized run of bathy particle filter. Elapsed time: {time.time() - start}")
    logger.info("=========================================")
    logger.info("COMPLETED!")
    logger.info("=========================================")
    logger.info("Running post-processing")
    print("Running post-processing")
    run_post_processing(os.path.join(output_dir, "delocalized"), "De-localized Particle Filter Estimate Error")
    run_post_processing(os.path.join(output_dir, "localized"), "Particle Filter Estimate Error")
    logger.info("Finished post-processing")
    print("Finished post-processing")
    print("Simulation complete!")
    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()
