import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from geophysical import gmt_toolbox as gmt
from data_management import dbmgr, m77t
from geophysical import particle_filter as pf
from geophysical.gmt_toolbox import GeophysicalMap, MeasurementType, GravityResolution
import haversine as hs
import h5py
from tqdm import tqdm
import logging
import multiprocessing as mp
from datetime import timedelta
import time
from anglewrapper import wrap

logger = logging.getLogger("grav_pf")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("grav_pf.log")
handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter("%(asctime)s || %(message)s")
handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)

N = 10

pf_config_delocalized = pf.ParticleFilterConfig.from_dict(
    {
        "n": N,
        "cov": [0.01, 0.01, 1.0, 0.10, 0.10, 0.10, 0.000, 0.000, 0, 10],
        "noise": [0.00, 0.00, 0.0, 1.75, 1.75, 0.01, 0.001, 0.001, 0.001, 1],
        "input_config": pf.ParticleFilterInputConfig.VELOCITY,
        "measurement_config": [{"name": "GRAVITY", "std": 5}],
    }
)

pf_config_localized = pf.ParticleFilterConfig.from_dict(
    {
        "n": N,
        "cov": [15 / (1852 * 60), 15 / (1852 * 60), 1, 0.1, 0.1, 0.1, 0, 0, 0, 10],
        "noise": [0.0, 0.0, 0.0, 1.75, 1.75, 0.01, 0.001, 0.001, 0.001, 1],
        "input_config": pf.ParticleFilterInputConfig.VELOCITY,
        "measurement_config": [{"name": "GRAVITY", "std": 5}],
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

    try:
        distance = hs.haversine_vector(
            trajectory[["lat", "lon"]].to_numpy(), trajectory[["lat", "lon"]].shift(1).to_numpy(), hs.Unit.METERS
        )
    except ValueError as ve:
        logger.info(f"Failed to calculate distance. Cause: {ve}. Attempting to calculate through loop...")
        distance = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            try:
                ll0 = trajectory[["lat", "lon"]].iloc[i - 1].to_numpy()
                ll1 = trajectory[["lat", "lon"]].iloc[i].to_numpy()
                # Wrap the longitude
                ll0[1] = wrap.to_180(ll0[1])
                ll1[1] = wrap.to_180(ll1[1])
                distance[i] = hs.haversine(ll0, ll1, hs.Unit.METERS)
            except ValueError as ve:
                logger.info(f"Failed to calculate distance for index {i}. Cause: {ve}. Setting distance to NaN.")
                distance[i] = np.nan
    distance = np.cumsum(np.nan_to_num(distance, nan=0))

    # time = trajectory.index.to_numpy()

    result["distance"] = distance
    trajectory["distance"] = distance

    lat_lon_var = np.sqrt(result[["lat_var", "lon_var"]].to_numpy())
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
        grav_map = GeophysicalMap(
            MeasurementType.GRAVITY, GravityResolution.ONE_MINUTE, min_lon, max_lon, min_lat, max_lat, 0.25
        )
    except Exception as e:
        logger.info(f"Failed to get GRAVITY map for trajectory {id}. Cause: {e}.")
        return
    geomaps = {gmt.MeasurementType.GRAVITY: grav_map}
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


def plot_estimate_error(
    trajectory: pd.DataFrame,
    result: pd.DataFrame,
    drift_rate: np.ndarray,
    map_resolution: float = 452,
    figure_size: tuple[int, int] = (14, 6),
    title_str: str = "Particle Filter Estimate Error",
    recovery_offset: float = 0,
) -> plt.Figure:
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
        result["estimate_error"] + result["planar_variance"] / 2,
        drift_rate,
        where=result["estimate_error"] + result["planar_variance"] / 2 <= drift_rate,
        color="magenta",
        alpha=0.3,
        label="Estimate error less than drift",
    )
    # TODO: #79 Refactor this to remove plotting the map resolution line, as it isn't a useful metric for magnetic and gravity data
    # axes[0].fill_between(
    #     trajectory["distance"] / 1000,
    #     result["estimate_error"] + result["planar_variance"] / 2,
    #     map_resolution,
    #     where=result["estimate_error"] + result["planar_variance"] / 2 <= map_resolution,
    #     color="green",
    #     alpha=0.3,
    #     label="Estimate Error less than map resolution",
    # )
    # axes[0].axhline(map_resolution, color="g", linestyle="--", label="Map resolution")
    axes[0].set_xlabel("Distance traveled (km)")
    axes[0].set_ylabel("Error (meters)")
    axes[0].set_xlim(left=0, right=trajectory["distance"].max() / 1000)
    axes[0].set_ylim(bottom=0, top=max(drift_rate) * 1.5)
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
        mean_variance = result["planar_variance"].iloc[inds[0] : inds[1]].mean()
        median_variance = result["planar_variance"].iloc[inds[0] : inds[1]].median()
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
                "mean_variance": [mean_variance],
                "median_variance": [median_variance],
            }
        )
        if duration > 0:
            recoveries = pd.concat([recoveries, recovery])

    return recoveries


def run_post_processing(
    output_dir: str, title_str: str, recovery_offset: float = 0.0, map_resolution: float = 452
) -> None:
    trajectory_summary = pd.DataFrame(
        columns=[
            "start",
            "end",
            "duration",
            "mean_error",
            "median_error",
            "distance",
            "start_distance",
            "end_distance",
            "distance_traveled",
        ]
    )
    # This is really a post processing scheme that is only valid for delocalized particle filters at least in terms of summary data. Should do localized data as a percentage of the drift rate.
    estimate_error_summary_distance = pd.DataFrame()
    estimate_error_summary_time = pd.DataFrame()
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
                drift_rate = trajectory.index.to_numpy() * 60 * 1852 / (24 * 3600) + recovery_offset
                trajectory_summary.loc[filename] = [
                    np.nan,
                    np.nan,
                    (trajectory.index[-1] - trajectory.index[0]) * 60,
                    np.nan,
                    np.nan,
                    result["distance"].iloc[-1],
                    np.nan,
                    np.nan,
                    np.nan,
                ]

                estimate_error_plot = plot_estimate_error(
                    trajectory,
                    result,
                    drift_rate=drift_rate,
                    map_resolution=map_resolution,
                    figure_size=(12, 4),
                    title_str=title_str,  # "De-localized Particle Filter Estimate Error",
                    recovery_offset=recovery_offset,
                )
                estimate_error_plot.savefig(os.path.join(root, f"{filename}.png"))
                # estimate_error_plot.close()
                plt.close(estimate_error_plot)
                estimate_error = result["estimate_error"].rename(filename)
                estimate_error.index = result["distance"]
                estimate_error_summary = pd.merge(
                    estimate_error_summary_distance, estimate_error, left_index=True, right_index=True, how="outer"
                )
                estimate_error_time = result["estimate_error"].rename(filename)
                estimate_error_time.index = trajectory.index * 60
                estimate_error_summary_time = pd.merge(
                    estimate_error_summary_time, estimate_error_time, left_index=True, right_index=True, how="outer"
                )
                recoveries = pd.concat(
                    [recoveries, find_estimate_statistic(result, result["estimate_error"] >= drift_rate)]
                )
                below_pixel = pd.concat(
                    [below_pixel, find_estimate_statistic(result, result["estimate_error"] >= map_resolution)]
                )

                certain_recoveries = pd.concat(
                    [certain_recoveries, find_estimate_statistic(result, result["planar_variance"] >= drift_rate)]
                )
                certain_below_pixel = pd.concat(
                    [certain_below_pixel, find_estimate_statistic(result, result["planar_variance"] >= map_resolution)]
                )

    trajectory_summary.to_csv(os.path.join(output_dir, "trajectory_summary.csv"))
    estimate_error_summary_distance.to_csv(os.path.join(output_dir, "estimate_error_summary_distance.csv"))
    estimate_error_summary_time.to_csv(os.path.join(output_dir, "estimate_error_summary_time.csv"))
    recoveries.to_csv(os.path.join(output_dir, "recoveries.csv"))
    below_pixel.to_csv(os.path.join(output_dir, "below_pixel.csv"))
    certain_recoveries.to_csv(os.path.join(output_dir, "certain_recoveries.csv"))
    certain_below_pixel.to_csv(os.path.join(output_dir, "certain_below_pixel.csv"))

    # Mean summary of recoveries
    summary = pd.DataFrame(columns=recoveries.columns)
    summary.loc["trajectories"] = trajectory_summary.mean()
    summary.loc["recoveries"] = recoveries.mean()
    summary.loc["below_pixel"] = below_pixel.mean()
    summary.loc["certain_recoveries"] = certain_recoveries.mean()
    summary.loc["certain_below_pixel"] = certain_below_pixel.mean()
    summary.to_csv(os.path.join(output_dir, "summary.csv"))


def print_summary_data(data: pd.DataFrame) -> str:
    mean_start = timedelta(seconds=data.mean()["start"])
    mean_end = timedelta(seconds=data.mean()["end"])
    mean_duration = timedelta(seconds=data.mean()["duration"])
    out = f"There were a total of {len(data)} position recoveries.\n"
    out += f"Mean recovery start: {mean_start}\n"
    out += f"Mean recovery end: {mean_end}\n"
    out += f"Mean recovery duration: {mean_duration}\n"
    return out


def summarize_results(base_dir: str, initialization: str) -> None:
    """
    Writes a set of summarizing statements about the results of the particle filter to a text file.
    Output is saved in the same directory as the results.

    Parameters
    ----------
    base_dir : str
        The base directory where the results are stored.
    initialization : str
        The initialization type of the particle filter. Should correspond to a folder within `base_dir`.

    Returns
    -------
    None
    """
    trajectories = pd.read_csv(os.path.join(base_dir, initialization, "trajectory_summary.csv"), index_col=0)
    recoveries = pd.read_csv(os.path.join(base_dir, initialization, "recoveries.csv"), index_col=0)
    recoveries_below_pixel = pd.read_csv(os.path.join(base_dir, initialization, "below_pixel.csv"), index_col=0)
    certain_recoveries = pd.read_csv(os.path.join(base_dir, initialization, "certain_recoveries.csv"), index_col=0)
    certain_below_pixel = pd.read_csv(os.path.join(base_dir, initialization, "certain_below_pixel.csv"), index_col=0)
    # summary = pd.read_csv(os.path.join(base_dir, initialization, "summary.csv"), index_col=0)
    estimate_error_summary = pd.read_csv(
        os.path.join(base_dir, initialization, "estimate_error_summary.csv"), index_col=0
    )
    number_trajectories = len(trajectories)
    with open(os.path.join(base_dir, initialization, "summary_statements.txt"), "w") as f:
        f.write(f"Number of trajectories: {number_trajectories}\n")
        f.write(f"Number of recoveries: {len(recoveries)} | {len(recoveries) / number_trajectories:.2f}\n")
        f.write(
            f"Number of recoveries below pixel: {len(recoveries_below_pixel)} | {len(recoveries_below_pixel) / number_trajectories:.2f}\n"
        )
        f.write(
            f"Number of certain recoveries: {len(certain_recoveries)} | {len(certain_recoveries) / number_trajectories:.2f}\n"
        )
        f.write(
            f"Number of certain recoveries below pixel: {len(certain_below_pixel)} | {len(certain_below_pixel) / number_trajectories:.2f}\n"
        )
        f.write("----------------------------------------\n")
        f.write(f"Mean trajectory duration: {str(timedelta(seconds=trajectories['duration'].mean()))}\n")
        f.write(
            f"Mean trajectory distance: {trajectories['distance'].mean() / 1000:.2f} km ({trajectories['distance'].mean() / 1852:.2f} nmi)\n"
        )
        f.write(
            f"Min trajectory distance: {trajectories['distance'].min() / 1000:.2f} km ({trajectories['distance'].min() / 1852:.2f} nmi)\n"
        )
        f.write(
            f"Max trajectory distance: {trajectories['distance'].max() / 1000:.2f} km ({trajectories['distance'].max() / 1852:.2f} nmi)\n"
        )
        f.write(f"Min trajectory duration: {str(timedelta(seconds=trajectories['duration'].min()))}\n")
        f.write(f"Max trajectory duration: {str(timedelta(seconds=trajectories['duration'].max()))}\n")
        f.write("----------------------------------------\n")
        f.write(f"Mean recovery duration: {str(timedelta(seconds=recoveries['duration'].mean()))}\n")
        f.write(
            f"Mean recovery distance: {recoveries['distance_traveled'].mean() / 1000:.2f} km ({recoveries['distance_traveled'].mean() / 1852:.2f} nmi)\n"
        )
        f.write(f"Mean recovery error: {recoveries['mean_error'].mean():.2f} meters\n")
        f.write(f"Median recovery error: {recoveries['median_error'].mean():.2f} meters\n")
        f.write("----------------------------------------\n")
        f.write(
            f"Mean recovery below pixel duration: {str(timedelta(seconds=recoveries_below_pixel['duration'].mean()))}\n"
        )
        f.write(
            f"Mean recovery below pixel distance: {recoveries_below_pixel['distance_traveled'].mean() / 1000:.2f} km ({recoveries_below_pixel['distance_traveled'].mean() / 1852:.2f} nmi)\n"
        )
        f.write(f"Mean recovery below pixel error: {recoveries_below_pixel['mean_error'].mean():.2f} meters\n")
        f.write(f"Median recovery below pixel error: {recoveries_below_pixel['median_error'].mean():.2f} meters\n")
        f.write("----------------------------------------\n")
        f.write(f"Mean certain recovery duration: {str(timedelta(seconds=certain_recoveries['duration'].mean()))}\n")
        f.write(
            f"Mean certain recovery distance: {certain_recoveries['distance_traveled'].mean() / 1000:.2f} km ({certain_recoveries['distance_traveled'].mean() / 1852:.2f} nmi)\n"
        )
        f.write(f"Mean certain recovery error: {certain_recoveries['mean_error'].mean():.2f} meters\n")
        f.write(f"Median certain recovery error: {certain_recoveries['median_error'].mean():.2f} meters\n")
        f.write("----------------------------------------\n")
        f.write(
            f"Mean certain recovery below pixel duration: {str(timedelta(seconds=certain_below_pixel['duration'].mean()))}\n"
        )
        f.write(
            f"Mean certain recovery below pixel distance: {certain_below_pixel['distance_traveled'].mean() / 1000:.2f} km ({certain_below_pixel['distance_traveled'].mean() / 1852:.2f} nmi)\n"
        )
        f.write(f"Mean certain recovery below pixel error: {certain_below_pixel['mean_error'].mean():.2f} meters\n")
        f.write(f"Median certain recovery below pixel error: {certain_below_pixel['median_error'].mean():.2f} meters\n")


def main():
    logger.info("=========================================")
    # logger.info("Beginning new run of GRAVITY particle filter")
    # all_trajs = db.get_all_trajectories()
    # grav_trajectories = all_trajs.loc[(all_trajs["duration"] >= 3600) & (all_trajs["freeair"])]
    # Create the output directory for saving results
    output_dir = "grav_pf_results"
    # if not os.path.exists(output_dir):
    #     os.makedirs(os.path.join(output_dir, "delocalized"))
    #     os.makedirs(os.path.join(output_dir, "localized"))
    # # Use multiprocessing to process trajectories in parallel
    # print("Processing delocalized initialization")
    # # Find the number of CPUs available
    # cpu_count = mp.cpu_count()
    # logger.info(f"Number of CPUs available: {cpu_count}")
    # logger.info(f"Number of trajectories to process: {len(grav_trajectories)}")
    # start = time.time()
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     pool.starmap(
    #         process_trajectory,
    #         [
    #             (id, pf_config_delocalized, os.path.join(output_dir, "delocalized"))
    #             for id in tqdm(grav_trajectories["id"])
    #         ],
    #     )
    # logger.info(f"Finished delocalized run of mag particle filter. Elapsed time: {time.time() - start}")
    # logger.info("=========================================")
    # print("Processing localized initialization")
    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     pool.starmap(
    #         process_trajectory,
    #         [(id, pf_config_localized, os.path.join(output_dir, "localized")) for id in tqdm(grav_trajectories["id"])],
    #     )
    # logger.info(f"Finished localized run of mag particle filter. Elapsed time: {time.time() - start}")
    # logger.info("=========================================")
    # logger.info("COMPLETED!")
    # logger.info("=========================================")
    # logger.info("Running post-processing")
    # print("Running post-processing")
    run_post_processing(
        os.path.join(output_dir, "delocalized"),
        "De-localized Particle Filter Estimate Error",
        recovery_offset=1852.0,
        map_resolution=1852 * 60,
    )
    run_post_processing(
        os.path.join(output_dir, "localized"), "Particle Filter Estimate Error", map_resolution=1852 * 60
    )
    logger.info("Finished post-processing")
    print("Finished post-processing")
    print("Simulation complete!")
    logger.info("Simulation complete!")


if __name__ == "__main__":
    main()
