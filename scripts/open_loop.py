import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pyins import transform
from geophysical import gmt_toolbox as gmt
from data_management import dbmgr
from geophysical import particle_filter as pf
from geophysical.gmt_toolbox import GeophysicalMap, MeasurementType, ReliefResolution
import haversine as hs
import h5py
from tqdm import tqdm
import logging
import multiprocessing as mp

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
    return result, trajectory, trajectory_sd


def process_trajectory(j: int, pf_config: dict, output_dir: str):
    name = db.get_all_trajectories()["source"][j] + "_" + str(j)
    logger.info(f"Processing trajectory {name}")
    try:
        truth = db.get_trajectory(j)
    except:
        logger.info(f"Failed to get trajectory {j}")
        return
    try:
        _, feedback = pf.calculate_truth(truth)
        trajectory = feedback.trajectory
    except:
        logger.info(f"Failed to calculate truth for trajectory {j}")
        return

    min_lat = truth.lat.min()
    max_lat = truth.lat.max()
    min_lon = truth.lon.min()
    max_lon = truth.lon.max()

    try:
        bathy_map = GeophysicalMap(
            MeasurementType.BATHYMETRY, ReliefResolution.FIFTEEN_SECONDS, min_lon, max_lon, min_lat, max_lat, 0.25
        )
    except:
        logger.info(f"Failed to get bathymetry map for trajectory {j}")
        return
    geomaps = {gmt.MeasurementType.BATHYMETRY: bathy_map}
    try:
        result = pf.run_particle_filter(truth, trajectory, geomaps, pf_config)
    except:
        logger.info(f"Failed to run particle filter for trajectory {j}")
        return
    try:
        out_path = os.path.join(output_dir, f"{name}")
        save_simulation_results(f"{out_path}", result, trajectory, feedback.trajectory_sd)
    except:
        logger.info(f"Failed to save results for trajectory {j}")
        return
    logger.info(f"Finished processing trajectory {name}")


# Define a function to process each trajectory
def process_trajectory_wrapper(bathy_trajectories: pd.DataFrame, pf_config: dict, j: int, output_dir: str):
    if bathy_trajectories.loc[bathy_trajectories.index[j], "duration"] < 3600:
        logger.info(f"Trajectory {j} is too short: {bathy_trajectories.iloc[j]['duration']}")
        return
    try:
        process_trajectory(j, pf_config, output_dir)
    except Exception as e:
        logger.warning("Failed to process trajectory {j} for some unknown reason.")
        logger.warning(e)


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
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(
            process_trajectory_wrapper,
            [
                (bathy_trajectories, pf_config_delocalized, j, os.path.join(output_dir, "delocalized"))
                for j in tqdm(bathy_trajectories["id"])
            ],
        )
    logger.info("Finished delocalized run of bathy particle filter")
    logger.info("=========================================")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(
            process_trajectory_wrapper,
            [
                (bathy_trajectories, pf_config_localized, j, os.path.join(output_dir, "localized"))
                for j in tqdm(bathy_trajectories["id"])
            ],
        )
    logger.info("Finished localized run of bathy particle filter")
    logger.info("=========================================")
    logger.info("COMPLETED!")


if __name__ == "__main__":
    main()
