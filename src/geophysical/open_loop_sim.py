"""
Main CLI Tool and eventual API for the geophysical navigation simulation module.
This module should take over from the original bathy_pf.py and gravity_pf.py scripts
and provide a more robust and flexible simulation environment.
"""

import json
import logging
import multiprocessing as mp
import os

from matplotlib import pyplot as plt
from pandas import read_csv, to_timedelta, DataFrame

from .db_tools import get_tables, save_dataset, table_to_df
from .particle_filter import (
    plot_error,
    plot_estimate,
    populate_velocities,
    process_particle_filter,
    summarize_results,
)

# Potential CLI inputs
CONFIG_FILE = "./scripts/config.json"
SOURCE_TRAJECTORIES = ".db/parsed.db"
DATA_TYPE = "gravity"
ANNOTATIONS = {"recovery": 1852, "res": 1852}

# GLOBALS
PLOTS_OUTPUT = f".db/plots_{DATA_TYPE}"
RESULTS_DB = ".db/results_{DATA_TYPE}.db"

# --- LOGGER SETUP ---
# Create a logger
logger = logging.getLogger(f"{DATA_TYPE}_pf")
logger.setLevel(logging.INFO)
# Create a file handler
handler = logging.FileHandler(f"{DATA_TYPE}pf.log")
handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter("%(asctime)s || %(message)s")
handler.setFormatter(formatter)
# Add the handlers to the logger
logger.addHandler(handler)


def main():
    """
    Main function
    """
    logger.info("=========================================")
    logger.info("Beginning new run of %s_pf.py", DATA_TYPE)
    tables = get_tables(SOURCE_TRAJECTORIES)

    # Validate data type string
    gravity_tables = [table for table in tables if "_G_" in table]
    logger.info("Found tables: %s", gravity_tables)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Loaded config file: %s", CONFIG_FILE)
    if not os.path.exists(os.path.join(PLOTS_OUTPUT, "estimate")):
        os.makedirs(os.path.join(PLOTS_OUTPUT, "estimate"))
    if not os.path.exists(os.path.join(PLOTS_OUTPUT, "errors")):
        os.makedirs(os.path.join(PLOTS_OUTPUT, "errors"))

    # Get completed tables
    logger.info("Checking for completed tables")
    try:
        completed_tables = get_tables(RESULTS_DB)
        remaining_tables = [table for table in gravity_tables if table not in completed_tables]
    except FileNotFoundError:
        completed_tables = []
        remaining_tables = gravity_tables
    while remaining_tables:  # empty list check would be not remaining_tables
        logger.info("Found completed tables: %s", completed_tables)
        logger.info("Found remaining tables: %s", remaining_tables)
        logger.info("Beginning multiprocessing")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            try:
                pool.starmap(
                    multiprocessing_wrapper,
                    [(table, config, ANNOTATIONS) for table in remaining_tables],
                )
            except OSError:
                logger.error("Error in multiprocessing... probably a memory error")
            # except Exception:
            #     logger.error("Error in multiprocessing... general error")
        logger.info("Finished multiprocessing")
        logger.info("Double checking to make sure all tables are complete")
        # Get completed tables
        completed_tables = get_tables(RESULTS_DB)
        remaining_tables = [table for table in gravity_tables if table not in completed_tables]
        logger.info("Found remaining tables: %s", remaining_tables)
    logger.info("Beginning second linear pass")
    # Second linear pass to check for memory errors
    completed_tables = get_tables(RESULTS_DB)
    for table in gravity_tables:
        if table not in completed_tables:
            multiprocessing_wrapper(table, config, ANNOTATIONS)
    logger.info("Summizing results")
    results_tables = get_tables(RESULTS_DB)
    output_path = os.path.join(PLOTS_OUTPUT, "summary.csv")
    for table in results_tables:
        df = table_to_df(RESULTS_DB, table)
        summary = summarize_results(table, df, 1852)
        summary.to_csv(
            output_path,
            mode="a",
            header=(not os.path.exists(output_path)),
        )
    logger.info("Finished summarizing results. Process complete. Executing post processing.")

    post_process_batch(f"{PLOTS_OUTPUT}/summary.csv", RESULTS_DB)


def post_process_batch(
    summary_file: str,
    results_db: str,
    output_location: str = ".db/plots_gravity/summary.txt",
    pixel_resolution: float = 452,
) -> None:
    """
    Run post processing on then entire batch of results to get the overall conclusions and results

    Parameters:
    -----------
    :param summary_file: the filepath to the summary file
    :type summary_file: str
    :param results_db: the filepath to the results database
    :type results_db: str
    :param output_location: the filepath to the output location
    :type output_location: str
    :param pixel_resolution: the pixel resolution in meters
    :type pixel_resolution: float

    Returns:
    --------
    :returns: None

    """
    results_tables = get_tables(results_db)
    # Load and pre-process the summary file
    summary = read_csv(
        summary_file,
        header=0,
        dtype={
            "": int,
            "name": str,
            "start": str,
            "stop": str,
            "duration": str,
            "average_error": float,
            "max_error": float,
            "min_error": float,
        },
    )
    summary["num"] = summary["Unnamed: 0"]
    summary = summary.drop(columns=["Unnamed: 0"])
    summary["start"] = to_timedelta(summary["start"])
    summary["duration"] = to_timedelta(summary["duration"])

    # Get pixel level results
    pixel = summary.loc[summary["min error"] <= pixel_resolution]
    # Find first fixes for each trajectory
    first = summary.loc[summary["num"] == 0]
    write_summary_data_file(output_location, results_tables, summary, pixel, first)


def _check_for_missing_tables(source_tables: list[DataFrame], test_tables: list[DataFrame]) -> int:
    """
    Check to see if tables from the source are missing from the test tables.
    """
    missing = []
    for table in source_tables:
        if table not in test_tables["name"].values:
            missing.append(table)
    return len(missing)


def write_summary_data_file(
    output_location: str,
    results_tables: DataFrame,
    summary: DataFrame,
    pixel: DataFrame,
    first: DataFrame,
):
    """
    Writes the general summary of the experiment to a file
    """
    total = len(results_tables)
    num_recoveries = total - _check_for_missing_tables(results_tables, summary)
    below_pixel_fixes = total - _check_for_missing_tables(results_tables, pixel)

    with open(output_location, "w", encoding="utf-8") as f:
        # Summary
        f.write(f"There are {total} total trajectories.\n")
        f.write("----- Summary -----\n")
        f.write(
            f"At least one position estimate below drift error in {num_recoveries} "
            + f"({num_recoveries / total :0.4f}) trajectories.\n"
        )
        f.write(f"Mean duration:\t  {summary['duration'].mean()}\n")
        f.write(f"Median duration:\t{summary['duration'].median()}\n")
        f.write(f"Minimum duration: {summary['duration'].min()}\n")
        f.write(f"Maximum duration: {summary['duration'].max()}.\n")
        f.write(f"Mean error:\t     {summary['average_error'].mean()}\n")
        f.write(f"Median error:\t   {summary['average_error'].median()}.\n")
        f.write(f"Minimum error:    {summary['average_error'].min()}\n")
        f.write(f"Maximum error:    {summary['average_error'].max()}\n")
        # First position recovery
        f.write("---- First Position Recovery ----\n")
        f.write(f"Mean start: {first['start'].mean()}\n")
        f.write(f"Median start: {first['start'].median()}\n")
        f.write(f"Min start: {first['start'].min()}\n")
        f.write(f"Max start: {first['start'].max()}\n")
        f.write(f"Mean duration: {first['duration'].mean()}\n")
        f.write(f"Median duration: {first['duration'].median()}\n")
        f.write(f"Minimum duration {first['duration'].min()}\n")
        f.write(f"Maximum duration {first['duration'].max()}\n")
        f.write(f"Mean error: {first['average_error'].mean()}\n")
        f.write(f"Median error {first['average_error'].median()}.\n")
        f.write(f"Min error: {first['average_error'].min()}\n")
        f.write(f"Max error {first['average_error'].max()}.\n")
        # Below pixel resolution fixes
        f.write("---- Pixel Level ----\n")
        f.write(f"There are {len(pixel)} total below pixel resolution fixes.\n")
        f.write(
            f"At least one estimate below pixel resolution in {below_pixel_fixes}"
            + f"({below_pixel_fixes/total :0.4f}) trajectories.\n"
        )
        f.write(f"Mean duration:\t  {pixel['duration'].mean()}\n")
        f.write(f"Median duration:\t{pixel['duration'].median()}\n")
        f.write(f"Minimum duration: {pixel['duration'].min()}\n")
        f.write(f"Maximum duration: {pixel['duration'].max()}.\n")
        f.write(f"Mean error:\t     {pixel['average_error'].mean()}\n")
        f.write(f"Median error:\t   {pixel['average_error'].median()}.\n")
        f.write(f"Minimum error     {pixel['average_error'].min()}\n")
        f.write(f"Maximum error     {pixel['average_error'].max()}\n")


def processing_wrapper(
    table: str, config: dict, annotations: dict, source_trajectories_location: str, output_plots_location: str
) -> None:
    """
    multiprocessing wrapper for process_particle_filter
    """

    logger.info("Starting processing for table: %s", table)
    df = table_to_df(source_trajectories_location, table)
    logger.info("Loaded table: %s", table)
    df = populate_velocities(df)
    logger.info("Begining run: %s", table)
    try:
        results, geo_map = process_particle_filter(df, config, map_type="gravity", map_resolution="01m")
    # Catch hdf5 errors
    except OSError as e:
        logger.error("Error processing table: %s", table)
        logger.error(e.with_traceback())
        # traceback.print_exc(file)
        return
    # catch other errors
    # except Exception as e:
    #     logger.error("Error processing table: %s", table)
    #     logger.error(e.with_traceback())
    #     return
    logger.info("%s run complete", table)
    save_dataset(
        [results],
        [table],
        output_location=".db",
        output_format="db",
        dataset_name="results_gravity",
    )
    logger.info("Saved results for table: %s", table)
    fig, _ = plot_estimate(geo_map, results, measurment_type=config["measurment_type"])
    logger.info("Plotting estimate for table: %s", table)
    fig.savefig(os.path.join(output_plots_location, "estimate", f"{table}_estimate.png"))
    plt.close()
    logger.info("Estimate plot saved for table: %s", table)
    fig, _ = plot_error(results, annotations=annotations)
    logger.info("Plotting error for table: %s", table)
    fig.savefig(os.path.join(output_plots_location, "errors", f"{table}_error.png"))
    plt.close()
    logger.info("Error plot saved for table: %s", table)
    logger.info("Finished processing for table: %s", table)


if __name__ == "__main__":
    main()
