"""
Executable for running the particle filter on bathymetry data
"""

import json
import os
import multiprocessing as mp
import logging

import matplotlib.pyplot as plt

from src.particle_filter import (
    process_particle_filter,
    populate_velocities,
    plot_error,
    plot_estimate,
    summarize_results,
)
from src import process_dataset as pdset

CONFIG_FILE = "config.json"
PLOTS_OUTPUT = ".db/plots"
ANNOTATIONS = {"recovery": 1852, "res": 1852 / 4}
SOURCE_TRAJECTORIES = ".db/parsed.db"

### LOGGER SETUP ##############################################################
# Create a logger
logger = logging.getLogger("bathy_pf")
logger.setLevel(logging.INFO)
# Add error to logger level


# Create a file handler
handler = logging.FileHandler("bathypf.log")
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
    logger.info("Beginning new run of bathy_pf.py")
    tables = pdset.get_tables(SOURCE_TRAJECTORIES)
    bathy_tables = [table for table in tables if "_D_" in table]
    logger.info("Found tables: %s", bathy_tables)
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Loaded config file: %s", CONFIG_FILE)
    if not os.path.exists(os.path.join(PLOTS_OUTPUT, "estimate")):
        os.makedirs(os.path.join(PLOTS_OUTPUT, "estimate"))
    if not os.path.exists(os.path.join(PLOTS_OUTPUT, "errors")):
        os.makedirs(os.path.join(PLOTS_OUTPUT, "errors"))
    print("Begginnig processing")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(
            multiprocessing_wrapper,
            [(table, config, ANNOTATIONS) for table in bathy_tables],
            chunksize=2,
        )
    # for table in bathy_tables:
    #    multiprocessing_wrapper(table, config, ANNOTATIONS)

    # Second linear pass to check for memory errors
    completed_tables = pdset.get_tables(".db/results.db")
    for table in bathy_tables:
        if table not in completed_tables:
            multiprocessing_wrapper(table, config, ANNOTATIONS)

    results_tables = pdset.get_tables(".db/results.db")
    output_path = os.path.join(PLOTS_OUTPUT, "summary.csv")
    for table in results_tables:
        summary = summarize_results(table, ".db/results.db")
        summary.to_csv(
            output_path,
            mode="a",
            header=(not os.path.exists(output_path)),
        )


def multiprocessing_wrapper(table, config, annotations):
    """
    multiprocessing wrapper for process_particle_filter
    """

    logger.info("Starting processing for table: %s", table)
    df = pdset.table_to_df(SOURCE_TRAJECTORIES, table)
    logger.info("Loaded table: %s", table)
    df = populate_velocities(df)
    logger.info("Begining run: %s", table)
    try:
        results, geo_map = process_particle_filter(df, config)
    # Catch hdf5 errors
    except OSError as e:
        logger.error("Error processing table: %s", table)
        logger.error(e.with_traceback())
        # traceback.print_exc(file)
        return
    # catch other errors
    except Exception as e:
        logger.error("Error processing table: %s", table)
        logger.error(e.with_traceback())
        return
    logger.info("%s run complete", table)
    pdset.save_dataset(
        [results],
        [table],
        output_location=".db",
        output_format="db",
        dataset_name="results",
    )
    logger.info("Saved results for table: %s", table)
    fig, _ = plot_estimate(geo_map, results)
    logger.info("Plotting estimate for table: %s", table)
    fig.savefig(os.path.join(PLOTS_OUTPUT, "estimate", f"{table}_estimate.png"))
    plt.close()
    logger.info("Estimate plot saved for table: %s", table)
    fig, _ = plot_error(results, annotations=annotations)
    logger.info("Plotting error for table: %s", table)
    fig.savefig(os.path.join(PLOTS_OUTPUT, "errors", f"{table}_error.png"))
    plt.close()
    logger.info("Error plot saved for table: %s", table)
    logger.info("Finished processing for table: %s", table)


if __name__ == "__main__":
    main()
