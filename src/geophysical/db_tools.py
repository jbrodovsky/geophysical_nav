"""
Sqlite3 database utility functions.
"""

import os
import sqlite3

import pandas as pd


# --- Database Utility Functions
def get_tables(db_path: str):
    """
    Get the names of all tables in a database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    conn.close()

    # The result is a list of tuples. Convert it to a list of strings.
    tables = [table[0] for table in tables]

    return tables


def table_to_df(db_path: str, table_name: str):
    """
    Load and convert a table in a database to a pandas data frame. Thin wrapper
    around pandas.read_sql_query that specifies the default data format for
    each column.
    """
    with sqlite3.connect(db_path) as conn:
        data = pd.read_sql_query(
            f"SELECT * FROM '{table_name}'",
            conn,
            index_col="TIME",
        )
        data.index = pd.to_datetime(data.index)

    return data


def df_to_table(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Write a pandas data frame to a table in a database. Thin wrapper around
    pandas.DataFrame.to_sql that specifies the default data format for each
    column.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=True,
                index_label="TIME",
            )
    except sqlite3.OperationalError as e:
        print(e)

    return None


def save_dataset(
    data: list[pd.DataFrame],
    names: list[str],
    output_location: str,
    output_format: str = "db",
    dataset_name: str = "parsed",
) -> None:
    """
    Used to save the parsed MGD77T data. Data is either saved to a folder as
    .csv or to a single .db file. Default is .db.

    Parameters
    ----------
    :param data: list of dataframes containing the parsed data
    :type data: list of pandas.DataFrame
    :param names: list of names of the files
    :type names: list of strings
    :param output_location: The file path to the root folder to search.
    :type output_location: STRING
    :param output_format: The format for the output (db or csv).
    :type output_format: STRING
    :param dataset_name: The name of the dataset to be saved.
    :type dataset_name: STRING

    Returns
    -------
    :returns: none
    """

    if output_format == "db":
        for df, name in zip(data, names):
            df_to_table(df, os.path.join(output_location, f"{dataset_name}.db"), name)
    elif output_format == "csv":
        for i, df in enumerate(data):
            df.to_csv(os.path.join(output_location, f"{names[i]}.csv"))

    else:
        raise NotImplementedError(
            f"Output format {output_format} not recognized. Please choose from the " + "following: db, csv"
        )
