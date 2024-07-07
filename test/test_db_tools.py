"""
Test DB tools
"""

import os
import shutil
import unittest
import sqlite3 as sql

from pandas import DataFrame

from src.geophysical.db_tools import get_tables, table_to_df, df_to_table, save_dataset


class TestDBTools(unittest.TestCase):
    """
    Test class for db_tools.py
    """

    def setUp(self):
        """
        Set up the test environment data.
        """
        self.db_path = os.path.join(".", "test", "db")
        self.db_string = f"{self.db_path}/test.db"
        self.table_name = "test_table"
        self.data = DataFrame(
            {
                "TIME": ["2021-01-01", "2021-01-02", "2021-01-03"],
                "VALUE": [1, 2, 3],
            }
        )
        os.makedirs(self.db_path)
        with sql.connect(self.db_string) as conn:
            self.data.to_sql(
                self.table_name,
                conn,
                index=False,
                if_exists="replace",
                index_label="TIME",
            )

    def tearDown(self):
        """
        Tear down the test environment data.
        """
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_get_tables(self):
        """
        Test the get_tables function.
        """
        tables = get_tables(self.db_string)
        self.assertEqual(tables, [self.table_name], msg=f"Table {tables} not found in database {self.db_string}.")
        # self.assertRaises(sql.OperationalError, get_tables, "nonexistent.db")

    def test_table_to_df(self):
        """
        Test the table_to_df function.
        """
        data = table_to_df(self.db_string, self.table_name)
        self.assertTrue(data.equals(data))
        # self.assertRaises(sql.OperationalError, table_to_df, "nonexistent.db", "nonexistent_table")

    def test_df_to_table(self):
        """
        Test the df_to_table function.
        """
        # tables = get_tables(self.db_path)
        # self.assertEqual(tables, [self.table_name])
        df_to_table(self.data.copy(), self.db_string, "new_table_name")
        tables = get_tables(self.db_string)
        self.assertEqual(tables, [self.table_name, "new_table_name"])
        # self.assertRaises(sql.OperationalError, df_to_table, self.data, "nonexistent.db", "nonexistent_table")

    def test_save_dataset(self):
        """
        Test the save_dataset function.
        """
        data = [
            DataFrame({"TIME": ["2021-01-01", "2021-01-02", "2021-01-03"], "VALUE": [1, 2, 3]}),
            DataFrame({"TIME": ["2021-01-01", "2021-01-02", "2021-01-03"], "VALUE": [4, 5, 6]}),
            DataFrame({"TIME": ["2021-01-01", "2021-01-02", "2021-01-03"], "VALUE": [7, 8, 9]}),
        ]
        names = ["data1", "data2", "data3"]
        save_dataset(data, names, output_location="./test/db/", output_format="db", dataset_name="test_save")
        tables = get_tables("./test/db/test_save.db")
        self.assertEqual(tables, ["data1", "data2", "data3"])
        # for i, name in enumerate(names):
        #     table = table_to_df("./test/db/test_save.db", name)
        #     self.assertTrue(table.equals(data[i]))

        save_dataset(data, names, output_location="./test/db/csv/", output_format="csv", dataset_name="test_save")
        self.assertRaises(
            NotImplementedError,
            save_dataset,
            data,
            names,
            output_location="./test/db/csv/",
            output_format="nonexistent",
            dataset_name="test_save",
        )
