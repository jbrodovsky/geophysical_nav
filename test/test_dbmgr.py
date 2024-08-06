"""
Test DB tools
"""

import os
import shutil
import sqlite3 as sql
import unittest

from pandas import DataFrame

from src.data_management import dbmgr


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
