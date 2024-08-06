"""
Test cases for open_loop_sim.py
"""

import os
import sqlite3 as sql
import unittest

from pandas import DataFrame


class TestOpenLoopSim(unittest.TestCase):
    """
    Test case for open_loop_sim.py
    """

    def setUp(self):
        """
        Create the files needed for the test
        """
        results = DataFrame(
            {
                "duration": [0, 1, 2, 3, 4, 5],
                "average_error": [0, 1, 2, 3, 4, 5],
                "name": ["a", "b", "c", "d", "e", "f"],
                "start": [0, 1, 2, 3, 4, 5],
                "stop": [1, 2, 3, 4, 5, 6],
                "max error": [0, 1, 2, 3, 4, 5],
                "min error": [0, 1, 2, 3, 4, 5],
            }
        )
        summary = results.copy()
        conn = sql.connect("./DB_ol.db")
        results.to_sql("init", conn)
        summary.to_csv("summary.csv")
        conn.close()

    def tearDown(self):
        """
        Remove the files created during the test
        """
        os.remove("./summary.csv")
        os.remove("./out.txt")
        os.remove("./DB_ol.db")
        pass
