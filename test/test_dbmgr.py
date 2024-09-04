"""
Test DB tools
"""

import os
import shutil
from typing import LiteralString
import unittest

from pandas import DataFrame

from src.data_management import dbmgr, m77t


class TestDatabaseManager(unittest.TestCase):
    """
    Test class for db_tools.py
    """

    def setUp(self):
        """
        Set up the test environment data.
        """
        self.db_path: LiteralString = os.path.join("test", "db")
        self.db_string: LiteralString = f"{self.db_path}/test.db"
        self.table_name = "test_table"
        os.makedirs(name=self.db_path)
        self.db: dbmgr.DatabaseManager = dbmgr.DatabaseManager(source=self.db_string)
        # Check that the database tables have been created
        tables: list[str] = self.db.get_all_tables()
        self.assertIsNotNone(tables)
        self.assertIsNotNone(self.db.get_table("trajectories"))

    def tearDown(self):
        """
        Tear down the test environment data.
        """

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_dbmgr(self) -> None:
        """test that the module can be imported"""
        self.assertIsNotNone(dbmgr)

    def test_trajectory_insertion(self) -> None:
        """test getting a trajectory"""
        data: list[DataFrame] = m77t.process_m77t_file("test/test_data.m77t")
        self.assertIsNotNone(data)
        id: int = self.db.insert_trajectory(trajectory=data[0], name="TestTrajectory")
        self.assertIsNotNone(id)
        trajectory: DataFrame = self.db.get_trajectory(id)
        self.assertIsNotNone(trajectory)
        self.assertEqual(data[0].shape[0], trajectory.shape[0])
        self.assertIsNotNone(self.db.get_all_trajectories())


def test_write_and_read_results_to_file() -> None:
    """test writing and reading results to a file"""
    filename: str = os.path.join("test", "db", "test_result.hdf5")
    configuration: dict = {"test": "test"}
    summary: DataFrame = DataFrame.from_dict(
        data={"row": [0, "bathy", 10]}, orient="index", columns=["error", "measurements", "time"]
    )
    results: list[DataFrame] = [
        DataFrame.from_dict(data={"row": [0, "bathy", 0]}, orient="index", columns=["error", "measurements", "time"]),
        DataFrame.from_dict(data={"row": [1, "mag", 10]}, orient="index", columns=["error", "measurements", "time"]),
        DataFrame.from_dict(data={"row": [2, "grav", 20]}, orient="index", columns=["error", "measurements", "time"]),
    ]
    dbmgr.write_results_to_file(filename=filename, configuration=configuration, summary=summary, results=results)
    assert os.path.exists(os.path.join(filename))
    configuration, summary, results = dbmgr.read_results_file(filename=filename)
    assert configuration == {"test": "test"}
    assert summary.shape == (1, 3)
    assert results[0].shape == (1, 3)
    assert results[1].shape == (1, 3)
    assert results[2].shape == (1, 3)
    shutil.rmtree(os.path.join("test", "db"))


if __name__ == "__main__":
    unittest.main()
