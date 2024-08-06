"""
Test the M77T toolbox implementation.
"""

import os
import shutil
import unittest

from pandas import DataFrame, read_csv

from src.data_management import m77t


class TestM77TToolbox(unittest.TestCase):
    """
    Test the M77T toolbox implementation.
    """

    def setUp(self) -> None:
        """
        Set up the test environment data.
        """
        # data, names = m77t_toolbox.process_mgd77("./test")
        df: DataFrame = read_csv("./test/test_data.m77t", delimiter="\t", header=0)
        self.df: DataFrame = m77t.m77t_to_df(df)
        self.df.to_csv("./test/test_data.csv", index=False)

    def tearDown(self) -> None:
        """
        Tear down the test environment data.
        """
        if os.path.exists("./test/db/tracklines.db"):
            os.remove("./test/db/tracklines.db")
        if os.path.exists("./test/csv"):
            shutil.rmtree("./test/csv")
        if os.path.exists("./test/db"):
            shutil.rmtree("./test/db")

    def test_m77t_toolbox(self) -> None:
        """
        Test that the M77T toolbox can be imported.
        """
        self.assertTrue(m77t)

    def test_find_periods(self) -> None:
        """
        Test that the periods can be found.
        """
        self.assertTrue(m77t.find_periods)
        self.assertEqual(m77t.find_periods([1, 0, 0, 1, 0]), [(1, 2), (4, 4)])
        self.assertEqual(m77t.find_periods([0, 0, 0, 0, 0]), [(0, 4)])
        self.assertEqual(m77t.find_periods([1, 1, 1, 1, 1]), [])
        self.assertEqual(m77t.find_periods([1, 0, 0, 1, 0, 0, 1]), [(1, 2), (4, 5)])

    def test_split_dataset(self) -> None:
        """
        Test that the dataset can be split.
        """
        self.assertTrue(m77t.split_dataset)
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        periods = [(0, 2), (3, 4)]
        splits = m77t.split_dataset(df, periods)
        self.assertEqual(len(splits[0]), 3)

    def test_m77t_to_df(self) -> None:
        """
        Test that the M77T data can be converted to a DataFrame.
        """
        df_in: DataFrame = read_csv(filepath_or_buffer="./test/test_data.m77t", delimiter="\t", header=0)
        df_out: DataFrame = m77t.m77t_to_df(data=df_in)
        self.assertIsInstance(obj=df_out, cls=DataFrame)
        self.assertIn(member="LAT", container=df_out.columns)
        self.assertIn(member="LON", container=df_out.columns)
        self.assertIn(member="CORR_DEPTH", container=df_out.columns)
        self.assertIn(member="MAG_TOT", container=df_out.columns)
        self.assertIn(member="MAG_RES", container=df_out.columns)
        self.assertIn(member="GRA_OBS", container=df_out.columns)
        self.assertIn(member="FREEAIR", container=df_out.columns)
        self.assertNotEqual(len(df_out), 0)

    def test_read_m77t(self) -> None:
        """
        Test that the M77T data can be read.
        """
        df: DataFrame = m77t.read_m77t(filepath="./test/test_data.m77t")
        self.assertIsInstance(obj=df, cls=DataFrame)
        self.assertNotEqual(first=len(df), second=0)
        self.assertRaises(FileNotFoundError, m77t.read_m77t, "./test/missing.m77t")
