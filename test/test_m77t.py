"""
Test the M77T toolbox implementation.
"""

import os
import shutil
import unittest
from typing import List

from numpy.typing import NDArray
from pandas import DataFrame, read_csv

from src.data_management import m77t


class TestM77TToolbox(unittest.TestCase):
    """
    Test the M77T toolbox implementation.
    """

    def test_read_m77t(self) -> None:
        """
        Test that the M77T data can be read.
        """
        df: DataFrame = m77t.read_m77t(filepath="./test/test_data.m77t")
        self.assertIsInstance(obj=df, cls=DataFrame)
        self.assertNotEqual(first=len(df), second=0)
        self.assertRaises(FileNotFoundError, m77t.read_m77t, "./test/missing.m77t")

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
        self.assertIn(member="DEPTH", container=df_out.columns)
        self.assertIn(member="MAG_TOT", container=df_out.columns)
        self.assertIn(member="MAG_RES", container=df_out.columns)
        self.assertIn(member="GRA_OBS", container=df_out.columns)
        self.assertIn(member="FREEAIR", container=df_out.columns)
        self.assertNotEqual(len(df_out), 0)

    def test_calculate_bearing(self) -> None:
        """Test that the bearing can be calculated successfully."""
        self.assertTrue(expr=m77t.calculate_bearing)
        self.assertAlmostEqual(
            first=m77t.calculate_bearing(coords1=(0, 0), coords2=(1, 1)),
            second=45,
            places=2,
        )
        self.assertAlmostEqual(
            first=m77t.calculate_bearing(coords1=(0, 0), coords2=(1, 0)),
            second=0,
            places=2,
        )
        self.assertAlmostEqual(
            first=m77t.calculate_bearing(coords1=(0, 0), coords2=(0, 1)),
            second=90,
            places=2,
        )
        self.assertAlmostEqual(
            first=m77t.calculate_bearing(coords1=(0, 0), coords2=(-1, 0)),
            second=180,
            places=2,
        )
        self.assertAlmostEqual(
            first=m77t.calculate_bearing(coords1=(0, 0), coords2=(0, -1)),
            second=270,
            places=2,
        )

    def test_calculate_bearing_vector(self) -> None:
        """Test that the bearing vector can be calculated successfully."""
        self.assertTrue(expr=m77t.calculate_bearing_vector)
        coords1: list[list[int]] = [[0, 0], [0, 0]]
        coords2: list[list[int]] = [[1, 1], [1, 0]]
        result: NDArray = m77t.calculate_bearing_vector(coords1=coords1, coords2=coords2)
        self.assertAlmostEqual(first=result[0], second=45, places=2)
        self.assertAlmostEqual(first=result[1], second=0, places=2)

    def test_process_m77t_file(self) -> None:
        """Test that the M77T file can be processed successfully."""
        self.assertTrue(expr=m77t.process_m77t_file)
        df: List[DataFrame] = m77t.process_m77t_file(filepath="./test/test_data.m77t")
        self.assertIsInstance(obj=df[0], cls=DataFrame)
        self.assertNotEqual(first=len(df), second=0)


if __name__ == "__main__":
    unittest.main()
