"""
Test the M77T toolbox implementation.
"""

import unittest
import shutil
import os

from pandas import DataFrame, read_csv

from src.geophysical import m77t_toolbox


class TestM77TToolbox(unittest.TestCase):
    """
    Test the M77T toolbox implementation.
    """

    def setUp(self):
        """
        Set up the test environment data.
        """
        # data, names = m77t_toolbox.process_mgd77("./test")
        df = read_csv("./test/test_data.m77t", delimiter="\t", header=0)
        self.df = m77t_toolbox.m77t_to_df(df)
        self.df.to_csv("./test/test_data.csv", index=False)

    def tearDown(self):
        """
        Tear down the test environment data.
        """
        if os.path.exists("./test/db/tracklines.db"):
            os.remove("./test/db/tracklines.db")
        if os.path.exists("./test/csv"):
            shutil.rmtree("./test/csv")
        if os.path.exists("./test/db"):
            shutil.rmtree("./test/db")

    def test_m77t_toolbox(self):
        """
        Test that the M77T toolbox can be imported.
        """
        self.assertTrue(m77t_toolbox)

    def test_find_periods(self):
        """
        Test that the periods can be found.
        """
        self.assertTrue(m77t_toolbox.find_periods)
        self.assertEqual(m77t_toolbox.find_periods([1, 0, 0, 1, 0]), [(1, 2), (4, 4)])
        self.assertEqual(m77t_toolbox.find_periods([0, 0, 0, 0, 0]), [(0, 4)])
        self.assertEqual(m77t_toolbox.find_periods([1, 1, 1, 1, 1]), [])
        self.assertEqual(m77t_toolbox.find_periods([1, 0, 0, 1, 0, 0, 1]), [(1, 2), (4, 5)])

    def test_split_dataset(self):
        """
        Test that the dataset can be split.
        """
        self.assertTrue(m77t_toolbox.split_dataset)
        df = DataFrame({"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]})
        periods = [(0, 2), (3, 4)]
        splits = m77t_toolbox.split_dataset(df, periods)
        self.assertEqual(len(splits[0]), 3)

    def test_m77t_to_df(self):
        """
        Test that the M77T data can be converted to a DataFrame.
        """
        df_in = read_csv("./test/test_data.m77t", delimiter="\t", header=0)
        df_out = m77t_toolbox.m77t_to_df(df_in)
        self.assertIsInstance(df_out, DataFrame)
        self.assertIn("LAT", df_out.columns)
        self.assertIn("LON", df_out.columns)
        self.assertIn("CORR_DEPTH", df_out.columns)
        self.assertIn("MAG_TOT", df_out.columns)
        self.assertIn("MAG_RES", df_out.columns)
        self.assertIn("GRA_OBS", df_out.columns)
        self.assertIn("FREEAIR", df_out.columns)
        self.assertNotEqual(len(df_out), 0)

    # def test_process_mgd77(self):
    #     """
    #     Test that the MGD77 data can be processed.
    #     """
    #     data, names = m77t_toolbox.process_mgd77("./test")
    #     self.assertNotEqual(len(data), 0)
    #     self.assertIn("test_data", names)

    # def test_save_mgd77_dataset(self):
    #     """
    #     Test that the MGD77 dataset can be saved.
    #     """
    #     data, names = m77t_toolbox.process_mgd77("./test")
    #     m77t_toolbox.save_mgd77_dataset(data, names, "./test/db", "db", "tracklines")
    #     m77t_toolbox.save_mgd77_dataset(data, names, "./test/", "csv", "tracklines")
    #     self.assertTrue(os.path.exists("./test/db/tracklines.db"))
    #     m77t_toolbox.save_mgd77_dataset(data, names, "./test/csv", "csv", "tracklines")
    #     self.assertTrue(os.path.exists(f"./test/csv/{names[0]}.csv"))
    #     self.assertRaises(
    #         NotImplementedError, m77t_toolbox.save_mgd77_dataset, data, names, "./test", "json", "tracklines"
    #     )

    def test_parse_trackline_from_file(self):
        """
        Test that the trackline can be parsed from a file.
        """
        tracklines, names = m77t_toolbox.parse_trackline_from_file("./test/test_data.m77t", data_types=["depth"])
        self.assertNotEqual(len(names), 0)
        trackline = tracklines[0]
        self.assertIsInstance(trackline, DataFrame)
        self.assertNotEqual(len(trackline), 0)
        self.assertRaises(FileNotFoundError, m77t_toolbox.parse_trackline_from_file, "./test/missing.m77t", ["depth"])
        # self.assertRaises(NotImplementedError, m77t_toolbox.parse_trackline_from_file, "./test/test_data.csv", [10])

    # def test_parse_tracklines_from_db(self):
    #     """
    #     Test that the trackline can be parsed from a database.
    #     """
    #     tracklines, _ = m77t_toolbox.parse_tracklines_from_db(
    #         "./test/db/tracklines.db", data_types=["depth", ["mag", "grav"]]
    #     )
    #     self.assertNotEqual(len(tracklines), 0)
    #     trackline = tracklines[0]
    #     self.assertIsInstance(trackline, DataFrame)
    #     self.assertNotEqual(len(trackline), 0)
    #     self.assertRaises(
    #         NotImplementedError, m77t_toolbox.parse_tracklines_from_db, "./test/db/tracklines.db", data_types=[10]
    #     )

    def test_validate_data_type_string(self):
        """
        Test that the data type string can be validated.
        """
        self.assertIsInstance(m77t_toolbox.validate_data_type_string("all"), str)
        self.assertEqual(m77t_toolbox.validate_data_type_string("all"), "DGM")
        self.assertEqual(m77t_toolbox.validate_data_type_string("relief"), "D")
        self.assertEqual(m77t_toolbox.validate_data_type_string("depth"), "D")
        self.assertEqual(m77t_toolbox.validate_data_type_string("bathy"), "D")
        self.assertEqual(m77t_toolbox.validate_data_type_string("mag"), "M")
        self.assertEqual(m77t_toolbox.validate_data_type_string("magnetic"), "M")
        self.assertEqual(m77t_toolbox.validate_data_type_string("grav"), "G")
        self.assertEqual(m77t_toolbox.validate_data_type_string("gravity"), "G")
        self.assertRaises(NotImplementedError, m77t_toolbox.validate_data_type_string, "other")

        self.assertIsInstance(m77t_toolbox.validate_data_type_string(["depth", "mag"]), list)
        self.assertEqual(m77t_toolbox.validate_data_type_string(["depth", "gravity", "mag"]), ["D", "G", "M"])
        self.assertEqual(m77t_toolbox.validate_data_type_string(["depth", ["gravity", "mag"]]), ["D", "GM"])

    def test_get_parsed_data_summary(self):
        """
        Test that the parsed data summary can be retrieved.
        """

        df1 = DataFrame(
            {
                "DEPTH": [1, 2, 3, 4, 5],
                "GRAV_ANOM": [6, 7, 8, 9, 10],
                "MAG_RES": [11, 12, 13, 14, 15],
                "LAT": [16, 17, 18, 19, 20],
                "LON": [21, 22, 23, 24, 25],
                "TIME": [26, 27, 28, 29, 30],
            }
        )
        df2 = DataFrame(
            {
                "DEPTH": [1, 2, 3, 4, 5],
                "GRAV_ANOM": [6, 7, 8, 9, 10],
                "MAG_RES": [11, 12, 13, 14, 15],
                "LAT": [16, 17, 18, 19, 20],
                "LON": [21, 22, 23, 24, 25],
                "TIME": [26, 27, 28, 29, 30],
            }
        )

        summary = m77t_toolbox.get_parsed_data_summary([df1, df2], ["track1", "track2"])
        self.assertIsInstance(summary, DataFrame)
        self.assertNotEqual(len(summary), 0)
        df1 = df1.drop(columns=["DEPTH", "GRAV_ANOM", "MAG_RES"])
        summary = m77t_toolbox.get_parsed_data_summary([df1], ["track1"])
        self.assertTrue(summary["depth_mean"].isna().all())
        self.assertTrue(summary["depth_std"].isna().all())
        self.assertTrue(summary["depth_range"].isna().all())
        self.assertTrue(summary["grav_mean"].isna().all())
        self.assertTrue(summary["grav_std"].isna().all())
        self.assertTrue(summary["grav_range"].isna().all())
        self.assertTrue(summary["mag_mean"].isna().all())
        self.assertTrue(summary["mag_std"].isna().all())
        self.assertTrue(summary["mag_range"].isna().all())

    def test_read_m77t(self):
        """
        Test that the M77T data can be read.
        """
        df = m77t_toolbox.read_m77t("./test/test_data.m77t")
        self.assertIsInstance(df, DataFrame)
        self.assertNotEqual(len(df), 0)
        self.assertRaises(FileNotFoundError, m77t_toolbox.read_m77t, "./test/missing.m77t")
