"""
Test the GMT Toolbox module.
"""

import os
import unittest

from numpy import ndarray
from xarray import DataArray

from src.geophysical import gmt_toolbox


class TestGMTToolbox(unittest.TestCase):
    """
    Testing class
    """

    def setUp(self):
        """
        Set up the test environment data.
        """
        min_x, min_y, max_x, max_y = 0, 0, 2, 2
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y)
        gmt_toolbox.save_map_file(section, "./test_map/test.nc")
        gmt_toolbox.save_map_file(section, "./test.nc")

    def tearDown(self):
        """
        Tear down the test environment data.
        """
        if os.path.exists("./test_map/test.nc"):
            os.remove("./test_map/test.nc")
        if os.path.exists("./test.nc"):
            os.remove("./test.nc")

    def test_inflate_bounds(self):
        """
        Test the inflate_bounds function.
        """
        min_x, min_y, max_x, max_y = 0, 0, 10, 10
        inflation_percent = 0.1
        new_min_x, new_min_y, new_max_x, new_max_y = gmt_toolbox.inflate_bounds(
            min_x, min_y, max_x, max_y, inflation_percent
        )
        self.assertEqual(new_min_x, -1)
        self.assertEqual(new_min_y, -1)
        self.assertEqual(new_max_x, 11)
        self.assertEqual(new_max_y, 11)

    def test_validate_relief_resolution(self):
        """
        Test the validate_relief_resolution function.
        """
        valid = [
            "1d",
            "30m",
            "20m",
            "15m",
            "10m",
            "06m",
            "05m",
            "04m",
            "03m",
            "02m",
            "01m",
            "30s",
            "15s",
            "03s",
            "01s",
        ]
        for res in valid:
            self.assertTrue(gmt_toolbox._validate_relief_resoltion(res))
        self.assertRaises(ValueError, gmt_toolbox._validate_relief_resoltion, "1s")
        self.assertRaises(ValueError, gmt_toolbox._validate_relief_resoltion, "02s")

    def test_validate_magentic_resolution(self):
        """
        Test the validate_magnetic_resolution function.
        """
        valid = ["01d", "30m", "20m", "15m", "10m", "06m", "05m", "04m", "03m", "02m"]
        for res in valid:
            self.assertTrue(gmt_toolbox._validate_magentic_resolution(res))
        self.assertRaises(ValueError, gmt_toolbox._validate_magentic_resolution, "01s")
        self.assertRaises(ValueError, gmt_toolbox._validate_magentic_resolution, "02s")

    def test_validate_gravity_resolution(self):
        """
        Test the validate_gravity_resolution function.
        """
        valid = [
            "01d",
            "30m",
            "20m",
            "15m",
            "10m",
            "06m",
            "05m",
            "04m",
            "03m",
            "02m",
            "01m",
        ]
        for res in valid:
            self.assertTrue(gmt_toolbox._validate_gravity_resolution(res))
        self.assertRaises(ValueError, gmt_toolbox._validate_gravity_resolution, "01s")
        self.assertRaises(ValueError, gmt_toolbox._validate_gravity_resolution, "02s")

    def test_get_map_section(self):
        """
        Test the get_map_section function.
        """
        min_x, min_y, max_x, max_y = 0, 0, 2, 2
        # Use default relief values
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y)
        self.assertIsInstance(section, DataArray)
        self.assertIsNotNone(section.shape)
        self.assertIsNotNone(section.coords)
        # Test gravity
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y, "gravity")
        self.assertIsInstance(section, DataArray)
        self.assertIsNotNone(section.shape)
        self.assertIsNotNone(section.coords)
        # Test magnetics
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y, "magnetic")
        self.assertIsInstance(section, DataArray)
        self.assertIsNotNone(section.shape)
        self.assertIsNotNone(section.coords)
        # Test invalid
        self.assertRaises(ValueError, gmt_toolbox.get_map_section, min_x, max_x, min_y, max_y, "other")

    def test_save_map_file(self):
        """
        Test the save_map_file function.
        """
        min_x, min_y, max_x, max_y = 0, 0, 2, 2
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y)
        gmt_toolbox.save_map_file(section, "./test.nc")
        self.assertTrue(os.path.exists("./test.nc"))
        gmt_toolbox.save_map_file(section, "./test_map/test.nc")
        self.assertTrue(os.path.exists("./test_map/test.nc"))

    def test_load_map_file(self):
        """
        Try to load ./test.nc and check if it is a DataArray.
        """
        section = gmt_toolbox.load_map_file("./test_map/test.nc")
        self.assertIsInstance(section, DataArray)
        self.assertIsNotNone(section.shape)
        self.assertIsNotNone(section.coords)
        self.assertRaises(FileNotFoundError, gmt_toolbox.load_map_file, "./tester.nc")

    def test_get_map_point(self):
        """
        Test the get_map_point function.
        """
        min_x, min_y, max_x, max_y = 0, 0, 2, 2
        section = gmt_toolbox.get_map_section(min_x, max_x, min_y, max_y)
        point = gmt_toolbox.get_map_point(section, 1, 1)
        self.assertIsInstance(point, ndarray)
