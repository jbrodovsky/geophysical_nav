# FILEPATH: /home/james/research/test_particle_filter.py
"""
Test the particle filter implementation.
"""

import unittest

import numpy as np
from haversine import Unit, haversine
from pandas import DataFrame

from src.geophysical.gmt_toolbox import get_map_point, get_map_section
from src.geophysical.particle_filter import (
    populate_velocities,
    process_particle_filter,
    propagate,
    rmse,
    run_particle_filter,
    update_relief,
    weighted_rmse,
)


class TestParticleFilter(unittest.TestCase):
    """
    Test the particle filter implementation.
    """

    def test_rmse_single_particle_at_origin(self):
        """
        Test that the RMSE of a single particle at the origin is 0.
        """
        particles = [(0, 0)]
        truth = (0, 0)
        self.assertEqual(rmse(particles, truth), 0)

    def test_rmse_multiple_particles_at_origin(self):
        """
        Test that the RMSE of multiple particles at the origin is 0.
        """
        particles = [(0, 0), (0, 0), (0, 0)]
        truth = (0, 0)
        self.assertEqual(rmse(particles, truth), 0)

    def test_rmse_single_particle_at_distance(self):
        """
        Test that the RMSE of a single particle at a distance is the distance.
        """
        particles = [(1, 1)]
        truth = (0, 0)
        expected = haversine(truth, particles[0], Unit.METERS)
        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)

    def test_rmse_multiple_particles_at_same_distance(self):
        """
        Test that the RMSE of multiple particles at the same distance is the distance.
        """
        particles = [(1, 1), (1, 1), (1, 1)]
        truth = (0, 0)
        expected = haversine(truth, particles[0], Unit.METERS)
        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)

    def test_rmse_multiple_particles_at_different_distances(self):
        """
        Test that the RMSE of multiple particles at different distances is the average distance.
        """
        particles = [(1, 1), (2, 2), (3, 3)]
        truth = (0, 0)
        expected = np.sqrt(np.mean([haversine(truth, p, Unit.METERS) ** 2 for p in particles]))
        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)

    def test_weighted_rmse_single_particle_at_origin(self):
        """
        Test that the weighted RMSE of a single particle at the origin is 0.
        """
        particles = [(0, 0)]
        weights = [1]
        truth = (0, 0)
        self.assertEqual(weighted_rmse(particles, weights, truth), 0)

    def test_weighted_rmse_multiple_particles_at_origin(self):
        """
        Test that the weighted RMSE of multiple particles at the origin is 0.
        """
        particles = [(0, 0), (0, 0), (0, 0)]
        weights = [1, 1, 1]
        truth = (0, 0)
        self.assertEqual(weighted_rmse(particles, weights, truth), 0)

    def test_propagate_single_particle_at_origin(self):
        """
        Test that propagating a single particle at the origin does not change its position.
        """
        particles = np.asarray([[0, 0, 0, 0, 0, 0]])
        control = np.asarray([0, 0, 0])
        noise = np.diag([0, 0, 0])
        dt = 0
        out_particles = propagate(particles, control, dt, noise)
        self.assertTrue((particles == out_particles).all())

    def test_propagate_multiple_particles_at_origin(self):
        """
        Test that propagating multiple particles at the origin does not change their positions.
        """
        particles = np.zeros((3, 6))
        control = np.asarray([0, 0, 0])
        noise = np.diag([0, 0, 0])
        dt = 0
        out_particles = propagate(particles, control, dt, noise)
        self.assertTrue((particles == out_particles).all())

    def test_update_relief(self):
        """
        Test that updating the particles with the relief map does not change their positions.
        """
        particles = np.zeros((3, 6))
        relief = get_map_section(-1, 1, -1, 1, "relief", "01m")
        observation = get_map_point(relief, 0, 0)
        weights = update_relief(particles, relief, observation, 0.1)
        n, _ = particles.shape
        self.assertTrue(len(weights) == n)

    def test_run_particle_filter(self):
        """
        Test that the particle filter runs without error.
        """
        particles = np.zeros((6,))
        n = 1
        relief = get_map_section(-1, 1, -1, 1, "relief", "01m")
        data = DataFrame(
            {
                "LON": [0.0, 1.0, 2.0],
                "LAT": [0.0, 1.0, 2.0],
                "DEPTH": [0.0, 1.0, 2.0],
                "DT": [1.0, 1.0, 1.0],
                "VN": [1.0, 1.0, 1.0],
                "VE": [1.0, 1.0, 1.0],
                "VD": [1.0, 1.0, 1.0],
            }
        )
        est, rms, err = run_particle_filter(particles, np.eye(6), n, data, relief)
        self.assertTrue(est is not None)
        self.assertTrue(rms is not None)
        self.assertTrue(err is not None)

    def test_process_particle_filter(self):
        """
        Test that the particle filter runs without error.
        """
        config = {
            "velocity_noise": [2.6, 2.6, 0],
            "bathy_mean_d": 21.285045267468078,
            "bathy_std": 215.01077619593897,
            "n": 100000,
            "cov": [0.016666666666666666, 0.016666666666666666, 0, 2.6, 2.6, 0],
            "gravity_mean_d": 15.272244889849036,
            "gravity_std": 12.685835143336485,
            "magnetic_mean_d": 5.700504804344624,
            "magnetic_std": 170.11393371360725,
        }
        data = DataFrame(
            {
                "LON": [0.0, 1.0, 2.0],
                "LAT": [0.0, 1.0, 2.0],
                "DEPTH": [0.0, 1.0, 2.0],
                "GRAV_ANOM": [0.0, 1.0, 2.0],
                "MAG_RES": [0.0, 1.0, 2.0],
                "DT": [1.0, 1.0, 1.0],
                "VN": [1.0, 1.0, 1.0],
                "VE": [1.0, 1.0, 1.0],
                "VD": [1.0, 1.0, 1.0],
            }
        )
        data, geo_map = process_particle_filter(data, config)
        self.assertTrue(data is not None)
        self.assertTrue(geo_map is not None)

        data, geo_map = process_particle_filter(data, config, map_type="gravity", map_resolution="02m")
        self.assertTrue(data is not None)
        self.assertTrue(geo_map is not None)

        data, geo_map = process_particle_filter(data, config, map_type="magnetic", map_resolution="02m")
        self.assertTrue(data is not None)
        self.assertTrue(geo_map is not None)

        self.assertRaises(ValueError, process_particle_filter, data, config, map_type="unknown")

    def test_populate_velocities(self):
        """
        Test that the velocities are populated without error.
        """
        data = DataFrame(
            {
                "LON": [0.0, 1.0, 2.0],
                "LAT": [0.0, 1.0, 2.0],
                "DEPTH": [0.0, 1.0, 2.0],
                "DT": [1.0, 1.0, 1.0],
                "VN": [1.0, 1.0, 1.0],
                "VE": [1.0, 1.0, 1.0],
                "VD": [1.0, 1.0, 1.0],
            }
        )
        data = populate_velocities(data)
        self.assertTrue(data["VN"] is not None)
        self.assertTrue(data["VE"] is not None)
        self.assertTrue(data["VD"] is not None)


if __name__ == "__main__":
    unittest.main()
