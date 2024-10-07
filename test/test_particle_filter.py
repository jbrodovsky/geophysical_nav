# FILEPATH: /home/james/research/test_particle_filter.py
"""
Test the particle filter implementation.
"""
import json
from os import remove
from os.path import join
from glob import glob
import unittest

import numpy as np


from src.geophysical.gmt_toolbox import GeophysicalMap, MapType, ReliefResolution, GravityResolution, MagneticResolution
from src.geophysical.particle_filter import (
    MeasurementType,
    GeophysicalMeasurement,
    ParticleFilterConfig,
    coning_and_sculling_correction,
    rmse,
    propagate_imu,
    update_anomaly,
    update_relief,
    vector_to_skew_symmetric,
    skew_symmetric_to_vector,
)


def test_rmse() -> None:
    """
    Test that the RMSE of a single particle at the origin is 0.
    """
    particles = np.array([[0, 0, 0]])
    truth = np.array([0, 0, 0])
    assert rmse(particles, truth) == 0
    assert rmse(particles, truth, include_altitude=True) == 0


def test_coning_and_sculling_correction() -> None:
    """
    Test the coning and sculling correction.
    """
    dt = 0.1
    current_gyros = np.array([0.1, 0.1, 0.1])
    current_accel = np.array([0, 0, 9.81])
    previous_gyros = np.array([0, 0, 0])
    previous_accel = np.array([0, 0, 9.81])

    thetas, dv = coning_and_sculling_correction(current_gyros, current_accel, previous_gyros, previous_accel, dt)
    assert thetas.shape == (3,)
    assert dv.shape == (3,)


def test_vector_to_skew_symmetric() -> None:
    """
    Test the vector to skew-symmetric matrix conversion.
    """
    v = np.array([1, 2, 3])
    skew_symmetric = vector_to_skew_symmetric(v)
    assert skew_symmetric.shape == (3, 3)
    assert np.all(skew_symmetric == np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]]))


def test_skew_symmetric_to_vector() -> None:
    """
    Test the skew-symmetric matrix to vector conversion.
    """
    skew_symmetric = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    v = skew_symmetric_to_vector(skew_symmetric)
    assert v.shape == (3,)
    assert np.all(v == np.array([1, 2, 3]))


def test_propagate_imu() -> None:
    """
    Test the IMU propagation.
    """
    dt = 0.1
    state = np.random.random((10, 15))
    gyro = np.array([0.1, 0.1, 0.1])
    accel = np.array([0, 0, 9.81])
    out_state = propagate_imu(state, gyro, accel, dt)
    assert out_state.shape == state.shape


def test_update_relief() -> None:
    """
    Test the relief update.
    """
    particles = np.random.random((10, 15))
    relief = GeophysicalMap(MeasurementType.RELIEF, ReliefResolution.ONE_MINUTE, -1, 1, -1, 1, 0.1)
    observation = relief.get_map_point(0, 0)
    weights = update_relief(particles, relief, observation, 0.1)
    assert len(weights) == 10


def test_update_anomaly() -> None:
    """
    Test the anomaly update.
    """
    particles = np.random.random((10, 15))
    anomaly = GeophysicalMap(MeasurementType.GRAVITY, GravityResolution.ONE_MINUTE, -1, 1, -1, 1, 0.1)
    observation = anomaly.get_map_point(0, 0)
    weights = update_anomaly(particles, anomaly, observation, 0.1)
    assert len(weights) == 10
    anomaly = GeophysicalMap(MeasurementType.MAGNETIC, MagneticResolution.TWO_MINUTES, -1, 1, -1, 1, 0.1)
    observation = anomaly.get_map_point(0, 0)
    weights = update_anomaly(particles, anomaly, observation, 0.1)
    assert len(weights) == 10


class TestParticleFilterConfig(unittest.TestCase):
    """
    Test the ParticleFilterConfig class.
    """

    def setUp(self) -> None:
        self.config_path = join("test", "pfconfig.json")
        pfconfig = {
            "n": 100,
            "cov": [1, 1, 1],
            "noise": [1, 1, 1],
            "measurement_config": [
                {"name": "BATHYMETRY", "std": 1},
                {"name": "RELIEF", "std": 1},
                {"name": "GRAVITY", "std": 1},
                {"name": "MAGNETIC", "std": 1},
            ],
        }

        with open(self.config_path, "w") as f:
            json.dump(pfconfig, f)

    def tearDown(self) -> None:
        files = glob(join("test", "pfconfig*"))
        for file in files:
            remove(file)

    def test_load(self) -> None:
        config = ParticleFilterConfig.load(self.config_path)
        assert config.n == 100
        assert np.all(config.noise == np.array([1, 1, 1]))
        assert len(config.measurement_config) == 4
        assert config.measurement_config[0].name == MeasurementType.BATHYMETRY
        assert config.measurement_config[0].std == 1
        assert config.measurement_config[1].name == MeasurementType.RELIEF
        assert config.measurement_config[1].std == 1
        assert config.measurement_config[2].name == MeasurementType.GRAVITY
        assert config.measurement_config[2].std == 1
        assert config.measurement_config[3].name == MeasurementType.MAGNETIC
        assert config.measurement_config[3].std == 1

    def test_save(self) -> None:
        config = ParticleFilterConfig.load(self.config_path)
        config.save(join("test", "pfconfig_out.json"))
        config2 = ParticleFilterConfig.load(join("test", "pfconfig_out.json"))
        assert config2.n == 100
        assert np.all(config.noise == np.array([1, 1, 1]))
        assert np.all(config.noise == np.array([1, 1, 1]))
        assert len(config.measurement_config) == 4
        assert config.measurement_config[0].name == MeasurementType.BATHYMETRY
        assert config.measurement_config[0].std == 1
        assert config.measurement_config[1].name == MeasurementType.RELIEF
        assert config.measurement_config[1].std == 1
        assert config.measurement_config[2].name == MeasurementType.GRAVITY
        assert config.measurement_config[2].std == 1
        assert config.measurement_config[3].name == MeasurementType.MAGNETIC
        assert config.measurement_config[3].std == 1


#
#
# class TestParticleFilter(unittest.TestCase):
#    """
#    Test the particle filter implementation.
#    """
#
#    def test_rmse_single_particle_at_origin(self):
#        """
#        Test that the RMSE of a single particle at the origin is 0.
#        """
#        particles = [(0, 0)]
#        truth = (0, 0)
#        self.assertEqual(rmse(particles, truth), 0)
#
#    def test_rmse_multiple_particles_at_origin(self):
#        """
#        Test that the RMSE of multiple particles at the origin is 0.
#        """
#        particles = [(0, 0), (0, 0), (0, 0)]
#        truth = (0, 0)
#        self.assertEqual(rmse(particles, truth), 0)
#
#    def test_rmse_single_particle_at_distance(self):
#        """
#        Test that the RMSE of a single particle at a distance is the distance.
#        """
#        particles = [(1, 1)]
#        truth = (0, 0)
#        expected = haversine(truth, particles[0], Unit.METERS)
#        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
#
#    def test_rmse_multiple_particles_at_same_distance(self):
#        """
#        Test that the RMSE of multiple particles at the same distance is the distance.
#        """
#        particles = [(1, 1), (1, 1), (1, 1)]
#        truth = (0, 0)
#        expected = haversine(truth, particles[0], Unit.METERS)
#        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
#
#    def test_rmse_multiple_particles_at_different_distances(self):
#        """
#        Test that the RMSE of multiple particles at different distances is the average distance.
#        """
#        particles = [(1, 1), (2, 2), (3, 3)]
#        truth = (0, 0)
#        expected = np.sqrt(np.mean([haversine(truth, p, Unit.METERS) ** 2 for p in particles]))
#        self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
#
#    def test_weighted_rmse_single_particle_at_origin(self):
#        """
#        Test that the weighted RMSE of a single particle at the origin is 0.
#        """
#        particles = [(0, 0)]
#        weights = [1]
#        truth = (0, 0)
#        self.assertEqual(weighted_rmse(particles, weights, truth), 0)
#
#    def test_weighted_rmse_multiple_particles_at_origin(self):
#        """
#        Test that the weighted RMSE of multiple particles at the origin is 0.
#        """
#        particles = [(0, 0), (0, 0), (0, 0)]
#        weights = [1, 1, 1]
#        truth = (0, 0)
#        self.assertEqual(weighted_rmse(particles, weights, truth), 0)
#
#    def test_propagate_single_particle_at_origin(self):
#        """
#        Test that propagating a single particle at the origin does not change its position.
#        """
#        particles = np.asarray([[0, 0, 0, 0, 0, 0]])
#        control = np.asarray([0, 0, 0])
#        noise = np.diag([0, 0, 0])
#        dt = 0
#        out_particles = propagate(particles, control, dt, noise)
#        self.assertTrue((particles == out_particles).all())
#
#    def test_propagate_multiple_particles_at_origin(self):
#        """
#        Test that propagating multiple particles at the origin does not change their positions.
#        """
#        particles = np.zeros((3, 6))
#        control = np.asarray([0, 0, 0])
#        noise = np.diag([0, 0, 0])
#        dt = 0
#        out_particles = propagate(particles, control, dt, noise)
#        self.assertTrue((particles == out_particles).all())
#
#    def test_update_relief(self):
#        """
#        Test that updating the particles with the relief map does not change their positions.
#        """
#        particles = np.zeros((3, 6))
#        relief = get_map_section(-1, 1, -1, 1, "relief", "01m")
#        observation = get_map_point(relief, 0, 0)
#        weights = update_relief(particles, relief, observation, 0.1)
#        n, _ = particles.shape
#        self.assertTrue(len(weights) == n)
#
#    def test_run_particle_filter(self):
#        """
#        Test that the particle filter runs without error.
#        """
#        particles = np.zeros((6,))
#        n = 1
#        relief = get_map_section(-1, 1, -1, 1, "relief", "01m")
#        data = DataFrame(
#            {
#                "LON": [0.0, 1.0, 2.0],
#                "LAT": [0.0, 1.0, 2.0],
#                "DEPTH": [0.0, 1.0, 2.0],
#                "DT": [1.0, 1.0, 1.0],
#                "VN": [1.0, 1.0, 1.0],
#                "VE": [1.0, 1.0, 1.0],
#                "VD": [1.0, 1.0, 1.0],
#            }
#        )
#        est, rms, err = run_particle_filter(particles, np.eye(6), n, data, relief)
#        self.assertTrue(est is not None)
#        self.assertTrue(rms is not None)
#        self.assertTrue(err is not None)
#
#    def test_process_particle_filter(self):
#        """
#        Test that the particle filter runs without error.
#        """
#        config = {
#            "velocity_noise": [2.6, 2.6, 0],
#            "bathy_mean_d": 21.285045267468078,
#            "bathy_std": 215.01077619593897,
#            "n": 100000,
#            "cov": [0.016666666666666666, 0.016666666666666666, 0, 2.6, 2.6, 0],
#            "gravity_mean_d": 15.272244889849036,
#            "gravity_std": 12.685835143336485,
#            "magnetic_mean_d": 5.700504804344624,
#            "magnetic_std": 170.11393371360725,
#        }
#        data = DataFrame(
#            {
#                "LON": [0.0, 1.0, 2.0],
#                "LAT": [0.0, 1.0, 2.0],
#                "DEPTH": [0.0, 1.0, 2.0],
#                "GRAV_ANOM": [0.0, 1.0, 2.0],
#                "MAG_RES": [0.0, 1.0, 2.0],
#                "DT": [1.0, 1.0, 1.0],
#                "VN": [1.0, 1.0, 1.0],
#                "VE": [1.0, 1.0, 1.0],
#                "VD": [1.0, 1.0, 1.0],
#            }
#        )
#        data, geo_map = process_particle_filter(data, config)
#        self.assertTrue(data is not None)
#        self.assertTrue(geo_map is not None)
#
#        data, geo_map = process_particle_filter(data, config, map_type="gravity", map_resolution="02m")
#        self.assertTrue(data is not None)
#        self.assertTrue(geo_map is not None)
#
#        data, geo_map = process_particle_filter(data, config, map_type="magnetic", map_resolution="02m")
#        self.assertTrue(data is not None)
#        self.assertTrue(geo_map is not None)
#
#        self.assertRaises(ValueError, process_particle_filter, data, config, map_type="unknown")
#
#    def test_populate_velocities(self):
#        """
#        Test that the velocities are populated without error.
#        """
#        data = DataFrame(
#            {
#                "LON": [0.0, 1.0, 2.0],
#                "LAT": [0.0, 1.0, 2.0],
#                "DEPTH": [0.0, 1.0, 2.0],
#                "DT": [1.0, 1.0, 1.0],
#                "VN": [1.0, 1.0, 1.0],
#                "VE": [1.0, 1.0, 1.0],
#                "VD": [1.0, 1.0, 1.0],
#            }
#        )
#        data = populate_velocities(data)
#        self.assertTrue(data["VN"] is not None)
#        self.assertTrue(data["VE"] is not None)
#        self.assertTrue(data["VD"] is not None)


if __name__ == "__main__":
    unittest.main()
