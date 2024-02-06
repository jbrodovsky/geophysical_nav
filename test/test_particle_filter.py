# FILEPATH: /home/james/research/test_particle_filter.py
"""
Test the particle filter implementation.
"""

# 
# import unittest
# 
# import numpy as np
# from haversine import Unit, haversine
# 
# from src.geophysical.particle_filter import rmse
# 
# 
# class TestParticleFilter(unittest.TestCase):
#     """
#     Test the particle filter implementation.
#     """
# 
#     def test_rmse_single_particle_at_origin(self):
#         """
#         Test that the RMSE of a single particle at the origin is 0.
#         """
#         particles = [(0, 0)]
#         truth = (0, 0)
#         self.assertEqual(rmse(particles, truth), 0)
# 
#     def test_rmse_multiple_particles_at_origin(self):
#         """
#         Test that the RMSE of multiple particles at the origin is 0.
#         """
#         particles = [(0, 0), (0, 0), (0, 0)]
#         truth = (0, 0)
#         self.assertEqual(rmse(particles, truth), 0)
# 
#     def test_rmse_single_particle_at_distance(self):
#         """
#         Test that the RMSE of a single particle at a distance is the distance.
#         """
#         particles = [(1, 1)]
#         truth = (0, 0)
#         expected = haversine(truth, particles[0], Unit.METERS)
#         self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
# 
#     def test_rmse_multiple_particles_at_same_distance(self):
#         """
#         Test that the RMSE of multiple particles at the same distance is the distance.
#         """
#         particles = [(1, 1), (1, 1), (1, 1)]
#         truth = (0, 0)
#         expected = haversine(truth, particles[0], Unit.METERS)
#         self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
# 
#     def test_rmse_multiple_particles_at_different_distances(self):
#         """
#         Test that the RMSE of multiple particles at different distances is the average distance.
#         """
#         particles = [(1, 1), (2, 2), (3, 3)]
#         truth = (0, 0)
#         expected = np.sqrt(np.mean([haversine(truth, p, Unit.METERS) ** 2 for p in particles]))
#         self.assertAlmostEqual(rmse(particles, truth), expected, places=5)
# 
# 
# if __name__ == "__main__":
#     unittest.main()
# 