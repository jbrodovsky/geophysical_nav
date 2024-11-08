#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

namespace transform {
/** Degrees to radians conversion factor. */
constexpr double DEG_TO_RAD = M_PI / 180.0;

/** Radians to degrees conversion factor. */
constexpr double RAD_TO_DEG = 1.0 / DEG_TO_RAD;

/**
 * @brief Convert latitude, longitude, altitude to ECEF coordinates.
 * @param lla A vector containing latitude, longitude, and altitude in degrees and meters.
 * @return ECEF Cartesian coordinates as a 3D vector.
 */
Eigen::Vector3d lla_to_ecef(const Eigen::Vector3d& lla);

/**
 * @brief Convert ECEF Cartesian coordinates to latitude, longitude, and altitude.
 * @param r_e ECEF coordinates as a 3D vector.
 * @return Latitude, longitude, and altitude as a 3D vector.
 */
Eigen::Vector3d ecef_to_lla(const Eigen::Vector3d& r_e);

/**
 * @brief Convert latitude, longitude, altitude to NED coordinates relative to an origin.
 * @param lla Current latitude, longitude, altitude as a 3D vector.
 * @param lla_origin Origin latitude, longitude, altitude as a 3D vector.
 * @return NED coordinates as a 3D vector.
 */
Eigen::Vector3d lla_to_ned(const Eigen::Vector3d& lla, const Eigen::Vector3d& lla_origin);

/**
 * @brief Convert north, east, down coordinates to latitude, longitude, altitude.
 * @param ned NED coordinates as a 3D vector.
 * @return Latitude, longitude, and altitude as a 3D vector.
 */
Eigen::Vector3d ned_to_lla(const Eigen::Vector3d& ned, const Eigen::Vector3d& lla_origin);

/**
 * @brief Create a rotation matrix projecting from ECEF to NED frame.
 * @param lat Latitude in degrees.
 * @param lon Longitude in degrees.
 * @return Rotation matrix from ECEF to NED.
 */
Eigen::Matrix3d mat_en_from_ll(double lat, double lon);

/**
 * @brief Create a rotation matrix from roll, pitch, and heading angles.
 * @param rph Vector containing roll, pitch, and heading in degrees.
 * @return Rotation matrix.
 */
Eigen::Matrix3d mat_from_rph(const Eigen::Vector3d& rph);

Eigen::Matrix3d mat_from_rph(double roll, double pitch, double heading);
/**
 * @brief Convert a rotation matrix to roll, pitch, and heading angles.
 * @param mat Rotation matrix.
 * @return Vector of roll, pitch, and heading in degrees.
 */
Eigen::Vector3d mat_to_rph(const Eigen::Matrix3d& mat);

void say_hello();

}
#endif