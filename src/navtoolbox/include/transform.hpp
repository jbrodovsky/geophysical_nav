#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP
#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace transform {
/** Degrees to radians conversion factor. */
constexpr double DEG_TO_RAD = M_PI / 180.0;

/** Radians to degrees conversion factor. */
constexpr double RAD_TO_DEG = 1.0 / DEG_TO_RAD;

/**
 * @brief Convert ECEF coordinates into latitude, longitude, and altitude.
 * 
 * This function is based on the algorithm by D. K. Olson in [1] for converting Earth-Centered, Earth-Fixed (ECEF)
 * coordinates into geodetic latitude, longitude, and altitude.
 * 
 * References:
 * [1] D. K. Olson, "Converting Earth-Centered, Earth-Fixed Coordinates to Geodetic Coordinates", 
 * IEEE Transactions on Aerospace and Electronic Systems, 32 (1996) 473-476.
 * 
 * @param r_e A 3D vector containing the ECEF Cartesian coordinates [x, y, z].
 * @return A 3D vector [latitude, longitude, altitude] with latitude and longitude in degrees and altitude in meters.
 */
Eigen::Vector3d ecef_to_lla(const Eigen::Vector3d& r_e);

/**
 * @brief Overloaded version of `ecef_to_lla` to handle a stack of ECEF coordinates.
 * 
 * @param r_e_stack A vector of 3D vectors, where each inner vector represents [x, y, z] coordinates in ECEF.
 * @return A vector of 3D vectors, where each inner vector represents [latitude, longitude, altitude].
 */
std::vector<Eigen::Vector3d> ecef_to_lla(const std::vector<Eigen::Vector3d>& r_e_stack);

/**
 * @brief Convert ECEF Cartesian coordinates to latitude, longitude, and altitude.
 * @param r_e ECEF coordinates as a 3D vector.
 * @return Latitude, longitude, and altitude as a 3D vector.
 */
Eigen::Vector3d ecef_to_lla(const Eigen::Vector3d& r_e);

/**
 * @brief Convert latitude, longitude, altitude to ECEF coordinates.
 * @param lla Latitude, longitude, altitude as a 3D vector.
 * @return ECEF coordinates as a 3D vector.
 */
Eigen::Vector3d lla_to_ecef(const Eigen::Vector3d& lla);

/**
 * @brief Overloaded version of `lla_to_ecef` to handle a stack of LLA coordinates.
 * @param lla_stack A vector of 3D vectors, where each inner vector represents [latitude, longitude, altitude].
 * @return A vector of 3D vectors, where each inner vector represents [x, y, z] coordinates in ECEF.
 */
std::vector<Eigen::Vector3d> lla_to_ecef(const std::vector<Eigen::Vector3d>& lla_stack);

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

/**
 * @brief Convert a rotation vector to a rotation matrix.
 * 
 * This function creates a rotation matrix from a rotation vector (axis-angle representation).
 * The rotation vector represents the axis of rotation, and its magnitude represents the rotation angle.
 *
 * @param rv The rotation vector (Eigen::Vector3d) where direction is the axis and magnitude is the angle.
 * @return Eigen::Matrix3d The corresponding 3x3 rotation matrix.
 */
Eigen::Matrix3d mat_from_rotvec(const Eigen::Vector3d& rotvec);


/**
 * @brief Convert a rotation matrix to a rotation vector (axis-angle representation).
 * 
 * This function computes the rotation vector (axis-angle representation) from a rotation matrix.
 * The rotation vector's direction represents the rotation axis, and its magnitude represents the rotation angle.
 *
 * @param mat A 3x3 rotation matrix.
 * @return Eigen::Vector3d The corresponding rotation vector, where the vector's direction is the axis and 
 *                         its magnitude is the rotation angle in radians.
 */
Eigen::Vector3d rotvec_from_mat(const Eigen::Matrix3d& mat);

void say_hello();

}
#endif