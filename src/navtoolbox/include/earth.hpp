/**
 * @file EarthModel.h
 * @brief Earth geometry and gravity models using WGS84 parameters.
 *
 * This header defines constants and computation models for an ellipsoidal Earth.
 * Models are based on [Groves, "Principles of GNSS, Inertial, and Multisensor Integrated
 * Navigation Systems", 2nd edition].
 */
#ifndef EARTH_HPP
#define EARTH_HPP
#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>
#include <iostream>

namespace earth {

/** @brief Earth's rotation rate in rad/s */
constexpr double RATE = 7.292115e-5;
/** @brief Semi-major axis of Earth ellipsoid */
constexpr double A = 6378137.0;
/** @brief Squared eccentricity of Earth ellipsoid */
constexpr double E2 = 6.6943799901413e-3;
/** @brief Gravity at the equator */
constexpr double GE = 9.7803253359;
/** @brief Gravity at the pole */
constexpr double GP = 9.8321849378;
const double F = (std::sqrt(1 - E2) * GP / GE) - 1;

/**
 * @brief Computes the principal radii of curvature of Earth ellipsoid.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return Array containing principal radii (rn, re, rp).
 */
std::tuple<double, double, double> principal_radii(double lat, double alt);

/**
 * @brief Computes the gravity magnitude using the Somigliana model with linear altitude correction.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return Gravity magnitude in m/s^2.
 */
double gravity(double lat, double alt);

/**
 * @brief Computes the gravity vector in the NED (North-East-Down) frame.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return 3D vector (Eigen::Vector3d) of gravity in the NED frame.
 */
Eigen::Vector3d gravity_n(double lat, double alt);

/**
 * @brief Computes the gravitational force vector in ECEF frame.
 * Accounts for Earth mass attraction and eliminates centrifugal force.
 * @param lla Array containing latitude (degrees), longitude (degrees), and altitude (meters).
 * @return 3D gravitational force vector (Eigen::Vector3d) in the ECEF frame.
 */

Eigen::Vector3d gravitation_ecef(const std::array<double, 3>& lla);


/**
 * @brief Computes the Earth curvature matrix.
 * Links linear displacement and angular rotation of NED frame.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return 3x3 curvature matrix (Eigen::Matrix3d).
 */
Eigen::Matrix3d curvature_matrix(double lat, double alt);

/**
 * @brief Computes Earth rate in the NED frame.
 * @param lat Latitude in degrees.
 * @return Earth rate vector (Eigen::Vector3d) in NED components.
 */
Eigen::Vector3d rate_n(double lat);

void say_hello();

} // namespace earth

#endif // EARTH_HPP