/**
 * @file earth.hpp
 * @author James Brodovsky (james.brodovsky@gmail.edu)
 * @brief WGS84 Earth model class definition and coordinate transformations
 * @version 0.1.0
 * @date 2025-03-03
 * 
 * @copyright Copyright (c) 2025
 * 
 */

// References 
// 1. Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, Second Edition
//    Chapter 2

#pragma once
#include <cmath>
#include <tuple>

#include "Eigen/Core"

// Earth constants
constexpr double RATE = 7.2921159e-5;   // Earth rotation rate (rad/s); omega_ie
constexpr double EQUATORIAL_RADIUS = 6378137.0;    // Earth radius (m); semi-major axis, equitorial radius R_0
constexpr double POLAR_RADIUS = 6356752.31425;     // Earth radius (m); semi-minor axis, polar radius R_p
constexpr double ECCENTRICITY = 0.0818191908425;   // Earth eccentricity
constexpr double ECCENTRICITY_SQUARED = ECCENTRICITY * ECCENTRICITY;   // Square of Earth eccentricity
constexpr double GE = 9.7803253359;     // Gravity at the equator.
constexpr double GP = 9.8321849378;     // Gravity at the poles.
constexpr double f = 1 / 298.257223563; // Flattening factor
const double F = sqrt(1 - ECCENTRICITY_SQUARED) * GP / GE - 1; // Flattening factor?

// Angle conversions
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double RAD2DEG = 180.0 / M_PI;

// Geodetic conversions
constexpr double DH2RS = DEG2RAD / 3600.0;  // Degrees (lat/lon) per second to radians per seconds
constexpr double RS2DH = 1.0 / DH2RS;       // Radians per second to degrees per second
constexpr double DRH2RRS = DEG2RAD / 60;    // Degrees per root-hour to radians per root-second

// Compute rotation matricies
// We are concerded about the following four frames:
// - ECI: earth-centered-inertial (inertial frame, non-rotating, origin at the center of the earth, xyz axes in meters)
// - ECEF: earth-centered-earth-fixed (rotating with the earth, origin at the center of the earth, xyz axes in meters)
// - NED: north-east-down (Local Tangent-Plane, origin at the reference point, coordinates in degrees latitude, degrees longitude, meters down)
// - Body: body frame (origin at the center of mass of the vehicle, xyz axes in meters)
// In general we should chain transformations, ex: ECI -> ECEF -> NED -> Body and not do a direct transformation from say ECI to Body.
// This is because the NED frame is a local frame and the ECI frame is a global frame.
Eigen::Matrix3d ECIToECEF(const double& t);    // We will generally not use this function, but it is included for completeness
Eigen::Matrix3d ECEFToNED(const double& lat, const double& lon);
Eigen::Matrix3d NEDToBody(const double& roll, const double& pitch, const double& yaw);
// Inverse rotations
Eigen::Matrix3d ECEFToECI(const double& t);     // Again, will generally not use this function, but it is included for completeness
Eigen::Matrix3d NEDToECEF(const double& lat, const double& lon);
Eigen::Matrix3d BodyToNED(const double& roll, const double& pitch, const double& yaw);
// Other useful transformations
Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d& v);
Eigen::Vector3d skewSymmetricToVector(const Eigen::Matrix3d& m);
// Cooridnate conversions
Eigen::Vector3d llaToECEF(const double& lat, const double& lon, const double& alt);
Eigen::Vector3d ecefToLLA(const Eigen::Vector3d& ecef);
// Earth properties
std::tuple<double, double, double> principalRadii(const double& lat, const double& alt);
double gravity(const double& lat, const double& alt);
Eigen::Vector3d gravitation(const double& lat, const double& lon, const double& alt);
Eigen::Vector3d rateNED(const double& lat);

// rotation matricies
// def mat_en_from_ll