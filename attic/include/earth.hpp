#pragma once
#include <cmath>
#include <tuple>
#include "Eigen/Core"

// Rotation matricies
// We are concerded about the following four frames:
// - ECI: earth-centered-inertial (inertial frame, non-rotating, origin at the center of the earth, xyz axes in meters)
// - ECEF: earth-centered-earth-fixed (rotating with the earth, origin at the center of the earth, xyz axes in meters)
// - NED: north-east-down (Local Tangent-Plane, origin at the reference point, coordinates in degrees latitude, degrees longitude, meters down)
// - Body: body frame (origin at the center of mass of the vehicle, xyz axes in meters)
// In general we should chain transformations, ex: ECI -> ECEF -> NED -> Body and not do a direct transformation from say ECI to Body.
// This is because the NED frame is a local frame and the ECI frame is a global frame.
// Rotations
namespace earth{
    // Earth constants
    constexpr double RATE = 7.2921159e-5;   // Earth rotation rate (rad/s); omega_ie
    const Eigen::Vector3d RATE_VECTOR = {0, 0, RATE};    // Earth rotation rate vector (rad/s); omega_ie
    constexpr double EQUATORIAL_RADIUS = 6378137.0;    // Earth radius (m); semi-major axis, equitorial radius R_0
    constexpr double POLAR_RADIUS = 6356752.31425;     // Earth radius (m); semi-minor axis, polar radius R_p
    constexpr double ECCENTRICITY = 0.0818191908425;   // Earth eccentricity
    constexpr double ECCENTRICITY_SQUARED = ECCENTRICITY * ECCENTRICITY;   // Square of Earth eccentricity
    constexpr double GE = 9.7803253359;     // Gravity at the equator.
    constexpr double GP = 9.8321849378;     // Gravity at the poles.
    constexpr double f = 1.0 / 298.257223563; // Flattening factor
    constexpr double k = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE); // Somingliana's constant
    // Angle conversions
    constexpr double DEG2RAD = M_PI / 180.0;
    constexpr double RAD2DEG = 180.0 / M_PI;
    // Geodetic conversions
    constexpr double DH2RS = DEG2RAD / 3600.0;  // Degrees (lat/lon) per second to radians per seconds
    constexpr double RS2DH = 1.0 / DH2RS;       // Radians per second to degrees per second
    constexpr double DRH2RRS = DEG2RAD / 60;    // Degrees per root-hour to radians per root-second  
    Eigen::Matrix3d rotateECIToECEF(const double& t);    // We will generally not use this function, but it is included for completeness
    Eigen::Matrix3d rotateECEFToNED(const double& lat, const double& lon);
    Eigen::Matrix3d rotateNEDToBody(const double& roll, const double& pitch, const double& yaw);
    Eigen::Matrix3d rotateNEDToBody(const Eigen::Vector3d& rpy);    // Roll, pitch, yaw
    // Inverse rotations
    Eigen::Matrix3d rotateECEFToECI(const double& t);     // Again, will generally not use this function, but it is included for completeness
    Eigen::Matrix3d rotateNEDToECEF(const double& lat, const double& lon);
    Eigen::Matrix3d rotateBodyToNED(const double& roll, const double& pitch, const double& yaw);
    Eigen::Matrix3d rotateBodyToNED(const Eigen::Vector3d& rpy);    // Roll, pitch, yaw
    // Other useful transformations
    Eigen::Vector3d rotationMatrixToRPY(const Eigen::Matrix3d& R);
    Eigen::Matrix3d rpyToRotationMatrix(const Eigen::Vector3d& rpy);
    Eigen::Matrix3d rpyToRotationMatrix(const double& roll, const double& pitch, const double& yaw);
    Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d& v);
    Eigen::Vector3d skewSymmetricToVector(const Eigen::Matrix3d& m);
    // Cooridnate conversions
    // ECI to ECEF
    Eigen::Vector3d eciToECEF(const double& x, const double& y, const double& z, const double& t);
    Eigen::Vector3d eciToECEF(const Eigen::Vector3d& eci, const double& t);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> eciToECEF(const Eigen::Vector3d& eci, const Eigen::Vector3d& velocity, const double& t);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> eciToECEF(const Eigen::Vector3d& eci, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration, const double& t);
    // ECEF TO ECI
    Eigen::Vector3d ecefToECI(const double& x, const double& y, const double& z, const double& t);
    Eigen::Vector3d ecefToECI(const Eigen::Vector3d& ecef, const double& t);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> ecefToECI(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const double& t);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> ecefToECI(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration, const double& t);
    // NED to ECEF
    Eigen::Vector3d nedToECEF(const double& lat, const double& lon, const double& alt);
    Eigen::Vector3d nedToECEF(const Eigen::Vector3d& ned);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> nedToECEF(const Eigen::Vector3d& ned, const Eigen::Vector3d& velocity);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> nedToECEF(const Eigen::Vector3d& ned, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration);
    // ECEF to NED
    Eigen::Vector3d ecefToNED(const double& x, const double& y, const double& z);
    Eigen::Vector3d ecefToNED(const Eigen::Vector3d& ecef);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> ecefToNED(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity);
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> ecefToNED(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration);
    // Earth properties
    std::tuple<double, double, double> principalRadii(const double& lat, const double& alt);
    double gravity(const double& lat, const double& alt);
    Eigen::Vector3d gravitation(const double& lat, const double& lon, const double& alt);
    Eigen::Vector3d gravitation(const Eigen::Vector3d& lla);
    Eigen::Vector3d rateNED(const double& lat);
    Eigen::Vector3d calculateTransportRate(const double& lat, const double& alt, const double& vel_N, const double& vel_E, const double& vel_D);
    Eigen::Vector3d calculateTransportRate(const Eigen::Vector3d& lla, const Eigen::Vector3d& vel);
} // namespace earth