/**
 * @file earth.cppm
 * @author James Brodovsky (james.brodovsky@gmail.edu)
 * @brief WGS84 Earth model class definition and coordinate transformations
 * @version 0.1.0
 * @date 2025-03-03
 * @details 
 * This file contains the implementation details for the WGS84 Earth model and coordinate transformations.
 * The notation and equations are based on the book "Principles of GNSS, Inertial, and Multisensor Integrated 
 * Navigation Systems, Second Edition" by Paul D. Groves. This file corresponds to Chapter 2 of the book. 
 * Effort has been made to reproduce most of the equations and notation from the book. Generally, variable
 * and constants have been named for the quatity they represent. For example, the Earth rotation rate in 
 * vector form is denoted by \omega_ie in the book and called RATE_VECTOR in this file. Note the following
 * naming and coordinate conventions:
 * - ECI: Earth-Centered-Inertial (inertial frame, non-rotating, origin at the center of the earth, xyz axes in meters)
 * - ECEF: Earth-Centered-Earth-Fixed (rotating with the earth, origin at the center of the earth, xyz axes in meters)
 * - NED: North-East-Down (Local Tangent-Plane, origin at the reference point, coordinates in degrees latitude, degrees longitude, meters down)
 * - Body: Body frame (origin at the center of mass of the vehicle, xyz axes in meters)
 * Specifically take note of the down/altitude conventions with respect to variables names: NED (or ned) 
 * has altitude as positive down. If LLA (or lla) is used, the altitude is positive up and should be treated 
 * instead as a ENU (east-north-up) frame.
 * @copyright Copyright (c) 2025
 * 
 */

#include "Eigen/Geometry"
#include "earth.hpp"

/**
 * @brief Computes the rotation matrix from ECI to ECEF.
 * 
 * @param t double, time in seconds since epoch
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateECIToECEF(const double& t) {
    Eigen::Matrix3d R;
    R << cos(earth::RATE) * t, sin(earth::RATE) * t, 0,
        -sin(earth::RATE) * t, cos(earth::RATE) * t, 0,
        0, 0, 1;
    return R;
}
/**
 * @brief Computes rotation matrix from ECEF to NED.
 * 
 * @param lat double, latitude in degrees
 * @param lon double, longitude in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateECEFToNED(const double& lat, const double& lon) {
    Eigen::Matrix3d R;
    R << -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
        -sin(lon), cos(lon), 0,
        -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat);
    return R;
}
/**
 * @brief Computes the rotation matrix from NED to body.
 * 
 * @param roll double, roll angle in degrees
 * @param pitch double, pitch angle in degrees
 * @param yaw double, yaw angle in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateNEDToBody(const double& roll, const double& pitch, const double& yaw) {
    Eigen::Matrix3d R;
    R = (Eigen::AngleAxisd(yaw * earth::DEG2RAD, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch * earth::DEG2RAD, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll * earth::DEG2RAD, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return R;
}
/**
 * @brief Computes the rotation matrix from NED to body.
 * 
 * @param rpy Eigen::Vector3d, 3x1 roll, pitch, yaw in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateNEDToBody(const Eigen::Vector3d& rpy) {
    return earth::rotateNEDToBody(rpy(0), rpy(1), rpy(2));
}
// Inverse rotations
/**
 * @brief Computes the rotation matrix from ECEF to ECI.
 * 
 * @param t double, time in seconds since epoch
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateECEFToECI(const double& t) {
    return rotateECIToECEF(t).transpose();
}
/**
 * @brief Computes the rotation matrix from NED to ECEF.
 * 
 * @param lat double, latitude in degrees
 * @param lon double, longitude in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateNEDToECEF(const double& lat, const double& lon) {
    return earth::rotateECEFToNED(lat, lon).transpose();
}
/**
 * @brief Computes the rotation matrix from body to NED.
 * 
 * @param roll double, roll angle in degrees
 * @param pitch double, pitch angle in degrees
 * @param yaw double, yaw angle in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateBodyToNED(const double& roll, const double& pitch, const double& yaw) {
    // Rotation matrix from body to NED
    // roll: roll angle (rad)
    // pitch: pitch angle (rad)
    // yaw: yaw angle (rad)
    // Returns: 3x3 rotation matrix from body to NED
    return earth::rotateNEDToBody(roll, pitch, yaw).transpose();
}
/**
 * @brief Computes the rotation matrix from body to NED.
 * 
 * @param rpy Eigen::Vector3d, 3x1 roll, pitch, yaw in degrees
 * @return Eigen::Matrix3d 
 */
Eigen::Matrix3d earth::rotateBodyToNED(const Eigen::Vector3d& rpy) {
    return earth::rotateBodyToNED(rpy(0), rpy(1), rpy(2));
}
// Other useful transformations

/**
 * @brief Converts a rotation matrix to roll, pitch, yaw.
 * 
 * @param R Eigen::Matrix3d, 3x3 rotation matrix
 * @return Eigen::Vector3d, 3x1 roll, pitch, yaw in degrees
 */
Eigen::Vector3d earth::rotationMatrixToRPY(const Eigen::Matrix3d& R) {
    Eigen::Vector3d rpy;
    rpy(0) = atan2(R(2, 1), R(2, 2)) * earth::RAD2DEG;
    rpy(1) = asin(-R(2, 0)) * earth::RAD2DEG;
    rpy(2) = atan2(R(1, 0), R(0, 0)) * earth::RAD2DEG;
    return rpy;
}
/**
 * @brief Converts roll, pitch, yaw to a rotation matrix.
 * 
 * @param rpy Eigen::Vector3d, 3x1 roll, pitch, yaw in degrees
 * @return Eigen::Matrix3d, 3x3 rotation matrix
 */
Eigen::Matrix3d earth::rpyToRotationMatrix(const Eigen::Vector3d& rpy) {
    return earth::rotateNEDToBody(rpy);
}
/**
 * @brief Converts roll, pitch, yaw to a rotation matrix.
 * 
 * @param roll double, roll angle in degrees
 * @param pitch double, pitch angle in degrees
 * @param yaw double, yaw angle in degrees
 * @return Eigen::Matrix3d, 3x3 rotation matrix
 */
Eigen::Matrix3d earth::rpyToRotationMatrix(const double& roll, const double& pitch, const double& yaw) {
    return earth::rotateNEDToBody(roll, pitch, yaw);
}
/**
 * @brief Converts a vector to a skew-symmetric matrix.
 * 
 * @param v Eigen::Vector3d, 3x1 vector
 * @return Eigen::Matrix3d, 3x3 skew-symmetric matrix
 */
Eigen::Matrix3d earth::vectorToSkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
        v(2), 0, -v(0),
        -v(1), v(0), 0;
    return m;
}
/**
 * @brief Converts a skew-symmetric matrix to a vector.
 * 
 * @param m Eigen::Matrix3d, 3x3 skew-symmetric matrix
 * @return Eigen::Vector3d, 3x1 vector
 */
Eigen::Vector3d earth::skewSymmetricToVector(const Eigen::Matrix3d& m) {
    Eigen::Vector3d v;
    v << m(2, 1), m(0, 2), m(1, 0);
    return v;
}
// Cooridnate conversions
/**
 * @brief Converts ECI coordinates to ECEF coordinates.
 * 
 * @param x double, ECI x-coordinate in meters
 * @param y double, ECI y-coordinate in meters
 * @param z double, ECI z-coordinate in meters
 * @param t double, time in seconds since epoch
 * @return Eigen::Vector3d, 3x1 ECEF coordinates
 */
Eigen::Vector3d earth::eciToECEF(const double& x, const double& y, const double& z, const double& t) {
    Eigen::Vector3d eci = {x, y, z};
    return earth::eciToECEF(eci, t);
}
/**
 * @brief Converts ECI coordinates to ECEF coordinates.
 * 
 * @param eci Eigen::Vector3d, 3x1 ECI coordinates in meters
 * @param t double, time in seconds since epoch
 * @return Eigen::Vector3d, 3x1 ECEF coordinates
 */
Eigen::Vector3d earth::eciToECEF(const Eigen::Vector3d& eci, const double& t) {
    Eigen::Matrix3d C_ie = earth::rotateECIToECEF(t);
    return C_ie * eci;        
}
/**
 * @brief Converts ECI coordinates to ECEF coordinates.
 * 
 * @param eci Eigen::Vector3d, 3x1 ECI coordinates in meters
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @param t double, time in seconds since epoch
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d>, 3x1 ECEF coordinates, 3x1 ECEF velocity
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d> earth::eciToECEF(const Eigen::Vector3d& eci, const Eigen::Vector3d& velocity, const double& t) {
    Eigen::Matrix3d C_ie = earth::rotateECIToECEF(t);
    Eigen::Vector3d ecef = C_ie * eci;
    Eigen::Matrix3d Omega_ie = earth::vectorToSkewSymmetric(earth::RATE_VECTOR);
    Eigen::Vector3d ecef_velocity = C_ie * (velocity - Omega_ie * eci);
    return std::make_tuple(ecef, ecef_velocity);
}
/**
 * @brief Converts ECI coordinates to ECEF coordinates.
 * 
 * @param eci Eigen::Vector3d, 3x1 ECI coordinates in meters
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @param acceleration Eigen::Vector3d, 3x1 acceleration in m/s^2
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>, 3x1 ECEF coordinates, 3x1 ECEF velocity, 3x1 ECEF acceleration
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> earth::eciToECEF(const Eigen::Vector3d& eci, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration, const double& t) {
    Eigen::Matrix3d C_ie = earth::rotateECIToECEF(t);
    Eigen::Vector3d ecef = C_ie * eci;
    Eigen::Matrix3d Omega_ie = earth::vectorToSkewSymmetric(earth::RATE_VECTOR);
    Eigen::Vector3d ecef_velocity = C_ie * (velocity - Omega_ie * eci);
    Eigen::Vector3d ecef_acceleration = C_ie * (acceleration - 2 * Omega_ie * velocity - Omega_ie * Omega_ie * eci);
    return std::make_tuple(ecef, ecef_velocity, ecef_acceleration);
}
/**
 * @brief Converts ECEF coordinates to ECI coordinates.
 * 
 * @param t double, time in seconds since epoch
 * @return Eigen::Vector3d, 3x1 ECI coordinates
 */
Eigen::Vector3d earth::ecefToECI(const double& x, const double& y, const double& z, const double& t) {
    Eigen::Vector3d ecef = {x, y, z};
    return earth::ecefToECI(ecef, t);
}
/**
 * @brief Converts ECEF coordinates to ECI coordinates.
 * 
 * @param ecef Eigen::Vector3d, 3x1 ECEF coordinates in meters
 * @param t double, time in seconds since epoch
 * @return Eigen::Vector3d, 3x1 ECI coordinates
 */
Eigen::Vector3d earth::ecefToECI(const Eigen::Vector3d& ecef, const double& t) {
    Eigen::Matrix3d C_ei = earth::rotateECEFToECI(t);
    return C_ei * ecef;
}
/**
 * @brief Converts ECEF coordinates to ECI coordinates.
 * 
 * @param ecef Eigen::Vector3d, 3x1 ECEF coordinates in meters
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @param t double, time in seconds since epoch
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d>, 3x1 ECI coordinates, 3x1 ECI velocity
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d> earth::ecefToECI(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const double& t) {
    Eigen::Matrix3d C_ei = earth::rotateECEFToECI(t);
    Eigen::Vector3d eci = C_ei * ecef;
    Eigen::Matrix3d Omega_ie = earth::vectorToSkewSymmetric(earth::RATE_VECTOR);
    Eigen::Vector3d eci_velocity = C_ei * velocity + Omega_ie * eci;
    return std::make_tuple(eci, eci_velocity);
}
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> earth::ecefToECI(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration, const double& t) {
    Eigen::Matrix3d C_ei = earth::rotateECEFToECI(t);
    Eigen::Vector3d eci = C_ei * ecef;
    Eigen::Matrix3d Omega_ie = earth::vectorToSkewSymmetric(earth::RATE_VECTOR);
    Eigen::Vector3d eci_velocity = C_ei * velocity + Omega_ie * eci;
    Eigen::Vector3d eci_acceleration = C_ei * (acceleration + 2 * Omega_ie * velocity + Omega_ie * Omega_ie * ecef);
    return std::make_tuple(eci, eci_velocity, eci_acceleration);
}
/**
 * @brief Converts latitude, longitude, and altitude (local frame) to ECEF coordinates.
 * 
 * @param lat double, latitude in degrees
 * @param lon double, longitude in degrees
 * @param alt double, altitude below sea level in meters
 * @return Eigen::Vector3d, 3x1 ECEF coordinates
 */
Eigen::Vector3d earth::nedToECEF(const double& lat, const double& lon, const double& alt){
    double alt_ = -alt;
    std::tuple<double, double, double> radii = earth::principalRadii(lat, alt_);
    double R_e = std::get<1>(radii);
    double x = (R_e + alt_) * cos(earth::DEG2RAD * lat) * cos(earth::DEG2RAD * lon);
    double y = (R_e + alt_) * cos(earth::DEG2RAD * lat) * sin(earth::DEG2RAD * lon);
    double z = (R_e * (1 - earth::ECCENTRICITY_SQUARED) + alt_) * sin(earth::DEG2RAD * lat);
    return Eigen::Vector3d(x, y, z);
}
/**
 * @brief Converts latitude, longitude, and altitude to ECEF coordinates.
 * 
 * @param lla Eigen::Vector3d, 3x1 latitude, longitude, and altitude
 * @return Eigen::Vector3d, 3x1 ECEF coordinates
 */
Eigen::Vector3d earth::nedToECEF(const Eigen::Vector3d& lla){
    return earth::nedToECEF(lla(0), lla(1), lla(2));
}
/**
 * @brief Converts latitude, longitude, and altitude to ECEF coordinates.
 * 
 * @param lla Eigen::Vector3d, 3x1 latitude, longitude, and altitude
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d>, 3x1 ECEF coordinates, 3x1 ECEF velocity
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d> earth::nedToECEF(const Eigen::Vector3d& lla, const Eigen::Vector3d& velocity){
    Eigen::Vector3d ecef = earth::nedToECEF(lla);
    Eigen::Matrix3d R = earth::rotateECEFToNED(lla(0), lla(1));
    Eigen::Vector3d ecef_velocity = R * velocity;
    return std::make_tuple(ecef, ecef_velocity);
}
/**
 * @brief Converts latitude, longitude, and altitude to ECEF coordinates.
 * 
 * @param lla Eigen::Vector3d, 3x1 latitude, longitude, and altitude
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @param acceleration Eigen::Vector3d, 3x1 acceleration in m/s^2
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>, 3x1 ECEF coordinates, 3x1 ECEF velocity, 3x1 ECEF acceleration
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> earth::nedToECEF(const Eigen::Vector3d& lla, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration) {
    Eigen::Vector3d ecef = earth::nedToECEF(lla);
    Eigen::Vector3d ned_velocity = {0, 0, velocity(2)};
    Eigen::Matrix3d R = earth::rotateECEFToNED(lla(0), lla(1));
    Eigen::Vector3d ecef_velocity = R * ned_velocity;
    Eigen::Vector3d ned_acceleration = {0, 0, acceleration(2)};
    Eigen::Vector3d ecef_acceleration = R * ned_acceleration;
    return std::make_tuple(ecef, ecef_velocity, ecef_acceleration);
}
/**
 * @brief Converts ECEF coordinates to latitude, longitude, and altitude.
 * 
 * @param ecef Eigen::Vector3d, 3x1 ECEF coordinates
 * @return Eigen::Vector3d, 3x1 latitude, longitude, and altitude
 */
Eigen::Vector3d earth::ecefToNED(const Eigen::Vector3d& ecef){
    return earth::ecefToNED(ecef(0), ecef(1), ecef(2));
}
/**
 * @brief Converts ECEF coordinates to latitude, longitude, and altitude.
 * 
 * @param x double, ECEF x-coordinate
 * @param y double, ECEF y-coordinate
 * @param z double, ECEF z-coordinate
 * @return Eigen::V ector3d, 3x1 latitude, longitude, and altitude
 */
Eigen::Vector3d earth::ecefToNED(const double& x, const double& y, const double& z){
    // Follows the approximation using equations 2.116 & 2.117
    double xi = atan2(z, sqrt(1 - earth::ECCENTRICITY_SQUARED) * sqrt(x * x + y * y));
    double sin_xi = sin(xi);
    double cos_xi = cos(xi);
    double lat = atan2(
        z * sqrt(1 - earth::ECCENTRICITY_SQUARED) + earth::ECCENTRICITY_SQUARED * earth::EQUATORIAL_RADIUS * sin_xi * sin_xi * sin_xi, 
        sqrt(1 - earth::ECCENTRICITY_SQUARED) * (sqrt(x * x + y * y) - earth::ECCENTRICITY_SQUARED * earth::EQUATORIAL_RADIUS * cos_xi * cos_xi * cos_xi)
        );
    double lon = atan2(y, x);
    std::tuple<double, double, double> radii = earth::principalRadii(lat, 0);
    double R_e = std::get<1>(radii);
    double alt = z / sin(lat) - (1 - earth::ECCENTRICITY_SQUARED) * R_e;
    return Eigen::Vector3d(lat, lon, alt);
}
/**
 * @brief Converts ECEF coordinates to latitude, longitude, and altitude.
 * 
 * @param ecef Eigen::Vector3d, 3x1 ECEF coordinates
 * @param velocity Eigen::Vector3d, 3x1 velocity in m/s
 * @return std::tuple<Eigen::Vector3d, Eigen::Vector3d>, 3x1 latitude, longitude, and altitude, 3x1 NED velocity
 */
std::tuple<Eigen::Vector3d, Eigen::Vector3d> earth::ecefToNED(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity){
    Eigen::Vector3d lla = earth::ecefToNED(ecef);
    Eigen::Matrix3d C_en = earth::rotateECEFToNED(lla(0), lla(1));
    Eigen::Vector3d ned_velocity = C_en * velocity;
    return std::make_tuple(lla, ned_velocity);
}
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> earth::ecefToNED(const Eigen::Vector3d& ecef, const Eigen::Vector3d& velocity, const Eigen::Vector3d& acceleration) {
    Eigen::Vector3d lla = earth::ecefToNED(ecef);
    Eigen::Matrix3d C_en = earth::rotateECEFToNED(lla(0), lla(1));
    Eigen::Vector3d ned_velocity = C_en * velocity;
    Eigen::Vector3d ned_acceleration = C_en * acceleration;
    return std::make_tuple(lla, ned_velocity, ned_acceleration);
}
// Earth properties
/**
 * @brief Computes the principal radii of curvature at a given latitude and longitude.
 * 
 * @param lat double, latitude in degrees 
 * @param alt double, altitude above sea level in meters
 * @return std::tuple<double, double, double>, R_n, R_e, R_p
 */
std::tuple<double, double, double> earth::principalRadii(const double& lat, const double& alt){
    double lat_rad = lat * earth::DEG2RAD;
    double sin_lat = sin(lat_rad);
    double cos_lat = cos(lat_rad);
    //double denom = 1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat;
    double Rn = (earth::EQUATORIAL_RADIUS * (1 - earth::ECCENTRICITY_SQUARED)) / pow(1 - earth::ECCENTRICITY_SQUARED * sin_lat * sin_lat, 3/2);
    double Re = earth::EQUATORIAL_RADIUS / sqrt(1 - earth::ECCENTRICITY_SQUARED * sin_lat * sin_lat);
    double Rp = Re * cos_lat + alt;
    return std::make_tuple(Rn + alt, Re + alt, Rp);
}
/**
 * @brief Computes the gravity at a given latitude using the Somigliana method and applies the Cassinis height dependence correction for altitude.
 * 
 * @param lat double, latitude in degrees
 * @param alt double, altitude above sea level in meters
 * @return double, gravity in m/s^2
 */
double earth::gravity(const double& lat, const double& alt){
    double sin_lat = sin(earth::DEG2RAD * lat);
    double g0 =  (earth::GE * (1 + earth::k * sin_lat * sin_lat)) / 
        sqrt(1 - earth::ECCENTRICITY_SQUARED * sin_lat * sin_lat);
    return g0 - 3.08e-6 * alt; // Cassinis height dependence
}
/**
 * @brief Computes the gravitational force in the ECEF frame as a vector.
 * 
 * @param lat double, NED latitude in degrees
 * @param lon double, NED longitude in degrees
 * @param alt double, NED altitude above sea level in meters
 * @return Eigen::Vector3d 
 */
Eigen::Vector3d earth::gravitation(const double& lat, const double& lon, const double& alt){
    double sin_lat = sin(earth::DEG2RAD * lat);
    double cos_lat = cos(earth::DEG2RAD * lat);
    std::tuple<double, double, double> radii = earth::principalRadii(lat, alt);
    double R_p = std::get<2>(radii);
    Eigen::Vector3d g;
    g << 0, 
        0, 
        0;
    g(0) = pow(earth::RATE, 2) * R_p * sin_lat;
    g(2) = earth::gravity(lat, alt) + pow(RATE, 2) * R_p * cos_lat;
    Eigen::Matrix3d R = earth::rotateECEFToNED(lat, lon);
    return R * g;
}
/**
 * @brief Computes the gravitational force in the ECEF frame as a vector.
 * 
 * @param lla Eigen::Vector3d, 3x1 latitude, longitude, and altitude
 * @return Eigen::Vector3d 
 */
Eigen::Vector3d earth::gravitation(const Eigen::Vector3d& lla){
    return earth::gravitation(lla(0), lla(1), lla(2));
}
/**
 * @brief Computes the Earth-rotation vector resolved into the NED frame.
 * 
 * @param lat double, degrees latitude
 * @return Eigen::Vector3d 
 */
Eigen::Vector3d earth::rateNED(const double& lat){
    double sin_lat = sin(earth::DEG2RAD * lat);
    double cos_lat = cos(earth::DEG2RAD * lat);
    Eigen::Vector3d rate;
    rate << earth::RATE * cos_lat, 
            0, 
            -earth::RATE * sin_lat;
    return rate;
}

Eigen::Vector3d earth::calculateTransportRate(const double& lat, const double& alt, const double& vel_N, const double& vel_E, const double& vel_D) {
    std::tuple<double, double, double> radii = earth::principalRadii(lat, alt);
    double R_N = std::get<0>(radii);
    double R_E = std::get<1>(radii);
    
    double lat_rad = lat * earth::DEG2RAD;
    double omega_en_n_x = vel_E / (R_N + alt);
    double omega_en_n_y = -vel_N / (R_E + alt);
    double omega_en_n_z = -vel_E * std::tan(lat_rad) / (R_N + alt);
    
    return {omega_en_n_x, omega_en_n_y, omega_en_n_z};
}
Eigen::Vector3d earth::calculateTransportRate(const Eigen::Vector3d& lla, const Eigen::Vector3d& vel) {
    return calculateTransportRate(lla(0), lla(2), vel(0), vel(1), vel(2));
}