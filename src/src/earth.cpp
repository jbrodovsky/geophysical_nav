#include "Eigen/Geometry"
#include "earth.hpp"

Eigen::Matrix3d ECIToECEF(const double& t) {
    // Rotation matrix from ECI to ECEF
    // t: time in seconds since epoch
    // omega_ie: Earth rotation rate (rad/s)
    // R: Earth radius (m)
    // Returns: 3x3 rotation matrix from ECI to ECEF
    Eigen::Matrix3d R;
    R << cos(RATE) * t, sin(RATE) * t, 0,
         -sin(RATE) * t, cos(RATE) * t, 0,
         0, 0, 1;
    return R;
}
Eigen::Matrix3d ECEFToNED(const double& lat, const double& lon) {
    // Rotation matrix from ECEF to NED
    // lat: latitude (rad)
    // lon: longitude (rad)
    // Returns: 3x3 rotation matrix from ECEF to NED
    Eigen::Matrix3d R;
    R << -sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat),
         -sin(lon), cos(lon), 0,
         -cos(lat) * cos(lon), -cos(lat) * sin(lon), -sin(lat);
    return R;
}
Eigen::Matrix3d NEDToBody(const double& roll, const double& pitch, const double& yaw) {
    // Rotation matrix from NED to body
    // roll: roll angle (rad)
    // pitch: pitch angle (rad)
    // yaw: yaw angle (rad)
    // Returns: 3x3 rotation matrix from NED to body using a yaw-pitch-roll sequence
    Eigen::Matrix3d R;
    //R << cos(pitch) * cos(yaw), sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw), cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw),
    //     cos(pitch) * sin(yaw), sin(roll) * sin(pitch) * sin(yaw) + cos(roll) * cos(yaw), cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw),
    //     -sin(pitch), sin(roll) * cos(pitch), cos(roll) * cos(pitch);
    R = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
    
    return R;
}
// Inverse rotations
Eigen::Matrix3d ECEFToECI(const double& t) {
    // Rotation matrix from ECEF to ECI
    // t: time in seconds since epoch
    // Returns: 3x3 rotation matrix from ECEF to ECI
    return ECIToECEF(t).transpose();
}
Eigen::Matrix3d NEDToECEF(const double& lat, const double& lon) {
    // Rotation matrix from NED to ECEF
    // lat: latitude (rad)
    // lon: longitude (rad)
    // Returns: 3x3 rotation matrix from NED to ECEF
    return ECEFToNED(lat, lon).transpose();
}
Eigen::Matrix3d BodyToNED(const double& roll, const double& pitch, const double& yaw) {
    // Rotation matrix from body to NED
    // roll: roll angle (rad)
    // pitch: pitch angle (rad)
    // yaw: yaw angle (rad)
    // Returns: 3x3 rotation matrix from body to NED
    return NEDToBody(roll, pitch, yaw).transpose();
}
// Other useful transformations
Eigen::Matrix3d vectorToSkewSymmetric(const Eigen::Vector3d& v) {
    // Convert a vector to a skew-symmetric matrix
    // v: 3x1 vector
    // Returns: 3x3 skew-symmetric matrix
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return m;
}
Eigen::Vector3d skewSymmetricToVector(const Eigen::Matrix3d& m) {
    // Convert a skew-symmetric matrix to a vector
    // m: 3x3 skew-symmetric matrix
    // Returns: 3x1 vector
    Eigen::Vector3d v;
    v << m(2, 1), m(0, 2), m(1, 0);
    return v;
}
Eigen::Vector3d llaToECEF(const double& lat, const double& lon, const double& alt){
    double sin_lat = sin(DEG2RAD * lat);
    double cos_lat = cos(DEG2RAD * lat);
    double sin_lon = sin(DEG2RAD * lon);
    double cos_lon = cos(DEG2RAD * lon);
    double denom = pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 0.5);
    double R_n = EQUATORIAL_RADIUS / denom;
    double R_e = EQUATORIAL_RADIUS * (1 - ECCENTRICITY_SQUARED) / denom;
    double x = (R_n + alt) * cos_lat * cos_lon;
    double y = (R_n + alt) * cos_lat * sin_lon;
    double z = (R_e + alt) * sin_lat;
    Eigen::Vector3d ecef;
    ecef << x, y, z;
    return ecef;
}
Eigen::Vector3d ecefToLLA(const Eigen::Vector3d& ecef){
    double x = ecef(0);
    double y = ecef(1);
    double z = ecef(2);
    double p = pow(x * x + y * y, 0.5);
    double theta = atan2(z * EQUATORIAL_RADIUS, p * EQUATORIAL_RADIUS * (1 - ECCENTRICITY_SQUARED));
    double lat = atan2(z + ECCENTRICITY_SQUARED * EQUATORIAL_RADIUS * pow(sin(theta), 3), p - ECCENTRICITY_SQUARED * EQUATORIAL_RADIUS * pow(cos(theta), 3));
    double lon = atan2(y, x);
    double sin_lat = sin(lat);
    //double cos_lat = cos(lat);
    double denom = pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 0.5);
    double R_n = EQUATORIAL_RADIUS / denom;
    double alt = p / cos(lat) - R_n;
    return Eigen::Vector3d(lat, lon, alt);
}
// Earth properties
/**
 * @brief Computes the principal radii of curvature at a given latitude and longitude.
 * 
 * @param lat double, latitude in degrees 
 * @param alt double, altitude in meters
 * @return std::tuple<double, double, double>, R_n, R_e, R_p
 */
std::tuple<double, double, double> principalRadii(const double& lat, const double& alt){
    double lat_rad = lat * DEG2RAD;
    double sin_lat = sin(lat_rad);
    double cos_lat = cos(lat_rad);
    //double denom = 1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat;
    double Rn = (EQUATORIAL_RADIUS * (1 - ECCENTRICITY_SQUARED)) / pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 3/2);
    double Re = EQUATORIAL_RADIUS / sqrt(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat);
    double Rp = Re * cos_lat + alt;
    return std::make_tuple(Rn + alt, Re + alt, Rp);
}
/**
 * @brief Computes the gravity at a given latitude and altitude using the Somigliana method.
 * 
 * @param lat double, latitude in degrees
 * @param alt double, altitude in meters
 * @return double, gravity in m/s^2
 */
double gravity(const double& lat, const double& alt){
    double sin_lat = sin(DEG2RAD * lat);
    return (GE * (1 + F * sin_lat * sin_lat)) / 
           (pow(1 - ECCENTRICITY_SQUARED * sin_lat * sin_lat, 0.5) * (1 - 2 * alt / EQUATORIAL_RADIUS));
}
/**
 * @brief Computes the gravitational force in the ECEF frame as a vector.
 * 
 * @param lat double, latitude in degrees
 * @param lon double, longitude in degrees
 * @param alt double, altitude in meters
 * @return Eigen::Vector3d 
 */
Eigen::Vector3d gravitation(const double& lat, const double& lon, const double& alt){
    double sin_lat = sin(DEG2RAD * lat);
    double cos_lat = cos(DEG2RAD * lat);
    std::tuple<double, double, double> radii = principalRadii(lat, alt);
    //double R_n = std::get<0>(radii);
    //double R_e = std::get<1>(radii);
    double R_p = std::get<2>(radii);

    Eigen::Vector3d g;
    g << 0, 
         0, 
         0;
    g(0) = pow(RATE, 2) * R_p * sin_lat;
    g(2) = gravity(lat, alt) + pow(RATE, 2) * R_p * cos_lat;
    Eigen::Matrix3d R = ECEFToNED(lat, lon);
    return R * g;
}
Eigen::Vector3d rateNED(const double& lat){
    double sin_lat = sin(DEG2RAD * lat);
    double cos_lat = cos(DEG2RAD * lat);
    Eigen::Vector3d rate;
    rate << RATE * cos_lat, 
            0, 
            -RATE * sin_lat;
    return rate;
}