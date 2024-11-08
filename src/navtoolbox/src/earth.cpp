#include "earth.hpp"
#include "transform.hpp"
#include "util.hpp"

/**
 * @brief Computes the principal radii of curvature of Earth ellipsoid.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return Array containing principal radii (rn, re, rp).
 */
std::tuple<double, double, double> earth::principal_radii(double lat, double alt) {
    double sin_lat = std::sin(lat * M_PI / 180.0);
    double cos_lat = std::sqrt(1 - sin_lat * sin_lat);
    double x = 1 - E2 * sin_lat * sin_lat;
    double re = A / std::sqrt(x);
    double rn = re * (1 - E2) / x;
    return {rn + alt, re + alt, (re + alt) * cos_lat};
}

/**
 * @brief Computes the gravity magnitude using the Somigliana model with linear altitude correction.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return Gravity magnitude in m/s^2.
 */
double earth::gravity(double lat, double alt) {
    double sin_lat = std::sin(lat * M_PI / 180.0);
    return (GE * (1 + F * sin_lat * sin_lat) / std::sqrt(1 - E2 * sin_lat * sin_lat) *
            (1 - 2 * alt / A));
}

/**
 * @brief Computes the gravity vector in the NED (North-East-Down) frame.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return 3D vector (Eigen::Vector3d) of gravity in the NED frame.
 */
Eigen::Vector3d earth::gravity_n(double lat, double alt) {
    double g = gravity(lat, alt);
    return Eigen::Vector3d(0, 0, g);
}

/**
 * @brief Computes the gravitational force vector in ECEF frame.
 * Accounts for Earth mass attraction and eliminates centrifugal force.
 * @param lla Array containing latitude (degrees), longitude (degrees), and altitude (meters).
 * @return 3D gravitational force vector (Eigen::Vector3d) in the ECEF frame.
 */

Eigen::Vector3d earth::gravitation_ecef(const std::array<double, 3>& lla) {
    double lat = lla[0], lon = lla[1], alt = lla[2];

    double sin_lat = std::sin(lat * M_PI / 180.0);
    double cos_lat = std::cos(lat * M_PI / 180.0);

    auto [rn, re, rp] = principal_radii(lat, alt);

    Eigen::Vector3d g0_g;
    g0_g[0] = RATE * RATE * rp * sin_lat;
    g0_g[2] = gravity(lat, alt) + RATE * RATE * rp * cos_lat;

    // Assuming mat_en_from_ll and mv_prod are utility functions in separate namespaces
    Eigen::Matrix3d mat_eg = transform::mat_en_from_ll(lat, lon);
    return util::mv_prod(mat_eg, g0_g);
}


/**
 * @brief Computes the Earth curvature matrix.
 * Links linear displacement and angular rotation of NED frame.
 * @param lat Latitude in degrees.
 * @param alt Altitude in meters.
 * @return 3x3 curvature matrix (Eigen::Matrix3d).
 */
Eigen::Matrix3d earth::curvature_matrix(double lat, double alt) {
    auto [rn, re, _] = principal_radii(lat, alt);

    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
    F(0, 1) = 1 / re;
    F(1, 0) = -1 / rn;
    F(2, 1) = -F(0, 1) * std::tan(lat * M_PI / 180.0);
    return F;
}

/**
 * @brief Computes Earth rate in the NED frame.
 * @param lat Latitude in degrees.
 * @return Earth rate vector (Eigen::Vector3d) in NED components.
 */
Eigen::Vector3d earth::rate_n(double lat) {
    Eigen::Vector3d earth_rate_n = Eigen::Vector3d::Zero();
    earth_rate_n[0] = RATE * std::cos(lat * M_PI / 180.0);
    earth_rate_n[2] = -RATE * std::sin(lat * M_PI / 180.0);
    return earth_rate_n;
}

void earth::say_hello() {
    std::cout << "Hello from Earth!" << std::endl;
}
