#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "transform.hpp"
#include "earth.hpp"

Eigen::Vector3d transform::lla_to_ecef(const Eigen::Vector3d& lla) {
    double lat = lla[0] * DEG_TO_RAD;
    double lon = lla[1] * DEG_TO_RAD;
    double alt = lla[2];

    double sin_lat = std::sin(lat);
    double cos_lat = std::cos(lat);
    double sin_lon = std::sin(lon);
    double cos_lon = std::cos(lon);

    auto [rn, re, rp] = earth::principal_radii(lat, alt);

    Eigen::Vector3d r_ecef;
    r_ecef[0] = (re + alt) * cos_lat * cos_lon;
    r_ecef[1] = (re + alt) * cos_lat * sin_lon;
    r_ecef[2] = ((1 - earth::E2) * re + alt) * sin_lat;
    return r_ecef;
}

Eigen::Vector3d transform::ecef_to_lla(const Eigen::Vector3d& r_e) {
    double a = earth::A;
    double e2 = earth::E2;

    double x = r_e[0];
    double y = r_e[1];
    double z = r_e[2];
    double lon = std::atan2(y, x);

    double r = std::sqrt(x * x + y * y);
    double lat = std::atan2(z, r * (1 - e2));
    double prev_lat;
    do {
        prev_lat = lat;
        auto [rn, re, rp] = earth::principal_radii(lat * RAD_TO_DEG, r_e(2));
        lat = std::atan2(z + e2 * rn * std::sin(lat), r);
    } while (std::abs(lat - prev_lat) > 1e-12);
    std::tuple radii = earth::principal_radii(lat * RAD_TO_DEG, 0);
    double alt = r / std::cos(lat) - std::get<1>(radii);
    return Eigen::Vector3d(lat * RAD_TO_DEG, lon * RAD_TO_DEG, alt);
}

Eigen::Vector3d transform::lla_to_ned(const Eigen::Vector3d& lla, const Eigen::Vector3d& lla_origin) {
    Eigen::Vector3d delta_ecef = lla_to_ecef(lla) - lla_to_ecef(lla_origin);
    Eigen::Matrix3d mat_en = mat_en_from_ll(lla_origin[0], lla_origin[1]);
    return mat_en * delta_ecef;
}

Eigen::Vector3d transform::ned_to_lla(const Eigen::Vector3d& ned, const Eigen::Vector3d& lla_origin) {
    Eigen::Matrix3d mat_en = mat_en_from_ll(lla_origin[0], lla_origin[1]);
    Eigen::Matrix3d mat_ne = mat_en.transpose();
    Eigen::Vector3d delta_ecef = mat_ne * ned;
    return ecef_to_lla(lla_to_ecef(lla_origin) + delta_ecef);
}
Eigen::Matrix3d transform::mat_en_from_ll(double lat, double lon) {
    double sin_lat = std::sin(lat * DEG_TO_RAD);
    double cos_lat = std::cos(lat * DEG_TO_RAD);
    double sin_lon = std::sin(lon * DEG_TO_RAD);
    double cos_lon = std::cos(lon * DEG_TO_RAD);

    Eigen::Matrix3d mat;
    mat << -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat,
           -sin_lon, cos_lon, 0,
           -cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat;
    return mat;
}

Eigen::Matrix3d transform::mat_from_rph(const Eigen::Vector3d& rph) {
    Eigen::Matrix3d roll = Eigen::AngleAxisd(rph[0] * DEG_TO_RAD, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Matrix3d pitch = Eigen::AngleAxisd(rph[1] * DEG_TO_RAD, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d heading = Eigen::AngleAxisd(rph[2] * DEG_TO_RAD, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return heading * pitch * roll;
}

Eigen::Matrix3d transform::mat_from_rph(double roll, double pitch, double heading) {
    return mat_from_rph(Eigen::Vector3d(roll, pitch, heading));
}

Eigen::Vector3d transform::mat_to_rph(const Eigen::Matrix3d& mat) {
    Eigen::Vector3d rph = mat.eulerAngles(2, 1, 0);  // ZYX convention: heading, pitch, roll
    return rph * RAD_TO_DEG;
}

void transform::say_hello() {
    std::cout << "Hello from transform!" << std::endl;
}

