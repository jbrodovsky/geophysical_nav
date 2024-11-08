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

std::vector<Eigen::Vector3d> transform::lla_to_ecef(const std::vector<Eigen::Vector3d>& lla_stack) {
    std::vector<Eigen::Vector3d> ecef_stack;
    ecef_stack.reserve(ecef_stack.size());

    for (const auto& r_e : ecef_stack) {
        ecef_stack.push_back(ecef_to_lla(r_e));
    }

    return ecef_stack;
}

Eigen::Vector3d transform::ecef_to_lla(const Eigen::Vector3d& r_e) {
    const double a1 = earth::A * earth::E2;
    const double a2 = a1 * a1;
    const double a3 = a1 * earth::E2 / 2;
    const double a4 = 2.5 * a2;
    const double a5 = a1 + a3;
    const double a6 = 1 - earth::E2;

    double w = std::hypot(r_e[0], r_e[1]);
    double z = r_e[2];
    double zp = std::abs(z);
    double w2 = w * w;
    double r2 = z * z + w2;
    double r = std::sqrt(r2);
    double s2 = (z * z) / r2;
    double c2 = w2 / r2;
    double u = a2 / r;
    double v = a3 - a4 / r;

    double s = (zp / r) * (1 + c2 * (a1 + u + s2 * v) / r);
    double c = (w / r) * (1 - s2 * (a5 - u - c2 * v) / r);

    double lat;
    if (c2 > 0.3) {
        lat = std::asin(s);
    } else {
        lat = std::acos(c);
        s = std::sqrt(1 - c * c);
    }

    double ss = s * s;
    c = std::sqrt(1 - ss);
    double g = 1 - earth::E2 * ss;
    double rg = earth::A / std::sqrt(g);
    double rf = a6 * rg;
    u = w - rg * c;
    v = zp - rf * s;
    double f = c * u + s * v;
    double m = c * v - s * u;
    double p = m / ((rf / g) + f);
    lat += p;
    if (z < 0) lat = -lat;

    lat *= RAD_TO_DEG;
    double lon = std::atan2(r_e[1], r_e[0]) * RAD_TO_DEG;
    double alt = f + 0.5 * m * p;

    return Eigen::Vector3d(lat, lon, alt);
}

std::vector<Eigen::Vector3d> transform::ecef_to_lla(const std::vector<Eigen::Vector3d>& r_e_stack) {
    std::vector<Eigen::Vector3d> lla_stack;
    lla_stack.reserve(r_e_stack.size());

    for (const auto& r_e : r_e_stack) {
        lla_stack.push_back(ecef_to_lla(r_e));
    }

    return lla_stack;
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

Eigen::Matrix3d transform::mat_from_rotvec(const Eigen::Vector3d& rv) {
    Eigen::Matrix3d mat; // Rotation matrix to be returned
    double norm2 = rv.squaredNorm(); // Compute the squared norm (angle^2)

    if (norm2 > 1e-6) {
        double norm = std::sqrt(norm2);      // Compute the norm (angle)
        double cos_angle = std::cos(norm);   // Cosine of the angle
        double k1 = std::sin(norm) / norm;   // sin(angle) / angle
        double k2 = (1 - cos_angle) / norm2; // (1 - cos(angle)) / (angle^2)

        mat(0, 0) = k2 * rv[0] * rv[0] + cos_angle;
        mat(0, 1) = k2 * rv[0] * rv[1] - k1 * rv[2];
        mat(0, 2) = k2 * rv[0] * rv[2] + k1 * rv[1];
        mat(1, 0) = k2 * rv[1] * rv[0] + k1 * rv[2];
        mat(1, 1) = k2 * rv[1] * rv[1] + cos_angle;
        mat(1, 2) = k2 * rv[1] * rv[2] - k1 * rv[0];
        mat(2, 0) = k2 * rv[2] * rv[0] - k1 * rv[1];
        mat(2, 1) = k2 * rv[2] * rv[1] + k1 * rv[0];
        mat(2, 2) = k2 * rv[2] * rv[2] + cos_angle;
    } else {
        // Taylor series expansion for small angles
        double norm4 = norm2 * norm2;
        double cos_angle = 1 - norm2 / 2 + norm4 / 24;
        double k1 = 1 - norm2 / 6 + norm4 / 120;
        double k2 = 0.5 - norm2 / 24 + norm4 / 720;

        mat(0, 0) = k2 * rv[0] * rv[0] + cos_angle;
        mat(0, 1) = k2 * rv[0] * rv[1] - k1 * rv[2];
        mat(0, 2) = k2 * rv[0] * rv[2] + k1 * rv[1];
        mat(1, 0) = k2 * rv[1] * rv[0] + k1 * rv[2];
        mat(1, 1) = k2 * rv[1] * rv[1] + cos_angle;
        mat(1, 2) = k2 * rv[1] * rv[2] - k1 * rv[0];
        mat(2, 0) = k2 * rv[2] * rv[0] - k1 * rv[1];
        mat(2, 1) = k2 * rv[2] * rv[1] + k1 * rv[0];
        mat(2, 2) = k2 * rv[2] * rv[2] + cos_angle;
    }

    return mat;
}

Eigen::Vector3d transform::rotvec_from_mat(const Eigen::Matrix3d& mat) {
    Eigen::Vector3d rv; // Rotation vector to be returned

    // Compute the angle from the trace of the rotation matrix
    double cos_angle = (mat.trace() - 1.0) / 2.0;
    double angle = std::acos(std::clamp(cos_angle, -1.0, 1.0));  // Clamp to avoid numerical issues

    if (angle < 1e-6) {
        // If the angle is very small, the rotation vector is nearly zero
        rv.setZero();
    } else if (std::abs(angle - M_PI) < 1e-6) {
        // If the angle is close to 180 degrees (pi radians), handle the case carefully
        angle = M_PI;
        Eigen::Vector3d axis;
        axis[0] = std::sqrt((mat(0, 0) + 1) / 2);
        axis[1] = std::sqrt((mat(1, 1) + 1) / 2);
        axis[2] = std::sqrt((mat(2, 2) + 1) / 2);
        axis = axis.array() * ((mat.col(1).cross(mat.col(2)).array() >= 0).cast<double>() * 2 - 1).array(); // Correct sign

        rv = angle * axis;
    } else {
        // General case: angle between 0 and pi
        double sin_angle = std::sin(angle);
        Eigen::Vector3d axis;
        axis << (mat(2, 1) - mat(1, 2)),
                (mat(0, 2) - mat(2, 0)),
                (mat(1, 0) - mat(0, 1));
        axis /= (2.0 * sin_angle);
        rv = angle * axis;
    }

    return rv;
}


void transform::say_hello() {
    std::cout << "Hello from transform!" << std::endl;
}

