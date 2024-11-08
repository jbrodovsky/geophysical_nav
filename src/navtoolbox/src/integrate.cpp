#include <Eigen/Dense>
#include <cmath>
#include "earth.hpp"      // Assuming earth constants like earth::E2, earth::RATE
#include "transform.hpp"  // Assuming mat_from_rotvec is implemented here
#include "integrate.hpp"

/**
 * @brief Constructor to initialize the strapdown integrator with initial states.
 */
StrapdownIntegrator::StrapdownIntegrator(double init_lat, double init_lon, double init_alt,
                                         double init_vn, double init_ve, double init_vd,
                                         double init_roll, double init_pitch, double init_yaw)
    : lat(init_lat), lon(init_lon), alt(init_alt),
      vn(init_vn), ve(init_ve), vd(init_vd),
      roll(init_roll), pitch(init_pitch), yaw(init_yaw) {}

/**
 * @brief Perform a single integration step of the strapdown INS algorithm.
 */
void StrapdownIntegrator::integrate(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel) {
    // Convert angles from degrees to radians for internal calculations
    double lat_rad = lat * M_PI / 180.0;
    double lon_rad = lon * M_PI / 180.0;
    double roll_rad = roll * M_PI / 180.0;
    double pitch_rad = pitch * M_PI / 180.0;
    double yaw_rad = yaw * M_PI / 180.0;

    // Initial rotation matrix from NED to body frame using roll, pitch, yaw
    Eigen::Quaterniond q = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX());
    Eigen::Matrix3d Cnb = q.toRotationMatrix();

    // Update orientation using gyroscope data
    Eigen::Vector3d rot_vec = gyro * dt;           // Rotation vector
    Eigen::Matrix3d dCnb = transform::mat_from_rotvec(rot_vec); // Rotation increment matrix
    Cnb = dCnb * Cnb;                                // Update rotation matrix

    // Extract updated roll, pitch, yaw from rotation matrix
    Eigen::Vector3d rpy = Cnb.eulerAngles(2, 1, 0); // ZYX convention
    roll = rpy[2] * 180.0 / M_PI;
    pitch = rpy[1] * 180.0 / M_PI;
    yaw = rpy[0] * 180.0 / M_PI;

    // Calculate specific force in NED frame
    Eigen::Vector3d fn = Cnb.transpose() * accel;

    // Gravity calculation using latitude and altitude
    double sin_lat = std::sin(lat_rad);
    // double g = earth::GE * (1 + earth::F * sin_lat * sin_lat) /
    //           std::sqrt(1 - earth::E2 * sin_lat * sin_lat);
    //g = earth::gravity(lat, alt);
    Eigen::Vector3d gravity = earth::gravity_n(lat, alt);

    // Compute transport rate and Earth rate in NED frame
    double re = earth::A / std::sqrt(1 - earth::E2 * sin_lat * sin_lat);
    double rn = re * (1 - earth::E2) / (1 - earth::E2 * sin_lat * sin_lat);

    double omega_en_north = ve / (re + alt);
    double omega_en_up = vn / (rn + alt);
    Eigen::Vector3d omega_en_n(omega_en_north, 0, -omega_en_up); // Transport rate

    double omega_earth = earth::RATE;
    Eigen::Vector3d omega_ie_n(omega_earth * std::cos(lat_rad), 0, -omega_earth * std::sin(lat_rad)); // Earth rate in NED

    // Integrate velocity using specific force and Coriolis/centrifugal forces
    Eigen::Vector3d coriolis = 2 * omega_ie_n + omega_en_n;
    vn += (fn[0] - coriolis[1] * vd + coriolis[2] * ve - gravity[0]) * dt;
    ve += (fn[1] + coriolis[2] * vn - coriolis[0] * vd - gravity[1]) * dt;
    vd += (fn[2] - coriolis[1] * vn + coriolis[0] * ve - gravity[2]) * dt;

    // Integrate position (latitude, longitude, altitude) using updated velocity
    lat += vn / (rn + alt) * dt * 180.0 / M_PI;         // Convert to degrees
    lon += ve / ((re + alt) * std::cos(lat_rad)) * dt * 180.0 / M_PI; // Convert to degrees
    alt -= vd * dt;                                     // Down is positive

    // Normalize yaw angle to [0, 360] degrees for consistency
    if (yaw < 0) yaw += 360;
    else if (yaw >= 360) yaw -= 360;
}
