#include <tuple>
#include <cstddef>

#include "strapdown.hpp"
#include "earth.hpp"

/**
 * @brief Compute the inverse mechanization of the strapdown navigation equations. Requires that the previous and current navigation data have velocities in the NED frame in addition to the standard position and attitude values.
 * 
 * @param prev The previous navigation data
 * @param curr The current navigation data
 * @param dt The time difference between the previous and current navigation data. For INCREMENT calculations, dt should be 1. For RATE calculations, dt should be the time difference between the previous and current navigation data.
 * @return IMUData 
 */
IMUData inverse_mechanization(const NavData& prev, const NavData& curr, const double& dt) {
    Eigen::Vector3d accelerations((curr.vel_N - prev.vel_N) / dt, (curr.vel_E - prev.vel_E) / dt, (curr.vel_D - prev.vel_D) / dt);
    Eigen::Matrix3d C_n_b = earth::rotateNEDToBody(curr.roll, curr.pitch, curr.yaw);    // Attitude DCM
    Eigen::Vector3d transport_rate = earth::calculateTransportRate(curr.latitude, curr.altitude, curr.vel_N, curr.vel_E, curr.vel_D); // omega_en_n
    Eigen::Vector3d gyros = C_n_b * (earth::RATE_VECTOR + transport_rate);
    Eigen::Vector3d specific_forces = C_n_b * (accelerations + earth::gravitation(curr.latitude, curr.longitude, curr.altitude));
    return {gyros(0), gyros(1), gyros(2), specific_forces(0), specific_forces(1), specific_forces(2)};
}

std::vector<IMUData> inverse_mechanization(std::vector<NavData>& nav_data, const std::vector<double>& timestamps) {
    std::vector<IMUData> imu_data;
    if (nav_data.size() < 2 || nav_data.size() != timestamps.size()) {
        imu_data.push_back({0, 0, 0, 0, 0, 0});
        return imu_data;
    }
    for (std::size_t i = 1; i < nav_data.size(); ++i) {
        const NavData& prev = nav_data[i - 1];
        NavData& curr = nav_data[i];
        double dt = timestamps[i] - timestamps[i - 1];
        if (dt <= 0) continue;
        // Check to make sure all the velocities are not null
        if (curr.vel_N == 0 && curr.vel_E == 0 && curr.vel_D == 0) {
            std::tuple<double, double, double> radii = earth::principalRadii(curr.latitude, curr.altitude);
            double R_N = std::get<0>(radii);
            double R_E = std::get<1>(radii);
            curr.vel_N = ((curr.latitude - prev.latitude) / dt) * (R_N + curr.altitude);
            curr.vel_E = ((curr.longitude - prev.longitude) / dt) * (R_E + curr.altitude) * cos(curr.latitude * M_PI / 180);
            curr.vel_D = (curr.altitude - prev.altitude) / dt;
        }
        // This assumes that we have not been provided attitude data. While NED velocities 
        // can still be calculated using position data, to get the gyro and accelerometer 
        // data, we need attitude data. We can only recover the yaw data from NED position.
        // As such, we will not attempt to change the roll or pitch data in the case that
        // the actuall trajectory goes through (0, 0, 0) point.
        if (prev.roll == 0 && prev.pitch == 0 && prev.yaw == 0) {
            double lon_diff = (curr.longitude - prev.longitude) * M_PI / 180 ;
            double x = cos(curr.latitude * M_PI / 180) * sin(lon_diff);
            double y = cos(prev.latitude * M_PI / 180) * sin(curr.latitude * M_PI / 180) - sin(prev.latitude * M_PI / 180) * cos(curr.latitude * M_PI / 180) * cos(lon_diff);
            curr.yaw = atan2(x, y) * 180 / M_PI;
        }
        imu_data.push_back(inverse_mechanization(prev, curr, dt));
    }    
    return imu_data;
}

NavData forward_mechanization(const IMUData& imu_data, const NavData& initial_state, const double& dt) {
    std::tuple<double, double, double> radii = earth::principalRadii(initial_state.latitude, initial_state.altitude);
    double R_N = std::get<0>(radii);
    double R_E = std::get<1>(radii);
    
    // Convert angles to radians
    // double roll = initial_state.roll * M_PI / 180.0;
    // double pitch = initial_state.pitch * M_PI / 180.0;
    // double yaw = initial_state.yaw * M_PI / 180.0;
    
    // Rotation matrix from NED to Body
    Eigen::Matrix3d C_n_b = earth::rotateNEDToBody(initial_state.roll, initial_state.pitch, initial_state.yaw);
    
    // Compute body angular velocity
    Eigen::Vector3d omega_ib_b(imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z);
    omega_ib_b *= dt;
    
    // Compute transport rate
    Eigen::Vector3d omega_en_n = earth::calculateTransportRate(initial_state.latitude, 
        initial_state.altitude, initial_state.vel_N, initial_state.vel_E, initial_state.vel_D);
    
    // Compute Earth's rotation rate in navigation frame
    Eigen::Vector3d omega_ie_n = earth::rateNED(initial_state.latitude);
    
    // Compute total angular velocity in navigation frame in dcm form
    Eigen::Matrix3d omega_total_n = (earth::vectorToSkewSymmetric(omega_ie_n) + earth::vectorToSkewSymmetric(omega_en_n));
    
    // Update attitude dcm
    Eigen::Matrix3d C = C_n_b * (Eigen::Matrix3d::Identity() + earth::vectorToSkewSymmetric(omega_total_n) * dt) - omega_total_n * C_n_b * dt;
    
    // Compute accelerometer specific forces
    Eigen::Vector3d f_(imu_data.accel_x, imu_data.accel_y, imu_data.accel_z);
    Eigen::Vector3d f_b = 0.5 * (C_n_b + C) * f_;
    
    // Compute angular velocity in body frame
    Eigen::Vector3d omega_ib_n = C_n_b.transpose() * omega_ib_b;
    
    // Compute angular velocity in navigation frame
    Eigen::Vector3d omega_in_n = omega_total_n + omega_ib_n;
    
    // Compute angular velocity in body frame
    Eigen::Vector3d omega_in_b = C_n_b * omega_in_n;
    
    // Compute specific forces in navigation frame
    Eigen::Vector3d f_n = C_n_b.transpose() * f_b;
}
std::vector<NavData> forward_mechanization(const std::vector<IMUData>& imu_data, const NavData& initial_state, const std::vector<double>& timestamps) {
    std::vector<NavData> nav_data;
    nav_data.push_back(initial_state);
    if (imu_data.size() != timestamps.size() || imu_data.size() <= 2) { return nav_data; }
    for (std::size_t i = 0; i < imu_data.size(); ++i) {
        const auto& prev = nav_data.back();
        const auto& imu = imu_data[i];
        double dt = timestamps[i] - timestamps[i - 1];
        if (dt <= 0) continue;
        nav_data.push_back(forward_mechanization(imu, prev, dt));
    }
}

