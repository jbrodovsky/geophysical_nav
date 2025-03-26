/**
 * @file strapdown.hpp
 * @author James Brodovsky (james.brodovsky@gmail.com)
 * @brief Strapdown navigation equations for progagating position, velocity, and orientation from IMU measurements.
 * @version 0.1
 * @date 2025-03-04
 * @details
 * This file contains the implementation details for the strapdown navigation equations implemented in the Local 
 * Navigation Frame. The equations are based on the book "Principles of GNSS, Inertial, and Multisensor Integrated 
 * Navigation Systems, Second Edition" by Paul D. Groves. This file corresponds to Chapter 5.4 and 5.5 of the book. 
 * Effort has been made to reproduce most of the equations following the notation from the book. However, variable 
 * and constants should generally been named for the quatity they represent rather than the symbol used in the book.
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <iostream>
#include <string>
#include <cmath>

#include <Eigen/Dense>

// Working notes:
// - Shifting implementation to a more functional programming style.
// - The NavigationState struct will be used to store the state of the navigation system.

struct IMUData {
    double gyro_x, gyro_y, gyro_z;
    double accel_x, accel_y, accel_z;
};

struct NavData {
    double latitude, longitude, altitude;
    double roll, pitch, yaw;
    double vel_N, vel_E, vel_D;
    double timestamp;
};

IMUData inverse_mechanization(const NavData& prev, const NavData& curr, const double& dt);
std::vector<IMUData> inverse_mechanization(const std::vector<NavData>& nav_data, const std::vector<double>& timestamps);
std::vector<IMUData> inverse_mechanization(const std::vector<double>& latitudes, const std::vector<double>& longitudes, const std::vector<double>& altitudes);
std::vector<IMUData> inverse_mechanization(const std::vector<double>& latitudes, const std::vector<double>& longitudes, const std::vector<double>& altitudes, const std::vector<double>& rolls, const std::vector<double>& pitches, const std::vector<double>& yaws);
NavData forward_mechanization(const std::vector<IMUData>& imu_data, const NavData& initial_state, const double& dt);
std::vector<NavData> forward_mechanization(const std::vector<IMUData>& imu_data, const NavData& initial_state, const std::vector<double>& timestamps);
