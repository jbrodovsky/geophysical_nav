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

#include <Eigen/Core>

// Working notes:
// - Thinking about using a minimal object-oriented approach to this implementation.
// - The class should serve as a base navigation state object that can be updated with IMU measurements.
// - The class should be able to propagate the state forward in time.

class NavigationState
{
public:
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d orientation;
    // Constructor
    NavigationState() = default;
    NavigationState(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity, const Eigen::Vector3d& orientation);
    NavigationState(double latitude, double longitude, double altitude, double velocity_north, double velocity_east, double velocity_down, double roll, double pitch, double yaw);
    // Public methods
    void integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, const double& dt);
    Eigen::VectorXd getState() const noexcept;
    Eigen::Matrix3d getRotationMatrix() const noexcept;
    std::string toString() const;

private:
    void attitudeUpdate(const Eigen::Vector3d& gyro, const double& dt);
    void velocityUpdate(const Eigen::Vector3d& accel, const double& dt);
    void positionUpdate(const Eigen::Vector3d& accel, const double& dt);
};