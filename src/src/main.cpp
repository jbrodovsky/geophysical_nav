#include <iostream>

#include "Eigen/Core"

#include "earth.hpp"

int main() {
    std::cout << "Hello from geophysical-nav!" << std::endl;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;
    Eigen::Matrix3d nedToBody = NEDToBody(roll, pitch, yaw);
    std::cout << "NED to Body rotation matrix: " << std::endl << nedToBody << std::endl;
    Eigen::Vector3d lla = {40.03214, -75.22148, 40.0};
    Eigen::Matrix3d skew = vectorToSkewSymmetric(lla);
    std::cout << "Skew symmetric matrix: " << std::endl << skew << std::endl;
    Eigen::Vector3d skewVec = skewSymmetricToVector(skew);
    std::cout << "Skew symmetric vector: " << std::endl << skewVec << std::endl;
    Eigen::Vector3d ecef = llaToECEF(lla(0), lla(1), lla(2));
    std::cout << "ECEF: " << std::endl << ecef << std::endl;
    std::tuple<double, double, double> radii = principalRadii(lla(0), lla(2));
    std::cout << "Principal radii: " << std::get<0>(radii) << ", " << std::get<1>(radii) << ", " << std::get<2>(radii) << std::endl;
    double g = gravity(lla(0), lla(2));
    std::cout << "Gravity: " << g << std::endl;
    Eigen::Vector3d grav = gravitation(lla(0), lla(1), lla(2));
    std::cout << "Gravitation: " << std::endl << grav << std::endl;
    Eigen::Vector3d rate = rateNED(lla(0));
    std::cout << "Rate NED: " << std::endl << rate << std::endl;
    
    return 0;
}