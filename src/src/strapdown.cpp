#include "strapdown.hpp"
#include "earth.hpp"

// Constructors
NavigationState::NavigationState()
{
    position = Eigen::Vector3d::Zero();
    velocity = Eigen::Vector3d::Zero();
    orientation = Eigen::Vector3d::Zero();
}
NavigationState::NavigationState(const Eigen::Vector3d& position, const Eigen::Vector3d& velocity, const Eigen::Vector3d& orientation)
{
    this->position = position;
    this->velocity = velocity;
    this->orientation = orientation;
}
NavigationState::NavigationState(double latitude, double longitude, double altitude, double velocity_north, double velocity_east, double velocity_down, double roll, double pitch, double yaw)
{
    position(0) = latitude;
    position(1) = longitude;
    position(2) = altitude;
    velocity(0) = velocity_north;
    velocity(1) = velocity_east;
    velocity(2) = velocity_down;
    orientation(0) = roll;
    orientation(1) = pitch;
    orientation(2) = yaw;
}
void NavigationState::integrate(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, const double& dt) {
    Eigen::Matrix3d R = getRotationMatrix();
}
Eigen::VectorXd NavigationState::getState() const {
    Eigen::VectorXd state(9);
    state << position, velocity, orientation;
    return state;
}
Eigen::Matrix3d NavigationState::getRotationMatrix() const {
    return earth::rpyToRotationMatrix(orientation);
}
std::string NavigationState::toString() const {
    Eigen::VectorXd state = getState();
    std::stringstream stateString;
    stateString << state;
    return stateString.str();
}
void NavigationState::attitudeUpdate(const Eigen::Vector3d& gyro, const double& dt) {}
void NavigationState::velocityUpdate(const Eigen::Vector3d& accel, const double& dt) {}
void NavigationState::positionUpdate(const Eigen::Vector3d& accel, const double& dt) {}