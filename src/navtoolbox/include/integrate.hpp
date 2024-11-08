#ifndef STRAPDOWN_INTEGRATOR_H
#define STRAPDOWN_INTEGRATOR_H

#include <Eigen/Dense>

/**
 * @brief A class to perform strapdown inertial navigation integration in the NED frame.
 */
class StrapdownIntegrator {
public:
    // Navigation state variables
    double lat;      // Latitude (degrees)
    double lon;      // Longitude (degrees)
    double alt;      // Altitude (meters)
    double vn;       // North velocity (m/s)
    double ve;       // East velocity (m/s)
    double vd;       // Down velocity (m/s)
    double roll;     // Roll angle (degrees)
    double pitch;    // Pitch angle (degrees)
    double yaw;      // Yaw angle (degrees)

    /**
     * @brief Constructor to initialize the strapdown integrator with initial states.
     */
    StrapdownIntegrator(double init_lat, double init_lon, double init_alt,
                        double init_vn, double init_ve, double init_vd,
                        double init_roll, double init_pitch, double init_yaw);

    /**
     * @brief Perform a single integration step of the strapdown INS algorithm.
     * 
     * @param dt Time step (seconds).
     * @param gyro Gyroscope measurements (rad/s) in the body frame.
     * @param accel Accelerometer measurements (m/s^2) in the body frame.
     */
    void integrate(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel);
};

#endif // STRAPDOWN_INTEGRATOR_H
