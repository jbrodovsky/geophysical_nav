#ifndef PF_HPP
#define PF_HPP

#include <vector>
#include <Eigen/Core>

#include "integrate.hpp"

/**
 * @brief Basic particle for the particle filter. Contains both the navigation states and the weight.
 */
class Particle : public StrapdownIntegrator {
    private:
        double weight;  // Particle weight

    public:
        /**
         * @brief Constructor to initialize the particle with initial states and weight.
         */
        Particle(double init_lat, double init_lon, double init_alt,
                 double init_vn, double init_ve, double init_vd,
                 double init_roll, double init_pitch, double init_yaw,
                 double init_weight)
            : StrapdownIntegrator(init_lat, init_lon, init_alt, init_vn, init_ve, init_vd, init_roll, init_pitch, init_yaw),
              weight(init_weight) {}

        /**
         * @brief Constructor to initialize the particle from a navigation state
         */
        Particle(const StrapdownIntegrator& init_nav, double init_weight)
            : StrapdownIntegrator(init_nav), weight(init_weight) {}

        /**
         * @brief Get the weight of the particle.
         */
        double get_weight() const {
            return weight;
        }

        /**
         * @brief Set the weight of the particle.
         */
        void set_weight(double new_weight) {
            assert(new_weight >= 0);
            weight = new_weight;
        }
};

/**
 * @brief A class to perform particle filtering for navigation state estimation.
 */
class ParticleFilter {
    private:
        std::vector<Particle> particles;  // Vector of particles
        int num_particles;                // Number of particles
        Eigen::VectorXd process_noise;             // Process noise variance
        Eigen::VectorXd measurement_noise;         // Measurement noise variance

    public:
        // The particle filter should have a set of constructors to build the object
        // that set things like the number of partilces, the process noise, and the 
        // measurement noise. The noise parameters can be given as a double, as a 
        // vector of doubles, or as an Eigen::VectorXd of doubles. Will need to add
        // some logic in to check the size.

        /**
         * @brief Constructor to initialize the particle filter with a number of particles and noise variances. 
         */
        ParticleFilter(double process_noise, double measurement_noise);

        /**
         * @brief Constructor to initialize the particle filter with a number of particles and noise variances. 
         */
        ParticleFilter(const Eigen::VectorXd& process_noise, const Eigen::VectorXd& measurement_noise);

        /**
         * @brief Constructor to initialize the particle filter with a number of particles and noise variances from std::vector<double>
         * 
         */
        ParticleFilter(const std::vector<double>& process_noise, const std::vector<double>& measurement_noise);
        

        /**
         * @brief Initialize the particle filter with a specified set of particles.
         */
        void initialize(const std::vector<StrapdownIntegrator>& init_navs);

        /**
         * @brief Initialize the particle filter about a single navigation state 
         * using a normal distribution with the specified standard deviations.
         * 
         */
        void initialize_about(int& n, const StrapdownIntegrator& init_nav, const Eigen::VectorXd& std_devs);

        /**
         * @brief Perform a single integration step of the particle filter.
         * 
         * @param dt Time step (seconds).
         * @param gyro Gyroscope measurements (rad/s) in the body frame.
         * @param accel Accelerometer measurements (m/s^2) in the body frame.
         */
        void integrate(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel);

        /**
         * @brief Perform the particle filter prediction step.
         */
        void predict(double dt, const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel);

        /**
         * @brief Perform the particle filter update step.
         */
        void update(const Eigen::Vector3d& meas);

        /**
         * @brief Perform the particle filter resampling step.
         */
        void resample();

        /**
         * @brief Get the best estimate of the navigation state.
         */
        StrapdownIntegrator get_best_estimate() const;

        /**
         * @brief Get the vector of particles.
         */
        std::vector<Particle> get_particles() const {
            return particles;
        }
};

#endif // PF_HPP