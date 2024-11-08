#include "util.hpp"

// Matrix-matrix product with optional transpose
Eigen::MatrixXd mm_prod(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, bool at = false, bool bt = false) {
    Eigen::MatrixXd mat_a = at ? a.transpose() : a;
    Eigen::MatrixXd mat_b = bt ? b.transpose() : b;
    return mat_a * mat_b;
}

// Symmetric matrix-matrix product (a * b * a.T)
Eigen::MatrixXd mm_prod_symmetric(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return mm_prod(mm_prod(a, b), a, false, true);
}

// Matrix-vector product with optional transpose
Eigen::VectorXd mv_prod(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, bool at = false) {
    Eigen::MatrixXd mat_a = at ? a.transpose() : a;
    return mat_a * b;
}

// Skew matrix creation from a 3D vector
Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& vec) {
    Eigen::Matrix3d skew = Eigen::Matrix3d::Zero();
    skew(0, 1) = -vec(2);
    skew(0, 2) = vec(1);
    skew(1, 0) = vec(2);
    skew(1, 2) = -vec(0);
    skew(2, 0) = -vec(1);
    skew(2, 1) = vec(0);
    return skew;
}

// Compute RMS of a vector
double compute_rms(const Eigen::VectorXd& data) {
    return std::sqrt((data.array().square().mean()));
}

// Reduce angle to the range [-180, 180] degrees
double to_180_range(double angle) {
    angle = std::fmod(angle, 360.0);
    if (angle < -180.0) angle += 360.0;
    else if (angle > 180.0) angle -= 360.0;
    return angle;
}

template<typename Derived>
Eigen::ArrayXd to_180_range(const Eigen::ArrayBase<Derived>& angles) {
    Eigen::ArrayXd result = angles;
    result = result.unaryExpr([](double angle) { return to_180_range(angle); });
    return result;
}

