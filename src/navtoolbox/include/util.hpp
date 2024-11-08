#ifndef UTIL_HPP
#define UTIL_HPP
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

/*
So in gene
*/

// Namespace for utility functions
namespace util {

// Matrix-matrix product with optional transpose
Eigen::MatrixXd mm_prod(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, bool at = false, bool bt = false);

// Symmetric matrix-matrix product (a * b * a.T)
Eigen::MatrixXd mm_prod_symmetric(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

/**
 * @brief Compute products of multiple matrices and vectors stored in a stack.
 *
 * @param a The matrix or stack of matrices, with dimensions (m, n) or (batch_size, m, n).
 * @param b The vector or stack of vectors, with dimensions (n) or (batch_size, n).
 * @param at Whether to use transpose of `a`.
 * @return Resulting product as an Eigen matrix or vector, with dimensions matching the input stack.
 */
Eigen::VectorXd mv_prod(const Eigen::MatrixXd& a, const Eigen::VectorXd& b, bool at = false);

// Skew matrix creation from a 3D vector
Eigen::Matrix3d skew_matrix(const Eigen::Vector3d& vec);

// Compute RMS of a vector
double compute_rms(const Eigen::VectorXd& data);

// Reduce angle to the range [-180, 180] degrees
double to_180_range(double angle);

template<typename Derived>
Eigen::ArrayXd to_180_range(const Eigen::ArrayBase<Derived>& angles);

}  // namespace util
#endif
