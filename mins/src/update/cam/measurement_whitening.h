#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <stdexcept>

namespace gestalt_mins {

inline Eigen::Matrix2d camera_noise_whitener(const Eigen::Matrix2d &covariance) {
  if (!covariance.allFinite())
    throw std::runtime_error("MINS camera measurement covariance is nonfinite");
  const Eigen::LLT<Eigen::Matrix2d> factor(covariance);
  if (factor.info() != Eigen::Success)
    throw std::runtime_error("MINS camera measurement covariance is not positive definite");
  // R = L L^T, so W = L^-1 gives W R W^T = I. Factoring L again
  // interprets its lower triangle as a symmetric matrix and is not whitening.
  return factor.matrixL().solve(Eigen::Matrix2d::Identity());
}

} // namespace gestalt_mins
