#include "measurement_whitening.h"
#include <cmath>
#include <limits>
#include <iostream>

int main() {
  for (double scale : {0.01, 1.0, 100.0}) {
    for (double correlation : {-0.9, -0.4, 0.0, 0.7, 0.99}) {
      Eigen::Matrix2d R;
      R << 4, correlation * 6, correlation * 6, 9;
      R *= scale;
      const auto W = gestalt_mins::camera_noise_whitener(R);
      const Eigen::Matrix2d whitened = W * R * W.transpose();
      if (!whitened.isApprox(Eigen::Matrix2d::Identity(), 1e-11)) {
        std::cerr << "Whitened covariance is not identity:\n" << whitened << '\n';
        return 1;
      }
      // Whitened residual energy must equal the original Mahalanobis distance.
      const Eigen::Vector2d residual(0.7, -1.2);
      const double expected = residual.dot(R.ldlt().solve(residual));
      if (std::abs((W * residual).squaredNorm() - expected) > 1e-10 * expected)
        return 2;
    }
  }
  for (double bad : {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN()}) {
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    R(0, 0) = bad;
    try {
      gestalt_mins::camera_noise_whitener(R);
      return 3;
    } catch (const std::runtime_error &) {
    }
  }
  std::cout << "Correlated covariance, residual energy and invalid-input checks passed\n";
}
