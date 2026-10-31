#include "cpi/CpiV1.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void near(double actual, double expected) {
  if (std::abs(actual - expected) > 1e-11 + 1e-6 * std::abs(expected)) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    throw std::runtime_error("preintegration disagrees with analytic covariance");
  }
}

ov_core::CpiV1 integrate(double direction, const Eigen::Vector3d &w,
                        const Eigen::Vector3d &a, double variance = 1.0) {
  ov_core::CpiV1 cpi(0.02, 0.003, 0.1, 0.005, true);
  cpi.set_noise_variance_scale(variance);
  cpi.setLinearizationPoints(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  for (int index = 0; index < 40; ++index) {
    const double t = 1.0 + direction * index * 0.005;
    cpi.feed_IMU(t, t + direction * 0.005, w, a, w, a);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> solver(cpi.P_meas);
    if (!cpi.P_meas.allFinite() || !cpi.P_meas.isApprox(cpi.P_meas.transpose(), 1e-12)
        || solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() < -1e-14)
      throw std::runtime_error("preintegration covariance is not positive semidefinite");
  }
  near(cpi.DT, direction * 0.2);
  return cpi;
}
}

int main() {
  const auto zero = Eigen::Vector3d::Zero().eval();
  for (double direction : {1.0, -1.0}) {
    const auto cpi = integrate(direction, zero, zero);
    const double t = std::abs(cpi.DT);
    const double qw = 0.02 * 0.02, qwb = 0.003 * 0.003;
    const double qa = 0.1 * 0.1, qab = 0.005 * 0.005;
    // Independent integrated-white-noise/bias-random-walk solution.
    // Time reversal changes odd cross-covariances, never variance signs.
    for (int axis = 0; axis < 3; ++axis) {
      const auto &p = cpi.P_meas;
      near(p(axis, axis), qw * t + qwb * std::pow(t, 3) / 3);
      near(p(axis + 3, axis + 3), qwb * t);
      near(p(axis, axis + 3), -direction * qwb * t * t / 2);
      near(p(axis + 6, axis + 6), qa * t + qab * std::pow(t, 3) / 3);
      near(p(axis + 9, axis + 9), qab * t);
      near(p(axis + 6, axis + 9), -direction * qab * t * t / 2);
      near(p(axis + 12, axis + 12), qa * std::pow(t, 3) / 3 + qab * std::pow(t, 5) / 20);
      near(p(axis + 12, axis + 6), direction * (qa * t * t / 2 + qab * std::pow(t, 4) / 8));
      near(p(axis + 12, axis + 9), -qab * std::pow(t, 3) / 6);
    }
    // A signed integration interval must still propagate the signed means.
    const Eigen::Vector3d acceleration(0.2, -0.4, 9.8);
    const auto translated = integrate(direction, zero, acceleration);
    for (int axis = 0; axis < 3; ++axis) {
      near(translated.beta_tau(axis), acceleration(axis) * translated.DT);
      near(translated.alpha_tau(axis), 0.5 * acceleration(axis) * translated.DT * translated.DT);
    }
    const auto nominal = integrate(direction, Eigen::Vector3d(0.3, -0.1, 0.4), acceleration);
    const auto uncertain = integrate(direction, Eigen::Vector3d(0.3, -0.1, 0.4), acceleration, 9.0);
    if (!uncertain.P_meas.isApprox(9 * nominal.P_meas, 1e-12) ||
        !uncertain.alpha_tau.isApprox(nominal.alpha_tau, 1e-12) ||
        !uncertain.beta_tau.isApprox(nominal.beta_tau, 1e-12) ||
        !uncertain.R_k2tau.isApprox(nominal.R_k2tau, 1e-12))
      throw std::runtime_error("IMU uncertainty did not scale covariance independently of means");
  }
  std::cout << "Forward/reverse covariance, cross terms and signed means passed\n";
}
