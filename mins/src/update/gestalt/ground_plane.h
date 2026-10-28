// A declared operating assumption, not a measurement of wheel speed/contact.
// Constrain only changes of the chassis normal and displacement normal to it.
// Horizontal translation (including skid-steer sideslip) and yaw remain free.
#pragma once
#include "options/OptionsEstimator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "utils/quat_ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>

struct GroundPlaneConfig {
  bool enabled = false;
  Eigen::Matrix3d imu_to_wheel = Eigen::Matrix3d::Identity();
  Eigen::Vector3d wheel_in_imu = Eigen::Vector3d::Zero();
  // Continuous uncertainty of tilt [rad/sqrt(Hz)] and height [m/sqrt(Hz)].
  // These are soft operating-model weights, not sensor calibration.
  double rotation_density = 0;
  double translation_density = 0;
};

class GroundPlaneUpdate {
public:
  explicit GroundPlaneUpdate(GroundPlaneConfig value = {}) : config(value) {}
  GroundPlaneConfig config;
  unsigned long long accepted = 0, conflicts = 0;
  double last_update = -1, last_chi2 = -1;
  const char *reason = "disabled";

  // h = 0: two components of the previous wheel-frame normal expressed in the
  // current wheel frame, and wheel-origin displacement along the old normal.
  // Using a normal rather than Euler angles remains well behaved through yaw
  // turns, including 180 degrees. Account for the rigid IMU/wheel lever arm.
  static Eigen::Vector3d observation(
      const Eigen::Matrix3d &R0, const Eigen::Vector3d &p0,
      const Eigen::Matrix3d &R1, const Eigen::Vector3d &p1,
      const GroundPlaneConfig &c) {
    const Eigen::Vector3d normal = c.imu_to_wheel * R1 * R0.transpose() *
        c.imu_to_wheel.transpose() * Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d displacement = c.imu_to_wheel * R0 *
        (p1 + R1.transpose() * c.wheel_in_imu -
         p0 - R0.transpose() * c.wheel_in_imu);
    return {normal.x(), normal.y(), displacement.z()};
  }

  // d h / d [theta0, p0, theta1, p1], for JPL left orientation errors.
  static Eigen::Matrix<double, 3, 12> jacobian(
      const Eigen::Matrix3d &R0, const Eigen::Vector3d &p0,
      const Eigen::Matrix3d &R1, const Eigen::Vector3d &p1,
      const GroundPlaneConfig &c) {
    const Eigen::Matrix3d &B = c.imu_to_wheel;
    const Eigen::Vector3d n = B.transpose() * Eigen::Vector3d::UnitZ();
    Eigen::Matrix<double, 3, 12> H = Eigen::Matrix<double, 3, 12>::Zero();
    H.block<2, 3>(0, 0) = (-B * R1 * R0.transpose() * ov_core::skew_x(n)).topRows<2>();
    H.block<2, 3>(0, 6) = (B * ov_core::skew_x(R1 * R0.transpose() * n)).topRows<2>();
    H.block<1, 3>(2, 0) = (B * ov_core::skew_x(
        R0 * (p1 + R1.transpose() * c.wheel_in_imu - p0))).bottomRows<1>();
    H.block<1, 3>(2, 3) = (-B * R0).bottomRows<1>();
    H.block<1, 3>(2, 6) = (-B * R0 * R1.transpose() *
        ov_core::skew_x(c.wheel_in_imu)).bottomRows<1>();
    H.block<1, 3>(2, 9) = (B * R0).bottomRows<1>();
    return H;
  }

  void update(const std::shared_ptr<mins::State> &state) {
    if (!config.enabled) return;
    if (!accepted && !conflicts) reason = "waiting_for_clones";
    if (!state->initialized) return;
#ifndef GESTALT_MINS_OPTIMIZED
    throw std::runtime_error("ground-plane prior requires the optimized MINS fork");
#else
    if (!reference && std::count_if(state->clones.begin(), state->clones.end(),
        [&](const auto &item) { return item.second->id() != state->imu->pose()->id(); }) < 2)
      return;
    // Keep one stochastic reference pose in the existing retained-clone set.
    // This preserves its covariance/cross-correlation and GPS frame transforms.
    // A moving pairwise reference could accept a different plane after every
    // rejected interval; the operator instead declared ONE plane for the run.
    // One extra clone is bounded, and reset destroys it with the native state.
    if (reference && !state->clones.count(reference_time))
      throw std::runtime_error("ground-plane reference clone disappeared");
    for (const auto &[time, pose] : state->clones) {
      if (pose->id() == state->imu->pose()->id()) continue;
      if (!reference) {
        reference = pose;
        reference_time = last_pair_time = time;
        state->retained_clones.insert(time);
      }
      if (time > last_pair_time) {
        // Initialization may retain a reference from the beginning of its
        // quiet window before normal cloning starts. A delayed clone is one
        // observation, not evidence for every unobserved epoch in that gap.
        const double dt = std::min(time - last_pair_time, 1.0 / state->op->clone_freq);
        if (!(dt > 0 && std::isfinite(dt)))
          throw std::runtime_error("invalid ground-plane clone interval");
        last_pair_time = time;
        constrain(state, reference, pose, time, dt);
      }
    }
#endif
  }

private:
  double last_pair_time = -1;
  double reference_time = -1;
  std::shared_ptr<ov_type::PoseJPL> reference;
  void constrain(const std::shared_ptr<mins::State> &state,
                 const std::shared_ptr<ov_type::PoseJPL> &p0,
                 const std::shared_ptr<ov_type::PoseJPL> &p1,
                 double time, double dt) {
    const Eigen::Vector3d residual = -observation(p0->Rot(), p0->pos(), p1->Rot(), p1->pos(), config);
    const auto H = jacobian(p0->Rot_fej(), p0->pos_fej(), p1->Rot_fej(), p1->pos_fej(), config);
    Eigen::Matrix3d noise = Eigen::Matrix3d::Zero();
    // This observes pose relative to the SAME plane, rather than integrating
    // a zero velocity. Divide by dt so information per second stays constant
    // across clone rates; do not make uncertainty grow with distance/time.
    noise.diagonal() << std::pow(config.rotation_density, 2) / dt,
                        std::pow(config.rotation_density, 2) / dt,
                        std::pow(config.translation_density, 2) / dt;
    std::vector<std::shared_ptr<ov_type::Type>> order{p0, p1};
    const auto P = mins::StateHelper::get_marginal_covariance(state, order);
    const Eigen::Matrix3d S = H * P * H.transpose() + noise;
    const Eigen::LDLT<Eigen::Matrix3d> factor(S);
    if (!S.allFinite() || !residual.allFinite() || factor.info() != Eigen::Success ||
        (factor.vectorD().array() <= 0).any())
      throw std::runtime_error("invalid ground-plane innovation covariance");
    last_chi2 = residual.dot(factor.solve(residual));
    const double normal_dot = (config.imu_to_wheel * p1->Rot() *
        p0->Rot().transpose() * config.imu_to_wheel.transpose())(2, 2);
    if (!std::isfinite(last_chi2))
      throw std::runtime_error("nonfinite ground-plane innovation");
    if (normal_dot <= 0)
      throw std::runtime_error("estimated ground normal flipped; estimator reset required");
    // The operator declares continuous ground contact, rather than supplying a
    // contact sensor which might be an outlier. Disagreement is diagnostic;
    // dropping the constraint when inertial prediction drifts would silently
    // abandon that declared model. Keep finite model uncertainty and the full
    // EKF correction/covariance, never clamp a published pose to the floor.
    const bool conflict = last_chi2 > 11.344866730144373; // 99%, three dimensions
    if (conflict) ++conflicts;
    if (!mins::StateHelper::EKFUpdate(state, order, H, residual, noise, "GROUND")) {
      const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(state->cov);
      std::ostringstream detail;
      detail << std::setprecision(17) << "ground-plane covariance update rejected: min_diagonal="
             << state->cov.diagonal().minCoeff() << ", min_eigenvalue="
             << spectrum.eigenvalues().minCoeff() << ", dimensions=" << state->cov.rows();
      throw std::runtime_error(detail.str());
    }
    ++accepted;
    last_update = time;
    reason = conflict ? "ground_plane_conflict" : "ground_plane";
  }
};
