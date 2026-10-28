// A stationary IMU measures gyro bias and gravity plus accelerometer bias.
// The caller must establish physical rest; low IMU variance alone cannot do so.
#pragma once
#include "options/OptionsIMU.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "utils/sensor_data.h"
#include "utils/quat_ops.h"
#include <deque>
#include <cmath>
#include <vector>

class StationaryBiasUpdate {
public:
  unsigned long long accepted = 0;
  const char *reason = "waiting_for_rest";
  double last_sample = -1;

  void update(const std::deque<ov_core::ImuData> &samples, double camera_time,
              double window, const std::shared_ptr<mins::State> &state) {
    // Do not count the same filtered samples as independent bias measurements
    // on every camera frame. Retain the ordinary velocity constraint separately.
    if (samples.empty() || samples.front().timestamp <= last_sample)
      return;
    std::vector<const ov_core::ImuData *> block;
    for (const auto &sample : samples)
      if (sample.timestamp <= camera_time)
        block.push_back(&sample);
    if (block.size() < 10 ||
        block.back()->timestamp - block.front()->timestamp < window - 0.05)
      return;
    const double span = block.back()->timestamp - block.front()->timestamp;
    last_sample = block.back()->timestamp;
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero(), accel = gyro;
    for (const auto *sample : block) {
      gyro += sample->wm;
      accel += sample->am;
    }
    gyro /= block.size();
    accel /= block.size();
    Eigen::Vector3d gyro_variance = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_variance = gyro_variance;
    for (const auto *sample : block) {
      gyro_variance += (sample->wm - gyro).array().square().matrix();
      accel_variance += (sample->am - accel).array().square().matrix();
    }
    gyro_variance /= block.size() - 1;
    accel_variance /= block.size() - 1;
    // Low-pass filtering correlates consecutive readings. Conservatively keep
    // their sample variance (not variance/N), plus the configured continuous
    // noise floor for a block mean. This is an observation weight, not a new
    // calibration fitted from a stationary pose.
    Eigen::Matrix<double, 6, 6> noise = Eigen::Matrix<double, 6, 6>::Zero();
    noise.diagonal().head<3>() = gyro_variance.array() +
        std::pow(state->op->imu->sigma_w, 2) / span;
    noise.diagonal().tail<3>() = accel_variance.array() +
        std::pow(state->op->imu->sigma_a, 2) / span;
    Eigen::Matrix<double, 6, 1> residual;
    residual.head<3>() = state->imu->bias_g() - gyro;
    residual.tail<3>() = state->imu->bias_a() +
        state->imu->Rot() * state->op->gravity - accel;
    // Keep orientation and accelerometer bias jointly uncertain: one resting
    // pose cannot identify both independently. Signs follow MINS/OpenVINS's
    // JPL left-error convention; regression checks use numerical perturbations.
    Eigen::Matrix<double, 6, 9> H = Eigen::Matrix<double, 6, 9>::Zero();
    H.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    H.block<3, 3>(3, 0) = -ov_core::skew_x(
        state->imu->Rot_fej() * state->op->gravity);
    H.block<3, 3>(3, 6) = -Eigen::Matrix3d::Identity();
    std::vector<std::shared_ptr<ov_type::Type>> order{
        state->imu->q(), state->imu->bg(), state->imu->ba()};
    const Eigen::MatrixXd P = mins::StateHelper::get_marginal_covariance(state, order);
    const Eigen::Matrix<double, 6, 6> innovation = H * P * H.transpose() + noise;
    const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> factor(innovation);
    if (!innovation.allFinite() || !residual.allFinite() ||
        factor.info() != Eigen::Success || (factor.vectorD().array() <= 0).any()) {
      reason = "invalid_covariance";
      return;
    }
    const double chi2 = residual.dot(factor.solve(residual));
    if (!std::isfinite(chi2) || chi2 > 16.811893829770927) {
      reason = "innovation_gate"; // 99%, six stationary inertial residuals.
      return;
    }
    if (mins::StateHelper::EKFUpdate(state, order, H, residual, noise, "REST")) {
      ++accepted;
      reason = "stationary_bias";
    } else {
      reason = "update_rejected";
    }
  }
};
