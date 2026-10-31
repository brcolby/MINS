// Native stationary velocity constraint. Detection is deliberately conjunctive:
// quiet bias-corrected IMU, low predicted speed, and persistent visual tracks.
// No commanded motor state is treated as a measurement of physical rest.
#pragma once
#include "feat/FeatureHelper.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "update/cam/UpdaterCamera.h"
#include "utils/sensor_data.h"
#include "stationary_bias.h"
#include <deque>

struct ZeroVelocityConfig {
  bool enabled;
  double window, max_velocity, max_disparity;
  int min_features;
  double max_gyro, max_acceleration, velocity_std;
  bool bias_update = false;
};

class ZeroVelocityUpdate {
public:
  explicit ZeroVelocityUpdate(ZeroVelocityConfig config) : config(config) {}
  bool active = false;
  const char *reason = "warming_up";
  int features = 0;
  double disparity = -1;
  double mean_disparity = -1, disparity_spread = -1;
  double gyro_residual = -1, acceleration_residual = -1;
  double imu_evidence_time = -1, visual_evidence_time = -1;
  double last_update = -1;
  unsigned long long accepted = 0;
  StationaryBiasUpdate bias;

  void imu(const ov_core::ImuData &sample,
           const std::shared_ptr<mins::State> &state) {
    if (!config.enabled) {
      reason = "disabled";
      return;
    }
    if (sample.noise_variance_scale != 1.0) {
      samples.clear();
      cameras.clear();
      return reject("imu_gap");
    }
    if (!samples.empty() &&
        sample.timestamp - samples.back().timestamp > 0.05) {
      samples.clear();
      cameras.clear();
      active = false;
    }
    samples.push_back(sample);
    while (samples.size() > 1 &&
           samples[1].timestamp < sample.timestamp - config.window)
      samples.pop_front();
    if (samples.size() > 4096)
      throw std::runtime_error("ZUPT IMU window overflow");
    if (!state->initialized)
      reject("initializing");
    else if (!quiet(sample, state))
      reject("imu_motion");
    else if (cameras.empty() || sample.timestamp - cameras.back() > 0.15)
      reject("camera_stale");
    else if (state->imu->vel().norm() > config.max_velocity)
      reject("velocity");
  }

  void wheel(double time, double left, double right) {
    if (!config.enabled)
      return;
    if (std::abs(left) > 1e-9 || std::abs(right) > 1e-9) {
      wheel_motion_until = time + config.window;
      reject("wheel_motion");
    }
  }

  void camera(double time, const std::shared_ptr<mins::State> &state,
              const std::shared_ptr<mins::UpdaterCamera> &camera) {
    active = false;
    if (!config.enabled)
      return reject("disabled");
    if (!cameras.empty() && time - cameras.back() > 0.15)
      cameras.clear();
    cameras.push_back(time);
    while (cameras.size() > 1 && cameras[1] <= time - config.window)
      cameras.pop_front();
    if (cameras.size() > 256)
      throw std::runtime_error("ZUPT camera window overflow");
    // A delayed static frame must not constrain the current moving state.
    if (!state->initialized)
      return reject("initializing");
    if (samples.size() < 10 ||
        samples.back().timestamp - samples.front().timestamp <
            config.window - 1e-6)
      return reject("imu_window");
    if (cameras.size() < 3 || time - cameras.front() < config.window - 1e-6)
      return reject("camera_window");
    if (state->time < time || state->time - time > 0.05)
      return reject("camera_late");
    if (state->imu->vel().norm() > config.max_velocity)
      return reject("velocity");
    if (time <= wheel_motion_until)
      return reject("wheel_motion");
    for (const auto &sample : samples)
      if (!quiet(sample, state))
        return reject("imu_motion");
#ifdef GESTALT_MINS_OPTIMIZED
    // SLAM/MSCKF consume the live measurement database before this check.
    // Reuse the bounded observation archive so those tracked features still
    // contribute to stationary evidence; require both exact frame timestamps.
    auto database = camera->get_observation_history(0);
#else
    auto database = camera->get_tracking_database(0);
#endif
    // Check the complete stationary span AND the latest frame pair. This
    // rejects a fresh motion onset even after a long stationary history.
    for (double from : {cameras.front(), cameras[cameras.size() - 2]}) {
      double mean = -1, spread = -1;
      features = 0;
      ov_core::FeatureHelper::compute_disparity(database, from, time, mean,
                                                spread, features);
      visual_evidence_time = time;
      mean_disparity = std::isfinite(mean) ? mean : -1;
      disparity_spread = std::isfinite(spread) ? spread : -1;
      disparity = std::isfinite(mean + spread) ? mean + spread : -1;
      if (features < config.min_features || disparity < 0)
        return reject("insufficient_features");
      if (disparity > config.max_disparity)
        return reject("visual_motion");
    }
    if (config.bias_update)
      bias.update(samples, time, config.window, state);
    const Eigen::Matrix3d R =
        Eigen::Matrix3d::Identity() * config.velocity_std * config.velocity_std;
    const Eigen::Vector3d residual = -state->imu->vel();
    const Eigen::Matrix3d S =
        mins::StateHelper::get_marginal_covariance(state, {state->imu->v()}) +
        R;
    // 99% chi-square innovation gate, three velocity components.
    double chi2 = residual.dot(S.llt().solve(residual));
    if (!std::isfinite(chi2) || chi2 > 11.344866730144373)
      return reject("innovation_gate");
    // This updates the actual filter state AND covariance (including correlated
    // biases/poses), rather than clamping the reported output velocity.
    if (mins::StateHelper::EKFUpdate(state, {state->imu->v()},
                                     Eigen::Matrix3d::Identity(), residual, R,
                                     "ZUPT")) {
      active = true;
      reason = "stationary";
      last_update = state->time;
      ++accepted;
    } else
      reject("update_rejected");
  }

private:
  void reject(const char *why) {
    active = false;
    reason = why;
  }
  bool quiet(const ov_core::ImuData &sample,
             const std::shared_ptr<mins::State> &state) {
    imu_evidence_time = sample.timestamp;
    gyro_residual = (sample.wm - state->imu->bias_g()).norm();
    acceleration_residual = (sample.am - state->imu->bias_a() -
                            state->imu->Rot() * state->op->gravity).norm();
    return gyro_residual <= config.max_gyro &&
           acceleration_residual <= config.max_acceleration;
  }
  ZeroVelocityConfig config;
  std::deque<ov_core::ImuData> samples;
  std::deque<double> cameras;
  double wheel_motion_until = -1;
};
