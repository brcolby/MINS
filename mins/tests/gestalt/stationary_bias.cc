// Exercise the real EKF update, including the tilt/bias ambiguity at rest.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "update/gestalt/stationary_bias.h"
#include <iostream>
#include <stdexcept>

extern "C" void gestalt_mins_update(const char*) {}
extern "C" void gestalt_mins_covariance_rejected(const char*) {}
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
extern "C" void gestalt_mins_gate(const char*, int, double, double) {}
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

static void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}

int main() try {
  auto options = std::make_shared<mins::OptionsEstimator>();
  options->load();
  options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
  options->vicon->enabled = options->wheel->enabled = false;
  auto state = std::make_shared<mins::State>(options);
  state->cov.setIdentity();
  state->cov *= 0.01;
  // A tilted robot: accelerometer mean is not a zero-acceleration measurement.
  Eigen::VectorXd value = state->imu->value();
  value.head<4>() << std::sin(.2), 0, 0, std::cos(.2);
  state->imu->set_value(value);
  state->imu->set_fej(value);
  const Eigen::Vector3d gravity = state->imu->Rot() * options->gravity;
  const Eigen::Vector3d gyro(.01, -.02, .005), accel = gravity + Eigen::Vector3d(.02, -.01, .03);
  std::deque<ov_core::ImuData> samples;
  for (int i = 0; i <= 50; ++i) {
    ov_core::ImuData sample;
    sample.timestamp = 1 + i * .01;
    sample.wm = gyro;
    sample.am = accel;
    samples.push_back(sample);
  }
  // Check the JPL gravity derivative with actual orientation perturbations.
  Eigen::Matrix3d numeric;
  const auto original_q = state->imu->q()->value();
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d delta = Eigen::Vector3d::Zero();
    delta(axis) = 1e-7;
    state->imu->q()->update(delta);
    numeric.col(axis) = (state->imu->Rot() * options->gravity - gravity) / 1e-7;
    state->imu->q()->set_value(original_q);
  }
  require(numeric.isApprox(ov_core::skew_x(gravity), 1e-6), "gravity Jacobian sign/frame changed");
  Eigen::VectorXd ambiguity = Eigen::VectorXd::Zero(state->cov.rows());
  ambiguity(state->imu->q()->id()) = 1;
  ambiguity.segment<3>(state->imu->ba()->id()) = -numeric.col(0);
  ambiguity.normalize();
  const double before_uncertainty = ambiguity.dot(state->cov * ambiguity);
  StationaryBiasUpdate update;
  update.update(samples, 1.5, .5, state);
  require(update.accepted == 1, "stationary observation rejected");
  require((state->imu->bias_g() - gyro).norm() < 1e-4, "gyro bias moved the wrong way");
  require((state->imu->bias_a() + state->imu->Rot() * options->gravity - accel).norm() < .002,
          "gravity-aware residual did not improve");
  require(std::abs(ambiguity.dot(state->cov * ambiguity) - before_uncertainty) < 1e-9,
          "single pose falsely observed the tilt/accelerometer-bias ambiguity");
  const auto covariance = state->cov;
  const auto estimate = state->imu->value();
  update.update(samples, 1.5, .5, state);
  require(update.accepted == 1 && covariance.isApprox(state->cov, 0), "reused overlapping observations");
  auto boundary = samples;
  for (auto &sample : boundary) sample.timestamp += .5;
  update.update(boundary, 2, .5, state);
  require(update.accepted == 1, "shared boundary sample counted twice");
  for (auto &sample : samples) { sample.timestamp += 1; sample.wm += Eigen::Vector3d::Ones() * 10; }
  update.update(samples, 2.5, .5, state);
  require(update.accepted == 1 && std::string(update.reason) == "innovation_gate" &&
          estimate.isApprox(state->imu->value(), 0), "outlier changed bias state");
  std::cout << "Stationary bias signs, gravity, observability, overlap and outlier checks passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
