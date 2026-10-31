// Analytic rotating-IMU integration checks against the actual MINS library.
// These are numerical unit tests, not robot simulation or sensor calibration.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "state/Propagator.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "utils/sensor_data.h"
#include "utils/quat_ops.h"
#include <iostream>
#include <stdexcept>

extern "C" void gestalt_mins_update(const char*) {}
extern "C" void gestalt_mins_covariance_rejected(const char*) {}
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
extern "C" void gestalt_mins_gate(const char*, int, double, double) {}
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

static const Eigen::Matrix3d anchor_rotation =
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()).toRotationMatrix();
static const Eigen::Vector3d initial_velocity(0.4, -0.2, 0.1);

struct Fixture {
  std::shared_ptr<mins::State> state;
  std::shared_ptr<mins::Propagator> propagator;
  Fixture() {
    auto options = std::make_shared<mins::OptionsEstimator>();
    options->load();
    options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
    options->vicon->enabled = options->wheel->enabled = false;
    options->gravity.setZero();
    state = std::make_shared<mins::State>(options);
    Eigen::VectorXd value = Eigen::VectorXd::Zero(16);
    value.head<4>() = ov_core::rot_2_quat(anchor_rotation);
    value.segment<3>(7) = initial_velocity;
    state->imu->set_value(value);
    state->imu->set_fej(value);
    state->time = 1;
    state->initialized = true;
    auto clone = std::make_shared<ov_type::PoseJPL>();
    clone->set_value(value.head<7>());
    clone->set_fej(value.head<7>());
    clone->set_local_id(15);
    state->clones[1] = clone;
    mins::State::CPI anchor;
    anchor.t = anchor.clone_t = 1;
    anchor.dt = 0;
    anchor.v = initial_velocity;
    state->cpis[1] = anchor;
    propagator = std::make_shared<mins::Propagator>(state);
    state->hook_propagator(propagator);
    for (int i = -30; i <= 30; ++i) {
      ov_core::ImuData sample;
      sample.timestamp = 1 + i * 0.01;
      sample.wm = Eigen::Vector3d(0, 0, 1);
      sample.am = Eigen::Vector3d(1, 0, 0);
      propagator->imu_data.push_back(sample);
    }
    propagator->reset_cpi(1);
  }
  ~Fixture() { state->hook_propagator(nullptr); }
};

struct InspectablePropagator : mins::Propagator {
  using mins::Propagator::Propagator;
  using mins::Propagator::predict_and_compute;
};

static void check_velocity(const mins::State::CPI &value) {
  const double dt = value.t - 1;
  // Integrate R_world_from_initial * R_z(t) * [1,0,0] exactly.
  const Eigen::Vector3d expected = initial_velocity + anchor_rotation.transpose() *
      Eigen::Vector3d(std::sin(dt), 1 - std::cos(dt), 0);
  if ((value.v - expected).norm() > 1e-8) {
    std::cerr << "dt=" << dt << " expected=" << expected.transpose()
              << " actual=" << value.v.transpose() << '\n';
    throw std::runtime_error("CPI velocity used a moving rather than anchor frame");
  }
}

int main(int argc, char **argv) try {
  if (argc != 2) throw std::runtime_error("select forward, reintegrate or linear");
  const std::string mode(argv[1]);
  if (mode == "forward") {
    Fixture forward;
    forward.propagator->propagate(1.1);
    forward.propagator->propagate(1.2);
    for (const auto &entry : forward.state->cpis) check_velocity(entry.second);
  } else if (mode == "reintegrate") {
    for (double t : {0.8, 1.2}) {
      Fixture reintegrated;
      if (!reintegrated.state->have_cpi(t)) throw std::runtime_error("CPI integration failed");
      check_velocity(reintegrated.state->cpis.at(t));
    }
  } else if (mode == "linear") {
    Fixture forward;
    forward.propagator->propagate(1.1);
    if (!forward.state->have_cpi(1.055)) throw std::runtime_error("CPI interpolation failed");
    if ((forward.state->cpis.at(1.055).w - Eigen::Vector3d(0, 0, 1)).norm() > 1e-12)
      throw std::runtime_error("interpolated CPI lost angular velocity");
  } else if (mode == "uncertainty") {
    Fixture nominal, uncertain;
    InspectablePropagator a(nominal.state), b(uncertain.state);
    ov_core::ImuData left, right;
    left.timestamp = 1; right.timestamp = 1.027;
    left.wm = right.wm = Eigen::Vector3d(0.3, 0.1, 0.2);
    left.am = right.am = Eigen::Vector3d(1, 2, 9.8);
    Eigen::Matrix<double, 15, 15> f1, f2, q1, q2;
    a.predict_and_compute(left, right, f1, q1);
    right.noise_variance_scale = 9;
    b.predict_and_compute(left, right, f2, q2);
    if (!f1.isApprox(f2, 1e-12) || !q2.isApprox(9*q1, 1e-12) ||
        !nominal.state->imu->value().isApprox(uncertain.state->imu->value(), 1e-12))
      throw std::runtime_error("gap noise changed means/Jacobians or failed to scale process covariance");
    const auto cut = mins::Propagator::interpolate_data(left, right, 1.01);
    if (cut.noise_variance_scale != 9 || cut.timestamp != 1.01)
      throw std::runtime_error("interpolation lost acquisition uncertainty");
    right.noise_variance_scale = 0;
    bool rejected = false;
    try { mins::Propagator::interpolate_data(left, right, 1.01); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid uncertainty accepted");
  } else {
    throw std::runtime_error("unknown check");
  }
  std::cout << "CPI velocity check passed: " << mode << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
