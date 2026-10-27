// Numerical lifetime check: no images/wheels/GPS, ordered stationary IMU.
// This is not a simulated trajectory or a physical accuracy measurement.
#include "core/SystemManager.h"
#include "options/OptionsCamera.h"
#include "options/OptionsEstimator.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "types/IMU.h"
#include <iostream>

extern "C" void gestalt_mins_update(const char*) {}
extern "C" void gestalt_mins_covariance_rejected(const char*) {}
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
extern "C" void gestalt_mins_gate(const char*, int, double, double) {}
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

int main() try {
  auto options = std::make_shared<mins::OptionsEstimator>();
  options->load();
  options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
  options->vicon->enabled = options->wheel->enabled = false;
  options->clone_during_camera_outage = true;
  options->dynamic_cloning = false;
  options->clone_freq = 10;
  options->window_size = 1;
  options->intr_order = 3;
  options->use_imu_res = true;
  options->use_imu_cov = true;
  mins::SystemManager manager(options);
  auto state = manager.state;
  state->initialized = true;
  state->time = 1;
  ov_core::ImuData sample;
  sample.wm.setZero();
  sample.am = options->gravity;
  for (int i = 0; i <= 10000; ++i) {
    sample.timestamp = 1 + .01 * i;
    manager.feed_measurement_imu(sample);
    if (i > 100 &&
        (state->clones.size() > 15 || state->cpis.size() > 200 ||
         manager.prop->imu_data.size() > 250 || state->cov.rows() > 105))
      throw std::runtime_error("camera outage leaves unbounded clone/IMU history");
  }
  if (state->time < 100 || state->newest_clone_time() < 100 ||
      !state->cov.allFinite() || state->imu->pos().norm() > 1e-8)
    throw std::runtime_error("camera outage stopped or corrupted IMU propagation");
  std::cout << "100 s camera outage: continuous finite state, bounded clone/CPI/IMU history\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
