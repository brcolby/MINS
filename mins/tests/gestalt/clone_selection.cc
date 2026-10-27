// Call the actual vendor helpers: the result must depend on the requested
// epoch, never on the previous/uninitialized contents of their output argument.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

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
  mins::State state(options);
  double output = 123;
  if (state.closest_clone_time(2.1, output) || state.closest_clone_time_not_imu(2.1, output))
    throw std::runtime_error("empty clone map reported a match");
  for (int i = 1; i <= 3; ++i) {
    auto clone = std::make_shared<ov_type::PoseJPL>();
    clone->set_local_id(15 + 6 * i);
    state.clones[i] = clone;
  }
  state.clones[4] = state.imu->pose();
  for (bool exclude_imu : {false, true}) {
    for (double requested : {0.1, 1., 1.5, 2.1, 2.9, 3.8, 8.}) {
      for (double seed : {0., 1., 1000., std::numeric_limits<double>::quiet_NaN()}) {
        output = seed;
        const bool found = exclude_imu ? state.closest_clone_time_not_imu(requested, output)
                                       : state.closest_clone_time(requested, output);
        double expected = 1;
        for (double candidate : {2., 3., 4.})
          if (!(exclude_imu && candidate == 4) &&
              std::abs(candidate - requested) < std::abs(expected - requested))
            expected = candidate;
        if (!found || output != expected) {
          std::cerr << "requested=" << requested << " output seed=" << seed
                    << " expected=" << expected << " actual=" << output << '\n';
          throw std::runtime_error("nearest clone selection depends on its output argument");
        }
      }
    }
  }
  state.clones.clear();
  state.clones[4] = state.imu->pose();
  if (state.closest_clone_time_not_imu(4, output))
    throw std::runtime_error("current IMU pose accepted as an independent clone");
  std::cout << "Nearest clone depends only on the requested epoch, including IMU exclusion\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
