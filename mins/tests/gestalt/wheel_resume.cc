// Exercise the real updater after an invalid/coasting interval. Fresh wheel
// data must bracket both clones; a gap must neither be bridged nor block an
// otherwise usable short drive until the old clone ages out.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "update/wheel/UpdaterWheel.h"
#include <iostream>
#include <stdexcept>
#include <string>

static int accepted = 0;
extern "C" void gestalt_mins_update(const char* sensor) {
  if (std::string(sensor) == "WHEEL") ++accepted;
}
extern "C" void gestalt_mins_covariance_rejected(const char*) {}
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
extern "C" void gestalt_mins_gate(const char*, int, double, double) {}
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

int main() try {
  for (double offset : {0., .05}) {
    auto options = std::make_shared<mins::OptionsEstimator>();
    options->load();
    options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
    options->vicon->enabled = false;
    options->wheel->enabled = true;
    options->wheel->do_calib_dt = options->wheel->do_calib_int = options->wheel->do_calib_ext = false;
    options->wheel->dt = offset;
    auto state = std::make_shared<mins::State>(options);
    state->initialized = true;
    const auto clone = [&](double time) {
      state->time = time;
      state->clones[time] = std::dynamic_pointer_cast<ov_type::PoseJPL>(
          mins::StateHelper::clone(state, state->imu->pose()));
      state->cov.setIdentity(); // Independent uncertain poses for this updater check.
    };
    clone(1.0); clone(1.1); clone(1.2);
    mins::UpdaterWheel updater(state);
    updater.reset_measurements();
    updater.try_update(); // Empty after a reset is valid.
    clone(1.3);
    for (double time : {1.225, 1.275, 1.325}) {
      mins::WheelData data; data.time = time - offset; data.m1 = data.m2 = 0.;
      updater.feed_measurement(data);
    }
    const int before = accepted;
    updater.try_update();
    if (accepted != before) throw std::runtime_error("wheel update bridged missing prefix");
    clone(1.4);
    for (double time : {1.375, 1.425, 1.475}) {
      mins::WheelData data; data.time = time - offset; data.m1 = data.m2 = 0.;
      updater.feed_measurement(data);
    }
    updater.try_update();
    if (accepted != before + 1)
      throw std::runtime_error("short valid wheel segment blocked by pre-gap clone");
    updater.try_update();
    if (accepted != before + 1) throw std::runtime_error("wheel information reused");
    updater.reset_measurements();
    updater.try_update();
    if (accepted != before + 1) throw std::runtime_error("reset retained wheel information");
  }
  std::cout << "Wheel fusion resumes on the first bracketed pair, without bridging gaps\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
