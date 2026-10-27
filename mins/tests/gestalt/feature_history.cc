// Cleanup must work AFTER propagation has already marginalized its clones.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"
#include "feat/FeatureDatabase.h"
#include "feat/Feature.h"
#include "update/cam/CamHelper.h"
#include <iostream>
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
  options->window_size = 2;
  options->dt_exp = .01;
  auto state = std::make_shared<mins::State>(options);
  auto offset = std::make_shared<ov_type::Vec>(1);
  offset->set_value(Eigen::Matrix<double, 1, 1>::Constant(.04));
  state->cam_dt[0] = offset;
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cameras;
  cameras[0] = std::make_shared<ov_core::CamRadtan>(40, 30);
  auto tracker = std::make_shared<ov_core::TrackKLT>(
      cameras, 10, 0, false, ov_core::TrackBase::NONE, 10, 2, 2, 5);
  auto active = tracker->get_feature_database();
  auto archive = std::make_shared<ov_core::FeatureDatabase>();
  auto unused = std::make_shared<ov_core::FeatureDatabase>();
  std::vector<std::shared_ptr<ov_core::Feature>> features;
  std::vector<Eigen::Vector3d> used;
  for (int frame = 0; frame < 6000; ++frame) {
    double time = frame / 60.;
    for (auto &db : {active, archive})
      db->update_feature(1, time, 0, frame, 0, frame, 0);
    if (frame < 120) continue;
    state->clones.clear();
    state->clones[time - 2] = std::make_shared<ov_type::PoseJPL>();
    state->clones[time] = std::make_shared<ov_type::PoseJPL>();
    mins::CamHelper::cleanup_features(state, features, unused, used, tracker, archive);
    for (auto &db : {active, archive}) {
      const auto &times = db->get_feature(1)->timestamps.at(0);
      if (times.size() > 125 || times.back() != time || times.front() < time - 2.051)
        throw std::runtime_error("feature history grows beyond the usable clone window");
    }
  }
  // Keep the exact lower boundary, including the measured camera time offset.
  const double boundary = state->oldest_clone_time() - options->dt_exp - .04;
  active->update_feature(2, boundary, 0, 0, 0, 0, 0);
  active->update_feature(3, boundary - 1e-6, 0, 0, 0, 0, 0);
  mins::CamHelper::cleanup_features(state, features, unused, used, tracker, archive);
  if (!active->get_feature(2) || active->get_feature(3))
    throw std::runtime_error("cleanup removed a usable temporal-boundary observation");
  std::cout << "100 seconds of feature history stay bounded; interpolation boundary retained\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
