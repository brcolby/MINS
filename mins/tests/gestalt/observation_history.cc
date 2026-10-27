// The stationary detector must still see observations consumed by SLAM/MSCKF.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureHelper.h"
#include "update/cam/UpdaterCamera.h"
#include "utils/sensor_data.h"
#include <opencv2/imgproc.hpp>
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
  cv::setNumThreads(1);
  auto options = std::make_shared<mins::OptionsEstimator>();
  options->load();
  options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
  options->vicon->enabled = options->wheel->enabled = false;
  options->cam->max_n = 1;
  options->cam->n_pts = 80;
  options->cam->use_stereo = false;
  auto state = std::make_shared<mins::State>(options);
  state->cam_intrinsic_model[0] = std::make_shared<ov_core::CamRadtan>(320, 240);
  Eigen::VectorXd intrinsics(8);
  intrinsics << 200, 200, 160, 120, 0, 0, 0, 0;
  state->cam_intrinsic_model.at(0)->set_value(intrinsics);
  mins::UpdaterCamera updater(state);
  cv::Mat texture(240, 320, CV_8UC1);
  cv::RNG random(4390);
  random.fill(texture, cv::RNG::UNIFORM, 0, 256);
  cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.6);
  ov_core::CameraData frame;
  frame.sensor_ids = {0};
  frame.images = {texture};
  frame.masks = {cv::Mat::zeros(texture.size(), CV_8UC1)};
  for (double time : {1.0, 1.1, 1.6}) {
    frame.timestamp = time;
    updater.feed_measurement(frame);
  }
  double mean, spread;
  int before = 0, after = 0;
  auto history = updater.get_observation_history(0);
  ov_core::FeatureHelper::compute_disparity(history, 1.1, 1.6, mean, spread, before);
  if (before < 20 || mean > 0.1)
    throw std::runtime_error("fixture did not retain quiet visual tracks");
  // The same extraction/cleanup used when selecting existing SLAM landmarks.
  auto measurements = updater.get_tracking_database(0);
  for (const auto &item : measurements->get_internal_data())
    measurements->get_feature(item.first, true);
  measurements->cleanup();
  ov_core::FeatureHelper::compute_disparity(history, 1.1, 1.6, mean, spread, after);
  if (after != before || !std::isfinite(mean) || mean > 0.1)
    throw std::runtime_error("filter consumption erased stationary observations");
  // Real movement still appears in the independent history; no rest is invented.
  cv::Mat moved, shift = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, 0);
  cv::warpAffine(texture, moved, shift, texture.size());
  frame.images = {moved};
  frame.timestamp = 1.7;
  updater.feed_measurement(frame);
  ov_core::FeatureHelper::compute_disparity(history, 1.6, 1.7, mean, spread, after);
  if (after < 20 || mean < 3)
    throw std::runtime_error("observation history failed to expose fresh motion");
  std::cout << "Stationary history survives filter consumption and retains fresh motion\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
