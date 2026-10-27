// Exercise the actual pinned/patched selector, including an acceleration pulse
// at the previously omitted penultimate sample. No estimator or devices run.
#include "utils/sensor_data.h"
#include "state/Propagator.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

// The static library contains these production observation hooks. This test
// only calls Propagator::select_imu_readings, which does not invoke any hook.
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
  mins::Propagator propagator(nullptr);
  for (int i = 0; i < 8; ++i) {
    ov_core::ImuData sample;
    sample.timestamp = 1 + i * 0.01;
    sample.wm = Eigen::Vector3d(i, 2 * i, -i);
    sample.am = Eigen::Vector3d(i == 3 ? 1 : 0, 0, 0);
    propagator.imu_data.push_back(sample);
  }
  for (const auto &bounds : std::vector<std::pair<double, double>>{
           {1.01, 1.04}, {1.015, 1.035}, {1.01, 1.02}, {1.011, 1.019}, {1.0, 1.07}}) {
    std::vector<ov_core::ImuData> selected;
    require(propagator.select_imu_readings(bounds.first, bounds.second, selected), "selection failed");
    require(selected.front().timestamp == bounds.first && selected.back().timestamp == bounds.second,
            "interpolated endpoint timestamps changed");
    std::vector<ov_core::ImuData> interior;
    for (const auto &sample : propagator.imu_data)
      if (sample.timestamp > bounds.first && sample.timestamp < bounds.second)
        interior.push_back(sample);
    require(selected.size() == interior.size() + 2, "an interior sample was lost or duplicated");
    for (size_t i = 0; i < interior.size(); ++i) {
      require(selected[i + 1].timestamp == interior[i].timestamp, "interior sample time changed");
      require(selected[i + 1].am == interior[i].am && selected[i + 1].wm == interior[i].wm,
              "interior measurement changed");
    }
    for (size_t i = 1; i < selected.size(); ++i)
      require(selected[i].timestamp > selected[i - 1].timestamp, "unordered or duplicate selection");
    for (const auto &sample : selected)
      require(std::abs(sample.wm.x() - (sample.timestamp - 1) * 100) < 1e-10,
              "endpoint interpolation changed the linear signal");
    if (bounds.first == 1.01 && bounds.second == 1.04) {
      double impulse = 0;
      for (size_t i = 1; i < selected.size(); ++i)
        impulse += (selected[i].timestamp - selected[i - 1].timestamp) *
                   (selected[i].am.x() + selected[i - 1].am.x()) / 2;
      require(std::abs(impulse - 0.01) < 1e-12, "acceleration pulse was erased");
    }
  }
  std::vector<ov_core::ImuData> unavailable;
  require(!propagator.select_imu_readings(0.99, 1.02, unavailable), "unbracketed start accepted");
  require(!propagator.select_imu_readings(1.01, 1.08, unavailable), "unbracketed end accepted");
  std::cout << "All interior IMU samples and interpolated boundaries preserved\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
