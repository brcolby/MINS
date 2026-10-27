// Invalid covariance must never look like a small, accepted innovation.
#include "options/OptionsEstimator.h"
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "update/UpdaterStatistics.h"
#include <iostream>
#include <limits>
#include <stdexcept>

static int gate_result = 0, covariance_rejections = 0;
static double measured_chi = -1;
extern "C" void gestalt_mins_update(const char*) {}
extern "C" void gestalt_mins_covariance_rejected(const char*) { ++covariance_rejections; }
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
extern "C" void gestalt_mins_gate(const char*, int result, double chi, double) {
  gate_result = result;
  measured_chi = chi;
}
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

static std::shared_ptr<mins::State> state() {
  auto options = std::make_shared<mins::OptionsEstimator>();
  options->load();
  options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
  options->vicon->enabled = options->wheel->enabled = false;
  auto value = std::make_shared<mins::State>(options);
  value->cov.setIdentity();
  return value;
}

int main(int argc, char **argv) try {
  if (argc != 2) throw std::runtime_error("select gate, update or propagation");
  const std::string mode(argv[1]);
  if (mode == "gate") {
    mins::UpdaterStatistics gate(1, "CAM");
    Eigen::Matrix2d P, R, H = Eigen::Matrix2d::Identity();
    P << 2, .2, .2, 3;
    R << 1, .2, .2, 2;
    Eigen::Vector2d residual(1, 2);
    if (!gate.Chi2Check(P, H, residual, R) || gate_result != 1 ||
        std::abs(measured_chi - 15.4 / 14.84) > 1e-12)
      throw std::runtime_error("correlated SPD innovation changed");
    // Ten ordinary independent unit residuals have norm sqrt(10) > 3, but
    // their chi-square score is 10 (below the 95% ten-DOF threshold 18.307).
    // The former camera-only norm cutoff rejected them before this test.
    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(10, 10);
    Eigen::VectorXd ordinary = Eigen::VectorXd::Ones(10);
    if (ordinary.norm() <= 3 ||
        !gate.Chi2Check(Eigen::MatrixXd::Zero(10, 10), identity, ordinary, identity) ||
        gate_result != 1 || std::abs(measured_chi - 10) > 1e-12)
      throw std::runtime_error("ordinary long track rejected by innovation gate");
    if (gate.Chi2Check(Eigen::MatrixXd::Zero(10, 10), identity, 2 * ordinary, identity) ||
        gate_result != 0 || std::abs(measured_chi - 40) > 1e-12)
      throw std::runtime_error("inconsistent long track passed innovation gate");
    P = Eigen::Matrix2d::Identity();
    P(0, 0) = -2;
    R.setIdentity();
    residual << 1, 0;
    if (gate.Chi2Check(P, H, residual, R) || gate_result != -1)
      throw std::runtime_error("indefinite innovation was accepted or not diagnosed");
    P.setZero();
    R.setZero();
    if (gate.Chi2Check(P, H, residual, R) || gate_result != -1)
      throw std::runtime_error("singular innovation was not rejected");
    P.setIdentity();
    P(0, 0) = std::numeric_limits<double>::infinity();
    if (gate.Chi2Check(P, H, residual, R) || gate_result != -1)
      throw std::runtime_error("nonfinite innovation was not rejected");
  } else if (mode == "update") {
    auto value = state();
    const Eigen::MatrixXd before = value->cov;
    const Eigen::VectorXd imu_before = value->imu->value();
    Eigen::Vector3d residual(1, 0, std::numeric_limits<double>::quiet_NaN());
    if (mins::StateHelper::EKFUpdate(value, {value->imu->v()},
            Eigen::Matrix3d::Identity(), residual, Eigen::Matrix3d::Identity(), "CAM"))
      throw std::runtime_error("nonfinite update was accepted");
    if (covariance_rejections != 1 || !value->cov.isApprox(before, 0) ||
        !value->imu->value().isApprox(imu_before, 0))
      throw std::runtime_error("rejected update changed state or hid its diagnostic");
  } else if (mode == "propagation") {
    auto value = state();
    bool rejected = false;
    try {
      mins::StateHelper::EKFPropagation(value, {value->imu}, {value->imu},
          Eigen::MatrixXd::Identity(15, 15), -2 * Eigen::MatrixXd::Identity(15, 15));
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("release build continued after negative covariance");
  } else {
    throw std::runtime_error("unknown check");
  }
  std::cout << "Innovation/covariance validity check passed: " << mode << '\n';
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
