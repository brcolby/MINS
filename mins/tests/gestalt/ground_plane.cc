// Numerical/filter invariants only; not simulated trajectory acceptance.
#include "options/OptionsCamera.h"
#include "options/OptionsGPS.h"
#include "options/OptionsLidar.h"
#include "options/OptionsVicon.h"
#include "options/OptionsWheel.h"
#include "update/gestalt/ground_plane.h"
#include <iostream>

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
static Eigen::Matrix3d rotation(double angle, Eigen::Vector3d axis) {
  return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}
int main() try {
  GroundPlaneConfig c;
  c.enabled = true;
  c.rotation_density = c.translation_density = .002;
  c.imu_to_wheel = rotation(.35, {0, 1, 0}) * rotation(1.2, {1, 0, 0});
  c.wheel_in_imu = {.1, -.04, .06};
  // Translate sideways while turning 180 degrees on a tilted plane. Neither
  // sideways slip nor the arbitrary world frame may trigger a plane residual.
  const Eigen::Matrix3d world_to_plane = rotation(.2, {1, 2, 0});
  Eigen::Matrix3d R0 = c.imu_to_wheel.transpose() * world_to_plane;
  Eigen::Matrix3d R1 = c.imu_to_wheel.transpose() * rotation(3.14159, {0, 0, 1}) * world_to_plane;
  Eigen::Vector3d p0 = -R0.transpose() * c.wheel_in_imu;
  Eigen::Vector3d p1 = world_to_plane.transpose() * Eigen::Vector3d(1, 2, 0) - R1.transpose() * c.wheel_in_imu;
  require(GroundPlaneUpdate::observation(R0, p0, R1, p1, c).norm() < 1e-12,
          "valid yaw/lateral motion on a tilted plane was constrained");
  // Numerical derivative at a nonzero residual, with an off-centre IMU.
  R1 = rotation(.03, {1, 2, 3}) * R1;
  p1 += Eigen::Vector3d(.02, .01, .03);
  const Eigen::Vector3d before = GroundPlaneUpdate::observation(R0, p0, R1, p1, c);
  Eigen::Matrix<double, 3, 12> numeric;
  const double eps = 1e-7;
  for (int i = 0; i < 12; ++i) {
    auto a = R0, b = R1;
    auto x = p0, y = p1;
    const Eigen::Vector3d axis = Eigen::Vector3d::Unit(i % 3);
    if (i < 3) a = rotation(-eps, axis) * a;
    else if (i < 6) x += eps * axis;
    else if (i < 9) b = rotation(-eps, axis) * b;
    else y += eps * axis;
    numeric.col(i) = (GroundPlaneUpdate::observation(a, x, b, y, c) - before) / eps;
  }
  require((numeric - GroundPlaneUpdate::jacobian(R0, p0, R1, p1, c)).norm() < 1e-6,
          "ground-plane Jacobian sign or lever-arm frame is incorrect");
  auto options = std::make_shared<mins::OptionsEstimator>();
  options->load();
  options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
  options->vicon->enabled = options->wheel->enabled = false;
  auto state = std::make_shared<mins::State>(options);
  state->initialized = true;
  // Independent clone uncertainty makes the information change easy to audit.
  for (int i = 0; i < 2; ++i) {
    Eigen::VectorXd v = state->imu->value();
    v.head<4>() = ov_core::rot_2_quat(i ? R1 : R0);
    v.segment<3>(4) = i ? p1 : p0;
    state->imu->set_value(v);
    state->imu->set_fej(v);
    state->time = 1 + 1.1 * i; // quiet-window initialization can leave a gap
    mins::StateHelper::augment_clone(state);
  }
  state->cov.setIdentity();
  state->cov *= .01;
  GroundPlaneUpdate update(c);
  update.update(state);
  require(update.accepted == 1 && update.conflicts == 0, "valid ground update rejected");
  require(state->retained_clones.size() == 1 && state->retained_clones.count(1),
          "the fixed reference plane did not retain exactly one clone");
  auto a = state->clones.at(1), b = state->clones.at(2.1);
  require(GroundPlaneUpdate::observation(a->Rot(), a->pos(), b->Rot(), b->pos(), c).norm() < before.norm() * .05,
          "ground update did not reduce the plane residual");
  const auto covariance = state->cov;
  update.update(state);
  require(update.accepted == 1 && state->cov.isApprox(covariance, 0), "reused a clone interval");
  // An inactive model must leave the filter exactly unchanged.
  GroundPlaneUpdate disabled;
  disabled.update(state);
  require(!disabled.accepted && state->cov.isApprox(covariance, 0), "disabled prior changed the filter");
  // Live IMU aliases are not independent cloned observations.
  state->clones[2.15] = state->imu->pose();
  update.update(state);
  require(update.accepted == 1, "live IMU pose was reused as a plane measurement");
  state->clones.erase(2.15);
  // The declared plane is a model, not a contact sensor to discard when the
  // inertial prediction disagrees. Apply its finite weight and report conflict.
  Eigen::VectorXd outlier = state->imu->value();
  outlier.segment<3>(4) += world_to_plane.transpose() * Eigen::Vector3d::UnitZ();
  state->imu->set_value(outlier);
  state->imu->set_fej(outlier);
  state->time = 2.2;
  mins::StateHelper::augment_clone(state);
  const auto prior_covariance = state->cov;
  const auto prior_value = state->imu->value();
  const auto latest = state->clones.at(2.2);
  const double prior_residual = GroundPlaneUpdate::observation(
      a->Rot(), a->pos(), latest->Rot(), latest->pos(), c).norm();
  update.update(state);
  require(update.accepted == 2 && update.conflicts == 1 &&
          std::string(update.reason) == "ground_plane_conflict",
          "off-plane contradiction was not reported");
  require(!prior_covariance.isApprox(state->cov, 0) && !prior_value.isApprox(state->imu->value(), 0) &&
          GroundPlaneUpdate::observation(a->Rot(), a->pos(), latest->Rot(), latest->pos(), c).norm() < prior_residual,
          "model conflict silently discarded the known plane constraint");
  // A conflict must not move the reference plane to the erroneous last pose.
  state->time = 2.3;
  mins::StateHelper::augment_clone(state);
  update.update(state);
  require(update.accepted == 3 && state->retained_clones.size() == 1 && state->retained_clones.count(1),
          "conflicting pose became a new reference plane");
  // GPS may release its initialization keyframes; the plane reference has a
  // separate owner and survives normal sliding-window marginalization.
  state->keyframes.clear();
  state->keyframes_candidate.clear();
  options->window_size = .15;
  mins::StateHelper::marginalize_old_clone(state);
  require(state->clones.count(1) && !state->clones.count(2.1) && state->clones.size() == 3,
          "reference retention prevented bounded history or was erased by GPS cleanup");
  require(state->oldest_keyframe_time() == 2.2,
          "reference clone retained the entire raw IMU history");
  const auto sensor_times = state->interpolation_clone_times();
  require(sensor_times.size() == 2 && sensor_times.front() == 2.2,
          "reference pose entered the sensor interpolation stencil");
  double older, newer;
  require(!state->bounding_times(1.5, older, newer),
          "interpolation bridged the old reference and current sensor window");
  require(state->closest_clone_time(1.1, newer) && newer == 2.2,
          "CPI selected the fixed plane reference as a capture-time clone");
  // A flipped normal is outside this local linearization; fail explicitly,
  // rather than treating the inverted orientation as another valid plane.
  Eigen::VectorXd inverted = state->imu->value();
  inverted.head<4>() = ov_core::rot_2_quat(c.imu_to_wheel.transpose() *
      rotation(3.141592653589793, {1, 0, 0}) * c.imu_to_wheel * a->Rot());
  state->imu->set_value(inverted);
  state->imu->set_fej(inverted);
  state->time = 2.4;
  mins::StateHelper::augment_clone(state);
  const auto before_flip = state->cov;
  bool failed = false;
  try { update.update(state); }
  catch (const std::runtime_error &error) {
    failed = std::string(error.what()).find("normal flipped") != std::string::npos;
  }
  require(failed && state->cov.isApprox(before_flip, 0) && state->imu->value().isApprox(inverted, 0),
          "flipped ground normal did not fail before updating the filter");
  std::cout << "Ground plane frames, Jacobians, yaw/slip freedom and EKF checks passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
