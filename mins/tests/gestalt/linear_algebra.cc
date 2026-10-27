// Factorization changes must preserve measurement information and EKF updates.
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
#include "types/Vec.h"
#include "utils/chi_square.h"
#include "update/UpdaterStatistics.h"
#include <Eigen/Cholesky>
#include <cmath>
#include <iostream>
#include <stdexcept>

extern "C" void gestalt_mins_update(const char*) {}
extern "C" void gestalt_mins_covariance_rejected(const char*) {}
extern "C" void gestalt_mins_gps_before(const double*, const double*) {}
extern "C" void gestalt_mins_gps_after(const double*, const double*) {}
static double last_gate_chi = -1;
extern "C" void gestalt_mins_gate(const char*, int, double chi, double) { last_gate_chi = chi; }
extern "C" void gestalt_mins_tracked_features(double, int) {}
extern "C" void gestalt_mins_feature_stage(const char*, bool) {}

static Eigen::MatrixXd data(int rows, int cols) {
  Eigen::MatrixXd x(rows, cols);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      x(i,j) = std::sin((i+1)*(j+2)*0.713) + .2*std::cos((i+3)*(j+1)*.219);
  return x;
}

int main() try {
  // Include the long, tall systems that dominate high-rate image updates,
  // dependent columns, and the no-compression (wide matrix) case.
  for (int rows : {5, 60, 4800}) {
    Eigen::MatrixXd H = data(rows, 18);
    H.col(17) = H.col(0) + H.col(2);
    Eigen::VectorXd r = data(rows, 1);
    Eigen::MatrixXd information = H.transpose()*H;
    Eigen::VectorXd rhs = H.transpose()*r;
    mins::StateHelper::measurement_compress_inplace(H, r);
    if (H.rows() != std::min(rows, 18) || H.rows() != r.rows() ||
        !(H.transpose()*H).isApprox(information, 2e-11) ||
        !(H.transpose()*r).isApprox(rhs, 2e-11))
      throw std::runtime_error("compression changed measurement information");
  }
  for (int measurements : {3, 25, 100}) {
    auto options = std::make_shared<mins::OptionsEstimator>();
    options->load();
    options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
    options->vicon->enabled = options->wheel->enabled = false;
    options->use_imu_res = true;
    auto state = std::make_shared<mins::State>(options);
    for (int i = 0; i < 4; ++i) {
      state->time = i;
      mins::StateHelper::augment_clone(state);
    }
    int n = state->cov.rows();
    Eigen::MatrixXd A = data(n,n);
    Eigen::MatrixXd P = .01*A*A.transpose() + .5*Eigen::MatrixXd::Identity(n,n);
    state->cov = P;
    std::vector<std::shared_ptr<ov_type::Type>> order = {
      state->imu->v(), state->clones.at(2)->p(), state->imu->bg()};
    Eigen::MatrixXd H = data(measurements, 9);
    Eigen::MatrixXd Hfull = Eigen::MatrixXd::Zero(measurements,n);
    for (int i = 0; i < 3; ++i)
      Hfull.middleCols(order[i]->id(),3) = H.middleCols(3*i,3);
    Eigen::VectorXd r = .001*data(measurements,1);
    Eigen::MatrixXd B = data(measurements,measurements);
    Eigen::MatrixXd R = .01*B*B.transpose() + .1*Eigen::MatrixXd::Identity(measurements,measurements);
    Eigen::MatrixXd S = Hfull*P*Hfull.transpose() + R;
    Eigen::MatrixXd K = P*Hfull.transpose()*S.llt().solve(Eigen::MatrixXd::Identity(measurements,measurements));
    Eigen::MatrixXd expected_cov = P - K*Hfull*P;
    Eigen::VectorXd dx = K*r;
    Eigen::Vector3d expected_velocity = state->imu->vel() + dx.segment(state->imu->v()->id(),3);
    if (!mins::StateHelper::EKFUpdate(state, order, H, r, R, "CAM") ||
        !state->cov.isApprox(expected_cov,2e-10) ||
        !state->imu->vel().isApprox(expected_velocity,2e-10))
      throw std::runtime_error("factorized EKF update differs from reference");
  }
  for (bool correlated : {false, true}) {
    Eigen::MatrixXd H = data(240,9);
    H.col(8) = H.col(0) + H.col(2);
    Eigen::MatrixXd A = data(9,9);
    Eigen::MatrixXd P = .01*A*A.transpose() + .2*Eigen::MatrixXd::Identity(9,9);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(240,240);
    if (correlated) R(0,1) = R(1,0) = 1e-6;
    Eigen::VectorXd r = data(240,1);
    Eigen::MatrixXd S = H*P*H.transpose() + R;
    double reference = r.dot(S.llt().solve(r));
    for (bool accept : {false, true}) {
      mins::UpdaterStatistics gate(reference*(accept ? 1.000001 : .999999)/
                                    mins::chi_square_quantile(.95,240),"CAM");
      if (gate.Chi2Check(P,H,r,R) != accept ||
          std::abs(last_gate_chi-reference) > 2e-10*reference)
        throw std::runtime_error("compressed innovation changed statistic or gate");
    }
  }
  // Compare landmark initialization with the full, uncompressed innovation.
  // Set the gate immediately above/below its statistic: the discarded residual
  // energy must still reject outliers even though it cannot change the state.
  for (int rows : {9, 240}) for (bool accept : {false, true}) {
    auto options = std::make_shared<mins::OptionsEstimator>();
    options->load();
    options->cam->enabled = options->gps->enabled = options->lidar->enabled = false;
    options->vicon->enabled = options->wheel->enabled = false;
    options->use_imu_res = true;
    auto actual = std::make_shared<mins::State>(options);
    auto reference = std::make_shared<mins::State>(options);
    auto order = [](const std::shared_ptr<mins::State> &s) {
      return std::vector<std::shared_ptr<ov_type::Type>>{
        s->imu->v(), s->imu->bg(), s->imu->ba()};
    };
    Eigen::MatrixXd H = .01*data(rows,9), L = data(rows,3);
    Eigen::MatrixXd R = .5*Eigen::MatrixXd::Identity(rows,rows);
    Eigen::VectorXd r = .2*data(rows,1) + L*Eigen::Vector3d(.02,.03,.04);
    Eigen::MatrixXd Href = H, Lref = L;
    Eigen::VectorXd rref = r;
    mins::StateHelper::Givens_Rotation(Lref, Href, rref);
    Eigen::MatrixXd Hup = Href.bottomRows(rows-3);
    Eigen::VectorXd rup = rref.tail(rows-3);
    Eigen::MatrixXd Rup = R.bottomRightCorner(rows-3,rows-3);
    Eigen::MatrixXd P = mins::StateHelper::get_marginal_covariance(reference,order(reference));
    Eigen::MatrixXd S = Hup*P*Hup.transpose() + Rup;
    double chi = rup.dot(S.llt().solve(rup));
    double multiplier = chi*(accept ? 1.000001 : .999999)/mins::chi_square_quantile(.95,rows);
    auto landmark = std::make_shared<ov_type::Vec>(3);
    Eigen::MatrixXd before = actual->cov;
    bool result = mins::StateHelper::initialize(actual,landmark,order(actual),H,L,R,r,multiplier,"CAM",true);
    if (result != accept) throw std::runtime_error("compressed landmark gate changed decision");
    if (!accept) {
      if (actual->cov != before) throw std::runtime_error("rejected landmark changed covariance");
      continue;
    }
    auto expected = std::make_shared<ov_type::Vec>(3);
    if (!mins::StateHelper::initialize_invertible(reference,expected,order(reference),
          Href.topRows(3),Lref.topRows(3),R.topLeftCorner(3,3),rref.head(3),"CAM",true) ||
        !mins::StateHelper::EKFUpdate(reference,order(reference),Hup,rup,Rup,"CAM") ||
        !actual->cov.isApprox(reference->cov,2e-10) ||
        !landmark->value().isApprox(expected->value(),2e-10))
      throw std::runtime_error("compressed landmark initialization differs from full update");
  }
  std::cout << "Compression and correlated EKF updates preserve information\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
