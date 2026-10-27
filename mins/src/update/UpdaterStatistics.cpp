#include "utils/gestalt_hooks.h"
/*
 * MINS: Efficient and Robust Multisensor-aided Inertial Navigation System
 * Copyright (C) 2019-2026 Woosik Lee
 * Copyright (C) 2019-2026 Guoquan Huang
 * Copyright (C) 2019-2026 MINS Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "UpdaterStatistics.h"

#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/Jabdongsani.h"
#include "utils/Print_Logger.h"
#include "utils/chi_square.h"

using namespace mins;

UpdaterStatistics::UpdaterStatistics(double chi2_mult, string type, int id) : type(type), id(id), chi2_mult(chi2_mult) {
  for (int i = 1; i < max_chi_size; i++)
    chi_squared_table[i] = chi_square_quantile(0.95, i);
  chi_stat = make_shared<STAT>();
  res_stat = make_shared<STAT>();
  std_stat = make_shared<STAT>();
}

bool UpdaterStatistics::Chi2Check(StatePtr state, const VecTypePtr &H_order, const MatrixXd &H, const VectorXd &res, bool print) {
  return Chi2Check(StateHelper::get_marginal_covariance(state, H_order), H, res, MatrixXd::Zero(res.rows(), res.rows()), print);
}

bool UpdaterStatistics::Chi2Check(StatePtr state, const VecTypePtr &H_order, const MatrixXd &H, const VectorXd &res, const MatrixXd &R, bool print) {
  return Chi2Check(StateHelper::get_marginal_covariance(state, H_order), H, res, R, print);
}

bool UpdaterStatistics::Chi2Check(const MatrixXd &P, const MatrixXd &H, const VectorXd &res, const MatrixXd &R, bool print) {

  // Get Chi2 values
  double chi, thr;
  if (!get_chi2(chi, thr, P, H, res, R)) {
    gestalt_mins_gate(type.c_str(), -1, -1, -1);
    return false;
  }
  gestalt_mins_gate(type.c_str(), chi < thr ? 1 : 0, chi, thr);

  // Record statistics
  res_stat->add_stat(res);
  std_stat->add_stat(R);
  chi_stat->add_stat(chi);

  // return Chi square test results
  if (chi < thr) {
    n_acp++;
    n_meas += (int)res.rows();
    if (print) {
      string sensor = type + (id < 0 ? "" : to_string(id));
      PRINT2("[%s] chi: %.2g accepted (< %.2g). acp/rej: %d/%d", sensor.c_str(), chi, thr, n_acp, n_rej);
      if (n_acp + n_rej != 0)
        PRINT2(" (%.2f %%, mean: %.2g, std %.2g)\n", (double)n_acp / (n_acp + n_rej) * 100, chi_stat->mean, chi_stat->std());
      else
        PRINT2("\n");
    }
    return true;
  } else {
    n_rej++;
    if (print) {
      string sensor = type + (id < 0 ? "" : to_string(id));
      PRINT2("[%s] chi: %.2g rejected (> %.2g). acp/rej: %d/%d", sensor.c_str(), chi, thr, n_acp, n_rej);
      if (n_acp + n_rej != 0)
        PRINT2(" (%.2f %%, mean: %.2g, std %.2g)\n", (double)n_acp / (n_acp + n_rej) * 100, chi_stat->mean, chi_stat->std());
      else
        PRINT2("\n");
    }
    return false;
  }
}

void UpdaterStatistics::print() {
  string sensor = type + (id < 0 ? "" : to_string(id));
  PRINT2("[%s] Avg res/std: %s%.0e/%.0e | R std: %.0e", sensor.c_str(), res_stat->mean < 0 ? "" : " ", res_stat->mean, res_stat->std(), std_stat->mean);

  if (n_acp + n_rej != 0)
    PRINT2(", Chi %.f %%, mean: %.0e, std %.0e", (double)n_acp / (n_acp + n_rej) * 100, chi_stat->mean, chi_stat->std());
  PRINT2(", # meas: %d\n", n_meas);
}
bool UpdaterStatistics::get_chi2(double &chi, double &thr, const MatrixXd &P, const MatrixXd &H, const VectorXd &res, const MatrixXd &R) {

  // Numerical failure is distinct from a valid rejected innovation.
  if (res.size() == 0 || P.rows() != P.cols() || H.cols() != P.rows() ||
      H.rows() != res.rows() || R.rows() != res.rows() || R.cols() != res.rows() ||
      !P.allFinite() || !H.allFinite() || !res.allFinite() || !R.allFinite())
    return false;

  // check dimensions right
  if (H.rows() != R.rows()) {
    string sensor = type + (id < 0 ? "" : to_string(id));
    PRINT2("[%s] H.rows(): %ld, R.rows(): %ld\n", sensor.c_str(), H.rows(), R.rows());
    assert(H.rows() == R.rows());
  }
  if (H.cols() != P.rows()) {
    string sensor = type + (id < 0 ? "" : to_string(id));
    PRINT2("[%s] H.cols(): %ld, P.rows(): %ld\n", sensor.c_str(), H.cols(), P.rows());
    assert(H.cols() == P.rows());
  }

  // test chi
  MatrixXd S;
  VectorXd projected = res;
  double discarded_chi2 = 0;
  // Camera residuals have already been whitened. Compress their tall system
  // before forming/factoring the innovation. Preserve the orthogonal residual
  // energy AND the original measurement count for exactly the same gate.
  // Other sensors/correlated noise retain the general covariance path.
  if (H.rows() > H.cols() && H.cols() > 0 && R.isIdentity(0.0)) {
    Eigen::HouseholderQR<MatrixXd> qr(H);
    VectorXd rotated = qr.householderQ().adjoint() * res;
    MatrixXd reduced = qr.matrixQR().topRows(H.cols()).template triangularView<Upper>();
    S = (reduced * P * reduced.transpose()).selfadjointView<Upper>();
    S.diagonal().array() += 1.0;
    projected = rotated.head(H.cols());
    discarded_chi2 = rotated.tail(H.rows() - H.cols()).squaredNorm();
  } else {
    S = (H * P * H.transpose() + R).selfadjointView<Upper>();
  }
  if (!S.allFinite()) return false;
  Eigen::LLT<MatrixXd> innovation(S);
  if (innovation.info() != Eigen::Success) return false;
  chi = projected.dot(innovation.solve(projected)) + discarded_chi2;
  if (res.rows() < max_chi_size) {
    thr = chi2_mult * chi_squared_table[res.rows()];
  } else {
    thr = chi2_mult * chi_square_quantile(0.95, res.rows());
  }
  if (!std::isfinite(chi) || chi < 0 || !std::isfinite(thr) || thr <= 0) {
    string sensor = type + (id < 0 ? "" : to_string(id));
    PRINT2("[%s] Got NAN during check. Here are debug messages.\n", sensor.c_str());

    PRINT2("res\n");
    for (int i = 0; i < res.rows(); i++) {
      PRINT2("%.2e\n", res(i));
    }

    PRINT2("H\n");
    for (int i = 0; i < H.rows(); i++) {
      for (int j = 0; j < H.cols(); j++) {
        PRINT2("%.2e ", H(i, j));
      }
      PRINT2("\n");
    }

    PRINT2("R\n");
    for (int i = 0; i < R.rows(); i++) {
      for (int j = 0; j < R.cols(); j++) {
        PRINT2("%.2e ", R(i, j));
      }
      PRINT2("\n");
    }

    PRINT2("S\n");
    for (int i = 0; i < S.rows(); i++) {
      for (int j = 0; j < S.cols(); j++) {
        PRINT2("%.2e ", S(i, j));
      }
      PRINT2("\n");
    }

    PRINT2("chi: %.2e\n", chi);
    return false;
  }
  return true;
}