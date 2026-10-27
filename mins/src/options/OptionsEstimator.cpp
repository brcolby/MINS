/*
 * MINS: Efficient and Robust Multisensor-aided Inertial Navigation System
 * Copyright (C) 2019-2026 Woosik Lee
 * Copyright (C) 2019-2026 Guoquan Huang
 * Copyright (C) 2019-2026 MINS Contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "OptionsEstimator.h"
#include "OptionsCamera.h"
#include "OptionsGPS.h"
#include "OptionsIMU.h"
#include "OptionsInit.h"
#include "OptionsLidar.h"
#include "OptionsVicon.h"
#include "OptionsWheel.h"
#include "utils/Print_Logger.h"
#include "utils/opencv_yaml_parse.h"

void mins::OptionsEstimator::load(const std::shared_ptr<ov_core::YamlParser> &parser) {
  if (parser == nullptr) {
    intr_err.mlt = 3;
    intr_err.reading_ori_slope = {0.0504,0.03465,0.03303,0.0351,0.04842};
    intr_err.reading_pos_slope = {0.05463,0.02493,0.0216,0.02079,0.02124};
    intr_err.set_values(4);
    intr_err.reading_ori_slope = {0.0306,0.01953,0.018,0.01791,0.02016};
    intr_err.reading_pos_slope = {0.03501,0.01422,0.01197,0.01125,0.01098};
    intr_err.set_values(5);
    intr_err.reading_ori_slope = {0.02349,0.01359,0.01224,0.01179,0.01224};
    intr_err.reading_pos_slope = {0.02538,0.00936,0.00792,0.00765,0.00765};
    intr_err.set_values(6);
    intr_err.reading_ori_slope = {0.018,0.00954,0.00837,0.00801,0.00801};
    intr_err.reading_pos_slope = {0.01899,0.00648,0.00549,0.00531,0.00531};
    intr_err.set_values(7);
    intr_err.reading_ori_slope = {0.00464,0.0022,0.00188,0.00184,0.00184};
    intr_err.reading_pos_slope = {0.00524,0.001,0.00088,0.00084,0.00084};
    intr_err.set_values(9);
    intr_err.reading_ori_slope = {0.00288,0.00126,0.00108,0.00102,0.00102};
    intr_err.reading_pos_slope = {0.00312,0.00087,0.00072,0.00066,0.00066};
    intr_err.set_values(10);
    intr_err.reading_ori_slope = {0.00138,0.00066,0.00063,0.00069,0.00087};
    intr_err.reading_pos_slope = {0.00144,0.00021,0.00018,0.00015,0.00015};
    intr_err.set_values(15);
    intr_err.reading_ori_slope = {0.00084,0.00012,6e-05,3e-05,3e-05};
    intr_err.reading_pos_slope = {0.00084,9e-05,6e-05,3e-05,3e-05};
    intr_err.set_values(20);
    intr_err.reading_ori_slope = {0.00051,0.00012,9e-05,9e-05,9e-05};
    intr_err.reading_pos_slope = {0.00054,6e-05,3e-05,3e-05,3e-05};
    intr_err.set_values(25);
    intr_err.reading_ori_slope = {0.00036,6e-05,3e-05,3e-05,3e-05};
    intr_err.reading_pos_slope = {0.00036,3e-05,3e-05,3e-05,0.0};
    intr_err.set_values(30);
  }
  if (parser != nullptr) {
    std::string f = "config_estimator";
    parser->parse_external(f, "est", "gravity_mag", gravity(2));
    parser->parse_external(f, "est", "use_imu_res", use_imu_res);
    parser->parse_external(f, "est", "use_imu_cov", use_imu_cov);
    parser->parse_external(f, "est", "use_pol_cov", use_pol_cov);
    parser->parse_external(f, "est", "window_size", window_size);
    parser->parse_external(f, "est", "clone_freq", clone_freq);
    parser->parse_external(f, "est", "dt_extrapolation", dt_exp);
    parser->parse_external(f, "est", "intr_order", intr_order);
    parser->parse_external(f, "est", "dynamic_cloning", dynamic_cloning);
    parser->parse_external(f, "est", "intr_error_mlt", intr_err.mlt);
    for (int i = 1; i < 40; i++) {
      parser->parse_external(f, "intr_ori", "Hz_" + std::to_string(i), intr_err.reading_ori_slope, false);
      parser->parse_external(f, "intr_pos", "Hz_" + std::to_string(i), intr_err.reading_pos_slope, false);
      intr_err.set_values(i);
    }
    parser->parse_external(f, "est", "intr_error_ori_thr", intr_err.threshold_ori);
    parser->parse_external(f, "est", "intr_error_pos_thr", intr_err.threshold_pos);
    double thr_mlt = 1.0;
    parser->parse_external(f, "est", "intr_error_thr_mlt", thr_mlt);
    intr_err.threshold_ori *= thr_mlt;
    intr_err.threshold_pos *= thr_mlt;
    // Must have at least one order
    if (intr_order < 1 || intr_order % 2 == 0) {
      PRINT4(RED "Estimator polynomial order should be >= 1 and odd number. Current value: %d\n" RESET, intr_order);
      std::exit(EXIT_FAILURE);
    }

    // Check if max clone works with interpolation order
    double max_clone_size = window_size * clone_freq + 1;
    if (max_clone_size < intr_order + 1) {
      PRINT4(RED "Max clone size is smaller than required for polynomial interpolation (%d < %f + 1).\n" RESET, max_clone_size, intr_order);
      PRINT4(RED "Set Max clone size = %d.\n" RESET, intr_order + 1);
      std::exit(EXIT_FAILURE);
    }
    // Set max_clone_size >= 6 (5 + 1) so that clones can cover the largest interpolation order
    if (dynamic_cloning && max_clone_size < 6) {
      PRINT4(RED "Max clone size is smaller than required for polynomial interpolation (%d < 6).\n" RESET, max_clone_size);
      PRINT4(RED "Set Max clone size = 6.\n" RESET);
      std::exit(EXIT_FAILURE);
    }

    if (use_imu_cov + use_pol_cov > 1) {
      PRINT4(RED "More than 1 cov method enabled.\n" RESET);
      PRINT4(RED "use_imu_cov: %s\n" RESET, use_imu_cov ? "true" : "false");
      PRINT4(RED "use_pol_cov: %s\n" RESET, use_pol_cov ? "true" : "false");
      std::exit(EXIT_FAILURE);
    }
  }
  init = make_shared<OptionsInit>();
  lidar = make_shared<OptionsLidar>();
  wheel = make_shared<OptionsWheel>();
  gps = make_shared<OptionsGPS>();
  vicon = make_shared<OptionsVicon>();
  cam = make_shared<OptionsCamera>();
  imu = make_shared<OptionsIMU>();
  init->load(parser);
  imu->load(parser);
  cam->load(parser);
  gps->load(parser);
  wheel->load(parser);
  lidar->load(parser);
  vicon->load(parser);
}

void mins::OptionsEstimator::print() {
  PRINT1(BOLDBLUE "Options - Estimator\n" RESET);
  PRINT1("\t- gravity: %.3f, %.3f, %.3f\n", 0.0, 0.0, gravity(2));
  PRINT1("\t- window_size: %f\n", window_size);
  PRINT1("\t- clone_freq: %d\n", clone_freq);
  PRINT1("\t- dt_extrapolation: %.2f\n", dt_exp);
  PRINT1("\t- intr_order: %d\n", intr_order);
  PRINT1("\t- dynamic_cloning: %s\n", dynamic_cloning ? "true" : "false");
  PRINT1("\t- use_imu_res: %s\n", use_imu_res ? "true" : "false");
  PRINT1("\t- use_imu_cov: %s\n", use_imu_cov ? "true" : "false");
  PRINT1("\t- use_pol_cov: %s\n", use_pol_cov ? "true" : "false");
  PRINT1("\t- intr_error_mlt: %.3f\n", intr_err.mlt);
  PRINT1("\t- intr_error_threshold_ori: %.6f\n", intr_err.threshold_ori);
  PRINT1("\t- intr_error_threshold_pos: %.6f\n", intr_err.threshold_pos);
  imu->print();
  cam->print();
  vicon->print();
  gps->print();
  wheel->print();
  lidar->print();
  init->print();
}

void mins::OptionsEstimator::load_print(const std::shared_ptr<ov_core::YamlParser> &parser) {
  load(parser);
  print();
}

void mins::OptionsEstimator::interpolation_error::set_values(int hz) {
  assert(reading_ori_slope.size() == 5);
  assert(reading_pos_slope.size() == 5);
  bool valid_slope = true;
  for (int i = 0; i < 5; i++) {
    if (reading_ori_slope.at(i) < 0 || reading_pos_slope.at(i) < 0)
      valid_slope = false;
  }

  // return if the reading is not valid
  if (!valid_slope) {
    // reset the reading value
    for (int i = 0; i < 5; i++) {
      reading_ori_slope.at(i) = -1;
      reading_pos_slope.at(i) = -1;
    }
    return;
  }

  // Insert a new map for this clone Hz
  ori_slope.insert({hz, std::map<int, double>()});
  pos_slope.insert({hz, std::map<int, double>()});

  // Insert order values
  int cnt = 0;
  for (int j = 1; j <= 9; j += 2) {
    ori_slope.at(hz).insert({j, reading_ori_slope[cnt]});
    pos_slope.at(hz).insert({j, reading_pos_slope[cnt]});
    cnt++;
  }

  // reset the reading value
  for (int i = 0; i < 5; i++) {
    reading_ori_slope.at(i) = -1;
    reading_pos_slope.at(i) = -1;
  }
}
