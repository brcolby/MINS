// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <opencv2/calib3d.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ov_core {

// Failed optical-flow predictions are not observations. Including them in
// geometric consensus both biases the fitted model and makes RANSAC spend its
// iteration budget trying to explain points that tracking already rejected.
inline std::vector<uchar> fundamental_inliers(const std::vector<cv::Point2f> &first,
                                             const std::vector<cv::Point2f> &second,
                                             const std::vector<uchar> &tracked,
                                             double threshold) {
  if (first.size() != second.size() || first.size() != tracked.size() ||
      !std::isfinite(threshold) || threshold <= 0)
    throw std::invalid_argument("Invalid geometric-verification inputs");
  std::vector<cv::Point2f> left, right;
  std::vector<size_t> indices;
  left.reserve(first.size()); right.reserve(first.size()); indices.reserve(first.size());
  std::vector<uchar> output(first.size(), 0);
  for (size_t i = 0; i < tracked.size(); ++i) {
    if (!tracked[i] || !std::isfinite(first[i].x) || !std::isfinite(first[i].y) ||
        !std::isfinite(second[i].x) || !std::isfinite(second[i].y))
      continue;
    left.push_back(first[i]); right.push_back(second[i]); indices.push_back(i);
  }
  // Preserve the tracker's minimum of ten usable correspondences. Unknown or
  // failed fits remain rejected, never promoted to valid by a performance path.
  if (indices.size() < 10) return output;
  std::vector<uchar> inliers;
  cv::findFundamentalMat(left, right, cv::FM_RANSAC, threshold, 0.999, inliers);
  for (size_t i = 0; i < inliers.size() && i < indices.size(); ++i)
    output[indices[i]] = inliers[i] ? 1 : 0;
  return output;
}
} // namespace ov_core
