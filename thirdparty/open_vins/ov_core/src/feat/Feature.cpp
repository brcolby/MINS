/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "Feature.h"

using namespace ov_core;

void Feature::clean_old_measurements(const std::vector<double> &valid_times) {

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are not in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(valid_times.begin(), valid_times.end(), *it1) == valid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_invalid_measurements(const std::vector<double> &invalid_times) {
  if (invalid_times.empty()) return;

  // Loop through each of the cameras we have
  for (auto const &pair : timestamps) {

    // Assert that we have all the parts of a measurement
    assert(timestamps[pair.first].size() == uvs[pair.first].size());
    assert(timestamps[pair.first].size() == uvs_norm[pair.first].size());

    // Our iterators
    auto it1 = timestamps[pair.first].begin();
    auto it2 = uvs[pair.first].begin();
    auto it3 = uvs_norm[pair.first].begin();

    // Loop through measurement times, remove ones that are in our timestamps
    while (it1 != timestamps[pair.first].end()) {
      if (std::find(invalid_times.begin(), invalid_times.end(), *it1) != invalid_times.end()) {
        it1 = timestamps[pair.first].erase(it1);
        it2 = uvs[pair.first].erase(it2);
        it3 = uvs_norm[pair.first].erase(it3);
      } else {
        ++it1;
        ++it2;
        ++it3;
      }
    }
  }
}

void Feature::clean_older_measurements(double timestamp) {
  // Compact all three arrays together in one stable pass. Repeated vector::erase
  // shifts the remaining history for every retired observation (quadratic work).
  // Do not assume ordering: merged/retried observations can be out of order.
  for (auto &pair : timestamps) {
    auto &times = pair.second;
    auto &pixels = uvs.at(pair.first);
    auto &normalized = uvs_norm.at(pair.first);
    assert(times.size() == pixels.size() && times.size() == normalized.size());
    size_t kept = 0;
    for (size_t i = 0; i < times.size(); ++i) {
      if (times[i] <= timestamp) continue;
      if (kept != i) {
        times[kept] = times[i];
        pixels[kept] = pixels[i];
        normalized[kept] = normalized[i];
      }
      ++kept;
    }
    times.resize(kept);
    pixels.resize(kept);
    normalized.resize(kept);
  }
}
