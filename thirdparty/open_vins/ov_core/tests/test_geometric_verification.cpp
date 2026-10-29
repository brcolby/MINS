// SPDX-License-Identifier: GPL-3.0-or-later
#include "track/GeometricVerification.h"
#include <iostream>
#include <limits>
#include <numeric>

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  std::vector<cv::Point2f> first, second;
  cv::RNG random(417);
  for (int i = 0; i < 36; ++i) {
    double x=random.uniform(-1.,1.), y=random.uniform(-1.,1.), z=random.uniform(3.,7.);
    first.emplace_back(x/z,y/z);
    second.emplace_back((std::cos(.03)*x+std::sin(.03)*z+.2)/(-std::sin(.03)*x+std::cos(.03)*z),
                        y/(-std::sin(.03)*x+std::cos(.03)*z));
  }
  second[7] += cv::Point2f(.15f,.3f);
  second[21] += cv::Point2f(-.2f,.2f);
  std::vector<uchar> reference;
  cv::findFundamentalMat(first,second,cv::FM_RANSAC,.002, .999,reference);
  require(reference.size()==36,"Reference fit failed");
  auto all=ov_core::fundamental_inliers(first,second,std::vector<uchar>(36,1),.002);
  require(all==reference,"All-tracked input changed its consensus");
  auto original_first=first, original_second=second;
  std::vector<uchar> tracked(36,1);
  const float nan=std::numeric_limits<float>::quiet_NaN();
  for(int i=0;i<500;++i) {
    first.emplace_back(nan,nan); second.emplace_back(float(i),float(-i));tracked.push_back(0);
  }
  auto filtered=ov_core::fundamental_inliers(first,second,tracked,.002);
  require(std::vector<uchar>(filtered.begin(),filtered.begin()+36)==reference,
          "Failed KLT predictions changed valid correspondence consensus");
  require(std::accumulate(filtered.begin()+36,filtered.end(),0)==0,"Failed tracks were promoted");
  require(!filtered[7] && !filtered[21],"Geometric outliers were accepted");
  for(size_t i=9;i<tracked.size();++i)tracked[i]=0;
  filtered=ov_core::fundamental_inliers(first,second,tracked,.002);
  require(std::accumulate(filtered.begin(),filtered.end(),0)==0,"Insufficient valid tracks were accepted");
  tracked.assign(first.size(),0);tracked.back()=1;
  filtered=ov_core::fundamental_inliers(first,second,tracked,.002);
  require(std::accumulate(filtered.begin(),filtered.end(),0)==0,"Nonfinite tracked coordinate was fitted");
  bool rejected=false;
  try {ov_core::fundamental_inliers(first,second,{},.002);}catch(const std::invalid_argument &){rejected=true;}
  require(rejected,"Mismatched inputs were accepted");
  std::cout << "Valid-only geometric consensus preserves masks and rejects failed/nonfinite tracks\n";
}
