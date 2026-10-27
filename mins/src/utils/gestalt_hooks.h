#pragma once

// Optional observation hooks for an embedding application.
// Ordinary ROS and standalone builds require no Gestalt runtime.
#ifdef GESTALT_DIAGNOSTIC_HOOKS
extern "C" void gestalt_mins_covariance_rejected(const char*);
extern "C" void gestalt_mins_feature_stage(const char*, bool);
extern "C" void gestalt_mins_gate(const char*, int, double, double);
extern "C" void gestalt_mins_gps_after(const double*, const double*);
extern "C" void gestalt_mins_gps_before(const double*, const double*);
extern "C" void gestalt_mins_tracked_features(double, int);
extern "C" void gestalt_mins_update(const char*);
#else
inline void gestalt_mins_covariance_rejected(const char*) {}
inline void gestalt_mins_feature_stage(const char*, bool) {}
inline void gestalt_mins_gate(const char*, int, double, double) {}
inline void gestalt_mins_gps_after(const double*, const double*) {}
inline void gestalt_mins_gps_before(const double*, const double*) {}
inline void gestalt_mins_tracked_features(double, int) {}
inline void gestalt_mins_update(const char*) {}
#endif
