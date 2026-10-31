# Optional embedding updates

These header-only updates are opt-in building blocks for native/ROS embedding.
Ordinary MINS builds do not enable them automatically. The caller supplies
validated capture epochs, measured frame transforms and explicit operating noise.

- `zero_velocity.h`: persistent visual/IMU rest evidence and a velocity update.
- `stationary_bias.h`: joint bias/gravity evidence, retaining tilt uncertainty.
- `ground_plane.h`: an explicit always-grounded assumption with a retained
  covariance-bearing reference; horizontal motion and yaw remain free.

The ground prior needs the optimized retained-reference API and the
`GESTALT_MINS_OPTIMIZED` compile definition. The numerical regressions in
`mins/tests/gestalt` check the update frames, Jacobians and observability.
Application protocols, sensor acquisition and local/global output frames live in
the embedding application.

Bounded acquisition gaps may carry `ImuData.noise_variance_scale` (default 1).
Propagation, CPI generation/reintegration and endpoint interpolation preserve this
per-interval uncertainty without changing measured timestamps or inertial means.
The input adapter owns the loss budget; this library validates finite scales in
1..4096. ZUPT clears its evidence on a scaled sample, requiring a fresh complete
quiet window before velocity/bias updates. Numerical regressions check unchanged
means/Jacobians and scaled process/preintegration covariance in both directions.
