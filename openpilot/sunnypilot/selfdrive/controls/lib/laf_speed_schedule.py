"""
Opt-in speed schedule on latAccelFactor for the torque lateral controllers (LatAccelFactorSpeedSchedule).

torqued learns one latAccelFactor from data above 15 m/s, but on the Ioniq 5 lateral accel per unit torque falls
at low speed (docs/lat/learner-bands.md: below 25 mph about 0.55x the highway value by one estimator, 0.8x by
another). Scaling latAccelFactor down there makes the feedforward ask for the torque the car actually needs.
The values sit between the two estimators and end at 1.0 where torqued's own data starts; they are a starting
point for the A/B, not a fit.
"""
import numpy as np

from openpilot.common.params import Params

LAF_SPEED_SCHEDULE_BP = [4.5, 8.9, 15.2]  # m/s: 10, 20, 34 mph
LAF_SPEED_SCHEDULE_V = [0.6, 0.75, 1.0]


class LatAccelFactorSpeedSchedule:
  def init_laf_speed_schedule(self, CP) -> None:
    # read once: an arm changes between drives, not mid-engagement
    self.laf_speed_schedule = Params().get_bool("LatAccelFactorSpeedSchedule")
    self.laf_scale = 1.0
    self.scheduled_params = CP.lateralTuning.torque.as_builder()

  def get_laf_scale(self, v_ego: float) -> float:
    # NNLC replaces this feedforward with its own model, which needs its own correction
    if not self.laf_speed_schedule or getattr(self.extension, "_nnlc_enabled", False):
      return 1.0
    return float(np.interp(v_ego, LAF_SPEED_SCHEDULE_BP, LAF_SPEED_SCHEDULE_V))

  def get_scheduled_params(self, v_ego: float):
    """torque_params with latAccelFactor scaled for speed. torque_params itself stays the learned (or overridden)
    value, so the scale never compounds. Friction is applied in torque space, so it is unaffected; the PID's
    output limits move with the factor, since full torque now maps to less lateral accel."""
    self.laf_scale = self.get_laf_scale(v_ego)
    if not self.laf_speed_schedule:
      return self.torque_params
    params = self.scheduled_params
    params.latAccelFactor = float(self.torque_params.latAccelFactor) * self.laf_scale
    params.latAccelOffset = self.torque_params.latAccelOffset
    params.friction = self.torque_params.friction
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, params),
                        self.lateral_accel_from_torque(-self.steer_max, params))
    return params
