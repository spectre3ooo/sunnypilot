"""
Opt-in steering-angle feedforward for the torque lateral controllers (LatAngleFeedforward). See model.py.

The model gives the torque the car needs along the desired steering angle and the lateral accel. Below
BLEND_SPEEDS[0] it replaces the lateral-accel feedforward entirely; it fades back to it by BLEND_SPEEDS[1]. The
model was fitted at every speed, but the first on-car test keeps the controller's proven highway behaviour.
PID feedback is unchanged. Skipped when NNLC drives the feedforward or no model exists for the car.
"""
import math

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.sunnypilot.selfdrive.controls.lib.angle_ff.model import load_model

MPH = 0.44704
BLEND_SPEEDS = [30 * MPH, 45 * MPH]  # full angle feedforward below the first, none above the second
RATE_FILTER_HZ = 1.2                 # the desired-angle rate is differentiated, so low-pass it like the jerk
# Replaying recorded drives through this controller (docs/lat/angle-feedforward.md): driving the model's damping
# and friction terms from the planned angle rate pushes the wheel toward centre on every planned unwind, on top of
# the return the EPS and steering geometry already give, and passes planner noise straight into torque steps.
# Without them the feedforward beats the lateral-accel one on reverse torque, step size and time at the clip.
USE_RATE_TERM = False
# Optional low-pass on the desired angle (Hz, 0 = off). 1 Hz was marginally smoother in replay but delays the
# feedforward by ~0.16 s on top of the steering lag, so it is off unless the car feels jumpy.
ANGLE_FILTER_HZ = 0.0
MAX_ANGLE = 600.0                    # deg; the model is flat beyond its last node anyway


class LatAngleFeedforward:
  def init_angle_ff(self, CP) -> None:
    # read once: an arm changes between drives, not mid-engagement
    self.angle_ff_model = load_model(CP.carFingerprint) if Params().get_bool("LatAngleFeedforward") else None
    self.angle_ff_weight = 0.0
    self.angle_ff_torque = 0.0
    self._angle_des_last: float | None = None
    self._angle_rate_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * RATE_FILTER_HZ), self.dt)
    self._angle_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * ANGLE_FILTER_HZ) if ANGLE_FILTER_HZ > 0 else 0.0, self.dt)

  def _reset_angle_ff(self) -> None:
    self.angle_ff_weight = 0.0
    self.angle_ff_torque = 0.0
    self._angle_des_last = None
    self._angle_rate_filter.x = 0.0

  def angle_ff_active(self) -> bool:
    return self.angle_ff_model is not None and not getattr(self.extension, "_nnlc_enabled", False)

  def update_angle_ff(self, active: bool, CS, VM, params, desired_curvature: float, gravity_adjusted_lat_accel: float):
    """-> (weight, torque). weight scales down the caller's lateral-accel feedforward; torque, in the controller's
    output_torque convention, is added after the lateral-accel -> torque conversion."""
    if not active or not self.angle_ff_active():
      self._reset_angle_ff()
      return 0.0, 0.0

    # steeringAngleDeg that produces the desired curvature: controlsd measures curvature as -VM.calc_curvature(angle)
    angle_des = math.degrees(VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll)) + params.angleOffsetDeg
    angle_des = float(np.clip(angle_des, -MAX_ANGLE, MAX_ANGLE))
    if ANGLE_FILTER_HZ > 0:
      if self._angle_des_last is None:
        self._angle_filter.x = angle_des
      angle_des = self._angle_filter.update(angle_des)
    rate = 0.0 if self._angle_des_last is None else self._angle_rate_filter.update((angle_des - self._angle_des_last) / self.dt)
    if not USE_RATE_TERM:
      rate = 0.0
    self._angle_des_last = angle_des

    weight = float(np.interp(CS.vEgo, BLEND_SPEEDS, [1.0, 0.0]))
    # the model is fitted in actuator convention (actuators.torque); the controller negates its output into that
    torque = -weight * self.angle_ff_model.evaluate(CS.vEgo, angle_des, rate, gravity_adjusted_lat_accel)
    self.angle_ff_weight, self.angle_ff_torque = weight, torque
    return weight, torque
