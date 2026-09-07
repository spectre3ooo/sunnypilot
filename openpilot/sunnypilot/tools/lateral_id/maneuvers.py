#!/usr/bin/env python3
"""Low-speed lateral system-identification maneuvers (T1.2).

comma's `tools/lateral_maneuvers` set is built around 20-30 mph steps and a single
sine. Fitting a plant model (gain K, lag tau, delay d, friction floor) per speed band
below 40 mph needs richer excitation, and needs it at speeds comma's set does not
cover. This module supplies that set; comma's list is left untouched and is still what
runs unless the `LateralIdMode` param is set.

Enabled with:  LateralManeuverMode=1 (starts the process) and LateralIdMode=1 (uses
this list instead of comma's).

Excitation per speed:
  step          - transient response; gain and delay
  sine 0.5 Hz   - comma's, kept for cross-comparison with their report generator
  log chirp     - frequency response 0.2-2.0 Hz in one run; the primary fit input
  triangle ramp - slow sweep through zero; friction floor and hysteresis
  small steps   - +/-0.1 and +/-0.2 m/s2, resolution near the friction floor

Jerk budget. A sinusoid of amplitude A at frequency f has peak jerk 2*pi*f*A. The
plan asked for a flat +/-0.5 m/s2 chirp to 2.0 Hz, which peaks at 6.28 m/s3 against a
planner envelope of ~3.6 (MAX_LATERAL_JERK), and +/-1.5 would reach 18.8. A flat
chirp therefore cannot satisfy its own acceptance test. Amplitude is instead tapered
as min(A, MAX_LATERAL_JERK / (2*pi*f)), so the sweep runs at full requested amplitude
where the envelope allows and becomes jerk-limited above the corner frequency. This is
normal for swept-sine identification and costs only signal-to-noise at the top of the
sweep, where the plant response is smallest anyway.
"""
import numpy as np

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.tools.lateral_maneuvers.lateral_maneuversd import Maneuver
from openpilot.tools.longitudinal_maneuvers.maneuversd import Action
from opendbc.car.lateral import MAX_LATERAL_ACCEL, MAX_LATERAL_JERK

# Speeds from the plan. 45 mph exists to cross-check the fit against liveTorqueParameters
# and lateralDelay, which are only learned above 15 m/s.
SPEEDS_MPH = [10, 15, 20, 30, 45]

CHIRP_F0 = 0.2          # Hz
CHIRP_F1 = 2.0          # Hz
CHIRP_DURATION = 15.0   # s
CHIRP_AMP_LOW = 1.5     # m/s2, at 10-15 mph where the plant is stiffest
CHIRP_AMP = 0.5         # m/s2, elsewhere

JERK_HEADROOM = 0.9     # stay this fraction under the planner envelope


def _amp_for_freq(amp: float, freq: float) -> float:
  """Largest amplitude at `freq` that keeps peak jerk under the envelope."""
  return min(amp, (JERK_HEADROOM * MAX_LATERAL_JERK) / (2.0 * np.pi * max(freq, 1e-6)))


def _enforce_jerk(a: np.ndarray, limit: float) -> np.ndarray:
  """Scale a profile so its sampled jerk is within `limit`.

  Tapering amplitude by 1/f alone is not sufficient. For A(t)*sin(phi(t)) the
  derivative is A*cos(phi)*dphi/dt + dA/dt*sin(phi); the second term is ignored by a
  taper derived from instantaneous frequency, so the analytic bound comes out slightly
  optimistic (measured 3.236 against an allowance of 3.230). Rather than widen the
  allowance, measure the sampled jerk and scale to fit, which makes the property hold
  by construction and stays correct if the sweep shape changes.
  """
  if len(a) < 2:
    return a
  peak = float(np.max(np.abs(np.diff(a) / DT_MDL)))
  if peak > limit > 0.0:
    a = a * (limit / peak)
  return a


def _chirp_action(amplitude: float, f0: float = CHIRP_F0, f1: float = CHIRP_F1,
                  duration: float = CHIRP_DURATION) -> Action:
  """Logarithmic frequency sweep with jerk-limited amplitude.

  Log sweep spends equal time per octave, so the low frequencies that carry most of
  the gain information are not under-sampled relative to the high ones.
  """
  t = np.arange(0.0, duration + DT_MDL, DT_MDL)
  k = (f1 / f0) ** (t / duration)
  # instantaneous phase of a log sweep
  phase = 2.0 * np.pi * f0 * duration * (k - 1.0) / np.log(f1 / f0)
  freq = f0 * k
  amp = np.array([_amp_for_freq(amplitude, f) for f in freq])
  a = _enforce_jerk(amp * np.sin(phase), JERK_HEADROOM * MAX_LATERAL_JERK)
  return Action(a.tolist(), t.tolist())


def _triangle_action(amplitude: float, duration: float = 8.0) -> Action:
  """Slow symmetric ramp 0 -> +A -> 0 -> -A -> 0.

  Crossing zero slowly is what exposes the friction floor: the torque needed before
  the car begins responding at all.
  """
  t = np.arange(0.0, duration + DT_MDL, DT_MDL)
  q = duration / 4.0
  a = np.interp(t, [0.0, q, 2 * q, 3 * q, duration],
                [0.0, amplitude, 0.0, -amplitude, 0.0])
  return Action(a.tolist(), t.tolist())


def _sine_action(amplitude: float, period: float, duration: float) -> Action:
  t = np.arange(0.0, duration + DT_MDL, DT_MDL)
  a = amplitude * np.sin(2.0 * np.pi * t / period)
  return Action(a.tolist(), t.tolist())


def _for_speed(mph: float) -> list[Maneuver]:
  v = mph * CV.MPH_TO_MS
  chirp_amp = CHIRP_AMP_LOW if mph <= 15 else CHIRP_AMP
  # 0.5 Hz sine amplitude: comma uses 1.0, which peaks at 3.14 m/s3. Keep it under
  # our headroom rather than riding the envelope.
  sine_amp = _amp_for_freq(1.0, 0.5)

  ms = [
    Maneuver(f"step right {mph:.0f}mph", [Action([0.5], [1.0]), Action([-0.5], [1.5])],
             repeat=2, initial_speed=v),
    Maneuver(f"step left {mph:.0f}mph", [Action([-0.5], [1.0]), Action([0.5], [1.5])],
             repeat=2, initial_speed=v),
    Maneuver(f"sine 0.5Hz {mph:.0f}mph", [_sine_action(sine_amp, 2.0, 2.0), Action([0.0], [0.5])],
             repeat=2, initial_speed=v),
    Maneuver(f"chirp 0.2-2.0Hz {mph:.0f}mph", [_chirp_action(chirp_amp), Action([0.0], [1.0])],
             repeat=1, initial_speed=v),
    Maneuver(f"ramp 8s {mph:.0f}mph", [_triangle_action(1.0), Action([0.0], [1.0])],
             repeat=1, initial_speed=v),
  ]
  # small steps probe resolution near the friction floor, where the linear fit is worst
  for small in (0.1, 0.2):
    ms.append(Maneuver(f"small step {small:+.1f} {mph:.0f}mph",
                       [Action([small], [1.5]), Action([-small], [1.5])],
                       repeat=2, initial_speed=v))
  return ms


MANEUVERS: list[Maneuver] = [m for mph in SPEEDS_MPH for m in _for_speed(mph)]


def action_profile(action: Action) -> tuple[np.ndarray, np.ndarray]:
  """Sample an Action onto the control grid, as lateral_maneuversd._step does."""
  t = np.arange(0.0, action.time_bp[-1] + DT_MDL, DT_MDL)
  return t, np.interp(t, action.time_bp, action.accel_bp)


def envelope_violations(maneuver: Maneuver) -> list[str]:
  """Accel and within-action jerk against the planner envelope.

  Jerk is checked WITHIN each action only. The boundary between two actions is a
  commanded step (comma's own step maneuvers are exactly that), which is deliberate:
  the planner and clip_curvature limit what actually reaches the car. Checking across
  boundaries would flag comma's set too and measure nothing useful.
  """
  out = []
  for i, action in enumerate(maneuver.actions):
    t, a = action_profile(action)
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak > MAX_LATERAL_ACCEL:
      out.append(f"{maneuver.description} action {i}: |accel| {peak:.2f} > {MAX_LATERAL_ACCEL:.2f}")
    if len(a) > 1:
      jerk = float(np.max(np.abs(np.diff(a) / DT_MDL)))
      if jerk > MAX_LATERAL_JERK:
        out.append(f"{maneuver.description} action {i}: jerk {jerk:.2f} > {MAX_LATERAL_JERK:.2f}")
  return out
