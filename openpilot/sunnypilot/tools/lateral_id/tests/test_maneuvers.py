#!/usr/bin/env python3
"""T1.2 acceptance: no maneuver may command more than the planner envelope allows.

If a maneuver asks for more lateral accel or jerk than the planner will pass, the
command is clipped downstream and the excitation the fit actually receives is not the
excitation that was designed. The fit would then be built on a signal nobody recorded.
"""
import unittest

import numpy as np

from opendbc.car.lateral import MAX_LATERAL_ACCEL, MAX_LATERAL_JERK
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.tools.lateral_id.maneuvers import (
    MANEUVERS, SPEEDS_MPH, _amp_for_freq, _chirp_action, _triangle_action,
    action_profile, envelope_violations, CHIRP_F1, JERK_HEADROOM)


class TestManeuverEnvelope(unittest.TestCase):

  def test_no_maneuver_exceeds_the_envelope(self):
    """The acceptance criterion."""
    violations = [v for m in MANEUVERS for v in envelope_violations(m)]
    self.assertEqual(violations, [], "\n".join(violations))

  def test_every_speed_is_covered(self):
    for mph in SPEEDS_MPH:
      matching = [m for m in MANEUVERS if f"{mph:.0f}mph" in m.description]
      self.assertGreaterEqual(len(matching), 5, f"{mph} mph under-covered")

  def test_chirp_amplitude_is_jerk_limited_not_flat(self):
    """A flat-amplitude chirp to 2 Hz would break the jerk limit; this is the fix.

    2*pi*2.0*0.5 = 6.28 m/s3 against an envelope of ~3.6, so amplitude must taper.
    """
    flat_peak_jerk = 2 * np.pi * CHIRP_F1 * 0.5
    self.assertGreater(flat_peak_jerk, MAX_LATERAL_JERK,
                       "premise changed: a flat chirp would now be legal")
    t, a = action_profile(_chirp_action(0.5))
    jerk = np.max(np.abs(np.diff(a) / DT_MDL))
    self.assertLessEqual(jerk, MAX_LATERAL_JERK)
    # and the taper must actually bite at the top of the sweep
    self.assertLess(_amp_for_freq(0.5, CHIRP_F1), 0.5)
    # while leaving the low end at full requested amplitude
    self.assertAlmostEqual(_amp_for_freq(0.5, 0.2), 0.5)

  def test_low_speed_chirp_uses_the_larger_amplitude(self):
    lo = [m for m in MANEUVERS if m.description.startswith("chirp") and "10mph" in m.description]
    hi = [m for m in MANEUVERS if m.description.startswith("chirp") and "30mph" in m.description]
    self.assertTrue(lo and hi)
    lo_peak = np.max(np.abs(action_profile(lo[0].actions[0])[1]))
    hi_peak = np.max(np.abs(action_profile(hi[0].actions[0])[1]))
    self.assertGreater(lo_peak, hi_peak, "10 mph chirp should be driven harder")

  def test_triangle_crosses_zero_slowly(self):
    """Friction floor shows up in the slow zero crossing; a fast one would hide it."""
    t, a = action_profile(_triangle_action(1.0, 8.0))
    jerk = np.max(np.abs(np.diff(a) / DT_MDL))
    self.assertLess(jerk, 1.0, "ramp is too abrupt to expose the friction floor")
    self.assertAlmostEqual(float(a[0]), 0.0, places=6)
    self.assertAlmostEqual(float(a[-1]), 0.0, places=6)
    self.assertGreater(np.max(a), 0.9)
    self.assertLess(np.min(a), -0.9)

  def test_headroom_is_actually_left(self):
    """Sitting exactly on the limit would be clipped by rounding alone."""
    self.assertLess(JERK_HEADROOM, 1.0)
    worst = 0.0
    for m in MANEUVERS:
      for act in m.actions:
        _, a = action_profile(act)
        if len(a) > 1:
          worst = max(worst, float(np.max(np.abs(np.diff(a) / DT_MDL))))
    self.assertLessEqual(worst, JERK_HEADROOM * MAX_LATERAL_JERK + 1e-6)

  def test_descriptions_are_unique(self):
    """The fitter keys on maneuverName; duplicates would merge different runs."""
    names = [m.description for m in MANEUVERS]
    self.assertEqual(len(names), len(set(names)))

  def test_accel_is_reachable_as_curvature(self):
    """desiredCurvature = accel / v^2 must stay inside the planner curvature bound."""
    for m in MANEUVERS:
      v = m.initial_speed
      peak = max((float(np.max(np.abs(action_profile(a)[1]))) for a in m.actions), default=0.0)
      curv = peak / max(v, 0.1) ** 2
      max_curv = MAX_LATERAL_ACCEL / max(v, 0.1) ** 2
      self.assertLessEqual(curv, max_curv + 1e-9, m.description)


if __name__ == "__main__":
  unittest.main()
