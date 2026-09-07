#!/usr/bin/env python3
"""Driver-input abort policy for the maneuver runner.

The car's stock cruise will not set below 20 mph, so the 10 and 15 mph maneuvers can
only be flown with a foot on the accelerator. comma's runner aborts on any driver
input including gas, which would make those two speeds unreachable by any means:
cruise cannot command them and the pedal cancels the run.

Steering must always abort. Gas is longitudinal and does not fight a lateral maneuver,
and the speed requirement is enforced independently by MAX_SPEED_DEV.
"""
import unittest
from dataclasses import dataclass

from openpilot.tools.lateral_maneuvers.lateral_maneuversd import should_abort


@dataclass
class FakeCS:
  steeringPressed: bool = False
  gasPressed: bool = False


class TestAbortPolicy(unittest.TestCase):

  def test_steering_always_aborts(self):
    for allow_gas in (True, False):
      self.assertEqual(should_abort(FakeCS(steeringPressed=True), allow_gas), "steering pressed")

  def test_gas_aborts_by_default(self):
    """comma's behaviour is unchanged when the ID set is not selected."""
    self.assertEqual(should_abort(FakeCS(gasPressed=True), allow_gas=False), "gas pressed")

  def test_gas_permitted_for_the_id_set(self):
    """Otherwise 10 and 15 mph cannot be flown at all on this car."""
    self.assertEqual(should_abort(FakeCS(gasPressed=True), allow_gas=True), "")

  def test_steering_still_wins_over_permitted_gas(self):
    cs = FakeCS(steeringPressed=True, gasPressed=True)
    self.assertEqual(should_abort(cs, allow_gas=True), "steering pressed")

  def test_no_input_continues(self):
    for allow_gas in (True, False):
      self.assertEqual(should_abort(FakeCS(), allow_gas), "")


if __name__ == "__main__":
  unittest.main()
