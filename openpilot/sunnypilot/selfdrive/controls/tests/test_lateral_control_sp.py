#!/usr/bin/env python3
"""Tests for the LateralControlSP telemetry publish path.

process_replay does NOT cover this message: controlsd's ProcessConfig captures
only subs=["carControl", "controlsState"], so neither lateralControlSP nor
sunnypilot's own carControlSP is compared against reference logs. A replay pass
therefore says nothing about whether this message is correct. These tests cover
it directly.
"""
import unittest

from opendbc.car import structs
from openpilot.cereal import log
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


class FakePubMaster:
  def __init__(self):
    self.sent = {}

  def send(self, name, msg):
    self.sent[name] = msg


class Stub:
  """Minimal stand-in: ControlsExt.__init__ blocks on CarParamsSP from Params."""

  def __init__(self, arm=0, steer_type=None):
    CP = structs.CarParams()
    CP.steerControlType = steer_type if steer_type is not None else \
        structs.CarParams.SteerControlType.torque
    self.CP = CP
    self.lat_experiment_arm = arm
    self._git_hash = "deadbeef1234"


def _sm(can_valid=True, out_torque=0.2):
  cs = log.Event.new_message().init('carState')
  cs.canValid = can_valid
  co = log.Event.new_message().init('carOutput')
  co.actuatorsOutput.torque = out_torque
  return {'carState': cs, 'carOutput': co}


def _cc(torque=0.2, lat_active=True):
  cc = log.Event.new_message().init('carControl')
  cc.latActive = lat_active
  cc.actuators.torque = torque
  return cc


def _publish(stub, cc, sm):
  pm = FakePubMaster()
  ControlsExt.publish_lateral_ext(stub, cc, sm, pm)
  assert 'lateralControlSP' in pm.sent, "nothing published"
  return pm.sent['lateralControlSP'].lateralControlSP


class TestLateralControlSP(unittest.TestCase):

  def test_arm_and_githash_always_present(self):
    """Every frame must be attributable to an arm and a build."""
    lat = _publish(Stub(arm=2), _cc(), _sm())
    self.assertEqual(lat.armId, 2)
    self.assertEqual(lat.gitHash, "deadbeef1234")

  def test_rate_limited_when_output_lags_request(self):
    lat = _publish(Stub(), _cc(torque=0.5), _sm(out_torque=0.2))
    self.assertTrue(lat.limitedRate)
    self.assertFalse(lat.limitedCap)

  def test_cap_limited_when_output_pinned(self):
    # Values are stored as Float32, so a difference of exactly 1e-2 is ambiguous
    # at this magnitude (0.99 round-trips to 0.99000000953..). Use a clear margin
    # on both conditions: limited by >1e-2, and |output| at/above the 0.98 cap.
    lat = _publish(Stub(), _cc(torque=1.0), _sm(out_torque=0.985))
    self.assertTrue(lat.limitedCap)
    self.assertFalse(lat.limitedRate)

  def test_not_limited_when_request_is_honoured(self):
    lat = _publish(Stub(), _cc(torque=0.2), _sm(out_torque=0.2))
    self.assertFalse(lat.limitedCap)
    self.assertFalse(lat.limitedRate)

  def test_not_limited_when_lateral_inactive(self):
    """A large difference while lateral is off is not the limiter acting."""
    lat = _publish(Stub(), _cc(torque=1.0, lat_active=False), _sm(out_torque=0.0))
    self.assertFalse(lat.limitedCap)
    self.assertFalse(lat.limitedRate)

  def test_angle_cars_report_no_torque_limiting(self):
    """Torque limiting is meaningless on an angle-control car."""
    stub = Stub(steer_type=structs.CarParams.SteerControlType.angle)
    lat = _publish(stub, _cc(torque=1.0), _sm(out_torque=0.0))
    self.assertFalse(lat.limitedCap)
    self.assertFalse(lat.limitedRate)

  def test_validity_follows_can(self):
    pm = FakePubMaster()
    ControlsExt.publish_lateral_ext(Stub(), _cc(), _sm(can_valid=False), pm)
    self.assertFalse(pm.sent['lateralControlSP'].valid)

  def test_phase3_fields_default_until_implemented(self):
    lat = _publish(Stub(), _cc(), _sm())
    self.assertEqual(lat.corridorCorrection, 0.0)
    self.assertFalse(lat.corridorValid)
    self.assertFalse(lat.rightBoundIsEdge)


if __name__ == "__main__":
  unittest.main()
