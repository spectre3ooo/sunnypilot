import unittest

from openpilot.cereal import custom
from opendbc.car import structs
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.sunnypilot.selfdrive.car.car_specific import CarSpecificEventsSP

EventNameSP = custom.OnroadEventSP.EventName


def cs_sp(state="none", missing=False):
  msg = custom.CarStateSP.new_message()
  msg.adrvInterceptorState = state
  msg.adrvMissing = missing
  return msg


class TestAdrvFailsafeEvents(unittest.TestCase):
  def setUp(self):
    self.ev = CarSpecificEventsSP(structs.CarParams(brand="hyundai"), structs.CarParamsSP())

  def names(self, msg):
    return self.ev.update_adrv(msg).names

  def test_capnp_enum_reads_back_as_string(self):
    self.assertEqual("trippedWatchdog", cs_sp("trippedWatchdog").adrvInterceptorState)

  def test_convert_to_capnp_accepts_opendbc_fields(self):
    # card converts the opendbc dataclass with convert_to_capnp; the StrEnum must map onto the capnp enum
    for st in structs.CarStateSP.AdrvInterceptorState:
      msg = convert_to_capnp(structs.CarStateSP(adrvInterceptorState=st, adrvMissing=True))
      self.assertEqual(str(st), msg.adrvInterceptorState)
      self.assertTrue(msg.adrvMissing)
    self.assertEqual("none", convert_to_capnp(structs.CarStateSP()).adrvInterceptorState)
    self.assertEqual("trippedTraffic", convert_to_capnp(structs.CarStateSP(adrvInterceptorState="trippedTraffic")).adrvInterceptorState)

  def test_quiet_normally(self):
    for st in ("none", "starting", "engaged", "canUpdate", "ignitionOff"):
      self.assertEqual([], self.names(cs_sp(st)))

  def test_tripped_states_raise_and_latch(self):
    for st in ("trippedTraffic", "trippedWatchdog"):
      self.setUp()
      self.assertIn(EventNameSP.adrvInterceptorTripped, self.names(cs_sp(st)))
      self.assertIn(EventNameSP.adrvInterceptorTripped, self.names(cs_sp("none")))  # latched for the drive
      self.assertIn(EventNameSP.adrvInterceptorTripped, self.names(cs_sp("engaged")))

  def test_missing(self):
    self.assertIn(EventNameSP.adrvMissing, self.names(cs_sp(missing=True)))
    self.assertNotIn(EventNameSP.adrvMissing, self.names(cs_sp(missing=False)))

  def test_other_brands_ignore_it(self):
    ev = CarSpecificEventsSP(structs.CarParams(brand="toyota"), structs.CarParamsSP())
    self.assertEqual([], ev.update_adrv(cs_sp("trippedTraffic", True)).names)

  def test_alerts_are_non_blocking_warnings(self):
    from openpilot.sunnypilot.selfdrive.selfdrived.events_base import ET, Priority
    from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP
    texts = {
      EventNameSP.adrvInterceptorTripped: ("ADRV Interceptor Tripped", "Pass-through until restart"),
      EventNameSP.adrvMissing: ("ADAS Unit Not Responding", "Unplug Interceptor"),
    }
    for name, (text1, text2) in texts.items():
      self.assertEqual({ET.PERMANENT}, set(EVENTS_SP[name]))
      alert = EVENTS_SP[name][ET.PERMANENT]
      self.assertEqual((text1, text2), (alert.alert_text_1, alert.alert_text_2))
      self.assertEqual(Priority.MID, alert.priority)


if __name__ == "__main__":
  unittest.main()
