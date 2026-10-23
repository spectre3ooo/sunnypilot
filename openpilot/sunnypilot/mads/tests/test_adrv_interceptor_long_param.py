from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP, HyundaiSafetyFlagsSP

from openpilot.sunnypilot.mads.helpers import set_car_specific_params


class FakeParams:
  def __init__(self, **bools):
    self.bools = bools

  def get_bool(self, key, *args, **kwargs):
    return self.bools.get(key, False)

  def get(self, key, *args, **kwargs):
    return None

  def put(self, *args, **kwargs):
    pass

  def put_bool(self, *args, **kwargs):
    pass


def _ioniq5(op_long):
  CP = structs.CarParams()
  CP.brand = "hyundai"
  CP.carFingerprint = "HYUNDAI_IONIQ_5"
  CP.flags = int(HyundaiFlags.CANFD | HyundaiFlags.CANFD_LKA_STEER_MSG | HyundaiFlags.EV)
  CP.openpilotLongitudinalControl = op_long
  return CP, structs.CarParamsSP()


class TestAdrvInterceptorLongParam:
  def test_off_by_default(self):
    CP, CP_SP = _ioniq5(op_long=True)
    set_car_specific_params(CP, CP_SP, FakeParams())
    assert not CP_SP.flags & HyundaiFlagsSP.ADRV_INTERCEPTOR_LONG
    assert not CP_SP.safetyParam & HyundaiSafetyFlagsSP.CANFD_ADRV_INTERCEPTOR_LONG

  def test_on_with_alpha_long(self):
    CP, CP_SP = _ioniq5(op_long=True)
    set_car_specific_params(CP, CP_SP, FakeParams(HyundaiAdrvInterceptorLong=True))
    assert CP_SP.flags & HyundaiFlagsSP.ADRV_INTERCEPTOR_LONG
    assert CP_SP.safetyParam & HyundaiSafetyFlagsSP.CANFD_ADRV_INTERCEPTOR_LONG

  def test_needs_op_long(self):
    # without alpha long the comma doesn't control speed, so there's nothing to intercept
    CP, CP_SP = _ioniq5(op_long=False)
    set_car_specific_params(CP, CP_SP, FakeParams(HyundaiAdrvInterceptorLong=True))
    assert not CP_SP.flags & HyundaiFlagsSP.ADRV_INTERCEPTOR_LONG
    assert not CP_SP.safetyParam & HyundaiSafetyFlagsSP.CANFD_ADRV_INTERCEPTOR_LONG
