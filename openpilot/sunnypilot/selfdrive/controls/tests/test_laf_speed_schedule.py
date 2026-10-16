from unittest.mock import PropertyMock, patch

import numpy as np

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params

from openpilot.cereal import log
from opendbc.car.structs import car
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.vehicle_model import VehicleModel
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.locationd.helpers import Pose
from openpilot.common.mock.generators import generate_deviceMotion
from openpilot.sunnypilot.selfdrive.car import interfaces as sunnypilot_interfaces
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.laf_speed_schedule import LAF_SPEED_SCHEDULE_BP, LAF_SPEED_SCHEDULE_V
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl

LAF = 2.8
# a constant desired lateral accel, small enough that the scaled torque stays off the steer_max clip even
# with V0's low speed P gain (65 at 2 m/s)
LAT_ACCEL = 0.02  # m/s^2


class TestLatAccelFactorSpeedSchedule(OpenpilotTestCase):
  def _controller(self, controller_cls, enabled):
    Params().put_bool("LatAccelFactorSpeedSchedule", enabled)
    CarInterface = interfaces[HYUNDAI.HYUNDAI_IONIQ_5]
    CP = CarInterface.get_non_essential_params(HYUNDAI.HYUNDAI_IONIQ_5)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, HYUNDAI.HYUNDAI_IONIQ_5)
    CI = CarInterface(CP, CP_SP)
    sunnypilot_interfaces.setup_interfaces(CI)
    controller = controller_cls(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
    # zero friction and offset, so torque is exactly lateral accel / latAccelFactor
    controller.update_torque_parameters(LAF, 0.0, 0.0)
    return controller, VehicleModel(CP)

  def _first_frame_torque(self, controller, VM, v_ego):
    CS = car.CarState.new_message()
    CS.vEgo = v_ego
    params = log.VehicleParameters.new_message()
    pose = Pose.from_device_motion(generate_deviceMotion().deviceMotion)
    torque, _, _ = controller.update(True, CS, VM, params, False, LAT_ACCEL / v_ego ** 2, pose, False, 0.2)
    return torque

  @parameterized.expand([(LatControlTorqueV0,), (LatControlTorque,)])
  def test_off_is_a_no_op(self, controller_cls):
    controller, _ = self._controller(controller_cls, enabled=False)
    for v in (2.0, 4.5, 8.9, 30.0):
      assert controller.get_scheduled_params(v) is controller.torque_params
      assert controller.laf_scale == 1.0

  @parameterized.expand([(LatControlTorqueV0,), (LatControlTorque,)])
  def test_scales_torque_at_low_speed_only(self, controller_cls):
    for v in (2.0, 4.5, 6.0, 8.9, 12.0, 15.2, 30.0):
      off, VM = self._controller(controller_cls, enabled=False)
      on, _ = self._controller(controller_cls, enabled=True)
      t_off = self._first_frame_torque(off, VM, v)
      t_on = self._first_frame_torque(on, VM, v)
      scale = float(np.interp(v, LAF_SPEED_SCHEDULE_BP, LAF_SPEED_SCHEDULE_V))
      np.testing.assert_allclose(on.laf_scale, scale, rtol=1e-6)  # vEgo is float32 in CarState
      assert 0.0 < abs(t_on) < 0.99, f"v_ego={v}: torque {t_on} is clipped, so the ratio below proves nothing"
      np.testing.assert_allclose(t_on, t_off / scale, rtol=1e-6, err_msg=f"v_ego={v}")
      # the learned value is never modified, so the scale cannot compound
      assert on.torque_params.latAccelFactor == np.float32(LAF)

  @parameterized.expand([(LatControlTorqueV0,), (LatControlTorque,)])
  def test_nnlc_disables_it(self, controller_cls):
    controller, _ = self._controller(controller_cls, enabled=True)
    with patch.object(NeuralNetworkLateralControl, "_nnlc_enabled", new_callable=PropertyMock, return_value=True):
      assert controller.get_laf_scale(4.5) == 1.0
    assert controller.get_laf_scale(4.5) == LAF_SPEED_SCHEDULE_V[0]
