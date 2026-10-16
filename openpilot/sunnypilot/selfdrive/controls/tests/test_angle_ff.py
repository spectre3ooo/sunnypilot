import math
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
from openpilot.sunnypilot.selfdrive.controls.lib.angle_ff.feedforward import BLEND_SPEEDS
from openpilot.sunnypilot.selfdrive.controls.lib.angle_ff.model import load_model
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_v0 import LatControlTorque as LatControlTorqueV0
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.nnlc import NeuralNetworkLateralControl

LAF = 2.8
LAT_ACCEL = 0.02  # m/s^2: small, so nothing reaches the steer_max clip even with V0's low speed P gain
CONTROLLERS = [(LatControlTorqueV0,), (LatControlTorque,)]


class TestLatAngleFeedforward(OpenpilotTestCase):
  def _controller(self, controller_cls, enabled, car_name=HYUNDAI.HYUNDAI_IONIQ_5):
    Params().put_bool("LatAngleFeedforward", enabled)
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
    CI = CarInterface(CP, CP_SP)
    sunnypilot_interfaces.setup_interfaces(CI)
    controller = controller_cls(CP.as_reader(), convert_to_capnp(CP_SP).as_reader(), CI, DT_CTRL)
    controller.update_torque_parameters(LAF, 0.0, 0.0)  # zero friction and offset: torque is exactly lat accel / LAF
    return controller, VehicleModel(CP)

  def _first_frame(self, controller, VM, v_ego, active=True, lat_accel=LAT_ACCEL):
    CS = car.CarState.new_message()
    CS.vEgo = v_ego
    params = log.VehicleParameters.new_message()
    pose = Pose.from_device_motion(generate_deviceMotion().deviceMotion)
    torque, _, _ = controller.update(active, CS, VM, params, False, lat_accel / v_ego ** 2, pose, False, 0.2)
    return -torque  # controllers return -output_torque; compare in output_torque convention

  def test_model_exists_for_the_ioniq5_only(self):
    assert load_model(HYUNDAI.HYUNDAI_IONIQ_5) is not None
    assert load_model(HYUNDAI.HYUNDAI_IONIQ_6) is None

  @parameterized.expand(CONTROLLERS)
  def test_low_speed_replaces_the_lat_accel_feedforward(self, controller_cls):
    model = load_model(HYUNDAI.HYUNDAI_IONIQ_5)
    for v in (2.0, 4.5, 8.9, 12.0):
      off, VM = self._controller(controller_cls, enabled=False)
      on, _ = self._controller(controller_cls, enabled=True)
      t_off, t_on = self._first_frame(off, VM, v), self._first_frame(on, VM, v)
      assert on.angle_ff_weight == 1.0, v
      angle_des = math.degrees(VM.get_steer_from_curvature(-LAT_ACCEL / v ** 2, v, 0.0))
      angle_torque = -model.evaluate(v, angle_des, 0.0, LAT_ACCEL)  # first frame: no desired-angle rate yet
      assert abs(t_on) < 0.99 and abs(t_off) < 0.99, "clipped: the comparison below would prove nothing"
      # the lateral-accel feedforward (LAT_ACCEL / LAF) is removed and the angle torque added
      np.testing.assert_allclose(t_on - t_off, angle_torque - LAT_ACCEL / LAF, atol=1e-6, err_msg=f"v_ego={v}")
      np.testing.assert_allclose(on.angle_ff_torque, angle_torque, atol=1e-9)

  @parameterized.expand(CONTROLLERS)
  def test_pushes_the_same_way_as_the_lat_accel_feedforward(self, controller_cls):
    on, VM = self._controller(controller_cls, enabled=True)
    for la in (0.5, -0.5):
      on._reset_angle_ff()
      self._first_frame(on, VM, 5.0, lat_accel=la)
      assert np.sign(on.angle_ff_torque) == np.sign(la), f"lat accel {la}: angle torque {on.angle_ff_torque}"

  @parameterized.expand(CONTROLLERS)
  def test_highway_is_unchanged(self, controller_cls):
    v = BLEND_SPEEDS[1] + 1.0
    off, VM = self._controller(controller_cls, enabled=False)
    on, _ = self._controller(controller_cls, enabled=True)
    assert self._first_frame(on, VM, v) == self._first_frame(off, VM, v)
    assert on.angle_ff_weight == 0.0

  @parameterized.expand(CONTROLLERS)
  def test_off_without_model_with_nnlc_or_inactive(self, controller_cls):
    no_model, VM = self._controller(controller_cls, enabled=True, car_name=HYUNDAI.HYUNDAI_IONIQ_6)
    self._first_frame(no_model, VM, 4.0)
    assert no_model.angle_ff_weight == 0.0

    on, VM = self._controller(controller_cls, enabled=True)
    # NNLC replaces the feedforward with its own model (running it here would need a modelV2), so stand down
    with patch.object(NeuralNetworkLateralControl, "_nnlc_enabled", new_callable=PropertyMock, return_value=True):
      assert not on.angle_ff_active()
      assert on.update_angle_ff(True, car.CarState.new_message(), VM, log.VehicleParameters.new_message(), 0.01, 0.1) == (0.0, 0.0)

    self._first_frame(on, VM, 4.0)
    assert on.angle_ff_weight == 1.0
    self._first_frame(on, VM, 4.0, active=False)
    assert on.angle_ff_weight == 0.0 and on.angle_ff_torque == 0.0 and on._angle_des_last is None
