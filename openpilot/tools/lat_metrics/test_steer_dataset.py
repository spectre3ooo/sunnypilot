#!/usr/bin/env python3
"""Tests for steer_dataset: decoding the car's body IMU (IMU_01_10ms)."""
import unittest

import numpy as np

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from openpilot.tools.lat_metrics.steer_dataset import COLS, G, IMU_ADDR, imu_values


def _vl(yaw=0.0, lat_g=0.0, long_g=0.0, yaw_sta=0, lat_sta=0, long_sta=0):
  return {"IMU_YawRtVal": yaw, "IMU_LatAccelVal": lat_g, "IMU_LongAccelVal": long_g,
          "IMU_YawSigSta": yaw_sta, "IMU_LatAccelSigSta": lat_sta, "IMU_LongAccelSigSta": long_sta}


class TestImuValues(unittest.TestCase):
  def test_units(self):
    yaw, lat, lon = imu_values(_vl(yaw=12.5, lat_g=0.25, long_g=-0.1))
    self.assertAlmostEqual(yaw, 12.5)
    self.assertAlmostEqual(lat, 0.25 * G)
    self.assertAlmostEqual(lon, -0.1 * G)

  def test_failed_or_initialising_signal_is_nan(self):
    # each status bit (failure, initialising) blanks only its own signal
    yaw, lat, lon = imu_values(_vl(yaw=3.0, lat_g=0.1, long_g=0.1, yaw_sta=1, long_sta=2))
    self.assertTrue(np.isnan(yaw))
    self.assertAlmostEqual(lat, 0.1 * G)
    self.assertTrue(np.isnan(lon))

  def test_round_trip_through_the_dbc(self):
    packer = CANPacker("hyundai_canfd_generated")
    parser = CANParser("hyundai_canfd_generated", [("IMU_01_10ms", 100)], 1)
    addr, dat, bus = packer.make_can_msg("IMU_01_10ms", 1, {"IMU_YawRtVal": -20.0, "IMU_LatAccelVal": 0.3, "IMU_LongAccelVal": 0.05})
    self.assertEqual(addr, IMU_ADDR)
    parser.update([0, [(addr, dat, bus)]])
    yaw, lat, lon = imu_values(parser.vl["IMU_01_10ms"])
    self.assertAlmostEqual(yaw, -20.0, places=2)
    self.assertAlmostEqual(lat, 0.3 * G, places=2)
    self.assertAlmostEqual(lon, 0.05 * G, places=2)

  def test_columns_appended_at_the_end(self):
    # older datasets are padded column-wise by angle_ff_fit.load, so new columns must only ever be appended
    self.assertEqual(COLS[-4:], ["aff_w", "car_yaw", "car_lat", "car_long"])


if __name__ == "__main__":
  unittest.main()
