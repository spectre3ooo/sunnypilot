#!/usr/bin/env python3
"""Tests for learner_bands.

The TLS-bias case is the reason the report carries a second torque slope at all: the
first full run showed torqued's own fit rising as speed fell, while the raw response
curve fell. Pinned so nobody "simplifies" the median slope away.
"""
import os
import tempfile
import unittest

import numpy as np

from opendbc.car.structs import car
from openpilot.tools.lat_metrics.learner_bands import (BandLagEstimator, balance_buckets, bootstrap, discover, fit_response_slope,
                                                       fit_steer_ratio, fit_steer_ratio_through, fit_torque, response_curve)


def _torque_points(slope, offset, noise, n=6000, seed=0):
  rng = np.random.default_rng(seed)
  steer = rng.uniform(-0.5, 0.5, n)
  return steer, slope * steer + offset + rng.normal(0, noise, n)


class TestTorqueFits(unittest.TestCase):
  def test_tls_recovers_clean_slope(self):
    steer, la = _torque_points(2.5, -0.3, 0.02)
    laf, off, _ = fit_torque(steer, la)
    self.assertAlmostEqual(laf, 2.5, delta=0.05)
    self.assertAlmostEqual(off, -0.3, delta=0.02)

  def test_tls_overestimates_noisy_low_slope_and_median_slope_does_not(self):
    """Low speed: slope ~1.3 with lat-accel spread comparable to the signal."""
    steer, la = _torque_points(1.3, -0.3, 0.4)
    laf, _, _ = fit_torque(steer, la)
    self.assertGreater(laf, 1.6, "TLS should rotate steep on noisy data; if this fails the " +
                                 "robust slope may no longer be needed")
    self.assertAlmostEqual(fit_response_slope(steer, la), 1.3, delta=0.1)

  def test_median_slope_needs_populated_buckets(self):
    steer = np.full(500, 0.05)
    with self.assertRaises(ZeroDivisionError):
      fit_response_slope(steer, steer * 2)

  def test_response_curve_blanks_sparse_buckets(self):
    steer, _ = _torque_points(2.0, 0.0, 0.01, n=4000)
    steer = steer[np.abs(steer) < 0.3]  # empty the two outer buckets
    curve = response_curve(steer, 2.0 * steer)
    self.assertIsNone(curve[0])
    self.assertIsNone(curve[-1])
    self.assertTrue(all(c is not None for c in curve[1:-1]))


class TestBalanceBuckets(unittest.TestCase):
  def test_caps_each_bucket_and_drops_out_of_range(self):
    rng = np.random.default_rng(0)
    steer = np.concatenate([np.full(5000, 0.05), np.full(10, 0.4), np.full(7, 0.9), np.full(3, -0.7)])
    idx = balance_buckets(steer, rng, per_bucket=100)
    kept = steer[idx]
    self.assertEqual(int(np.sum(kept == 0.05)), 100)
    self.assertEqual(int(np.sum(kept == 0.4)), 10)
    self.assertFalse(np.any(np.abs(kept) >= 0.5), "torqued ignores |steer| >= 0.5")


class TestSteerRatio(unittest.TestCase):
  def test_recovers_ratio_and_offset(self):
    rng = np.random.default_rng(1)
    x = rng.uniform(-0.05, 0.05, 5000)
    y = x / 15.6 + 2e-4 + rng.normal(0, 3e-4, 5000)
    sr, b = fit_steer_ratio(x, y)
    self.assertAlmostEqual(sr, 15.6, delta=0.2)
    self.assertAlmostEqual(b, 2e-4, delta=5e-5)
    self.assertAlmostEqual(fit_steer_ratio_through(x, y, b), 15.6, delta=0.2)


class TestBootstrap(unittest.TestCase):
  def test_resamples_segments_not_samples(self):
    # 10 segments, each internally constant: a per-sample CI would be ~0 wide
    seg = np.repeat(np.arange(10), 100)
    vals = np.repeat(np.arange(10, dtype=float), 100)
    lo, hi = bootstrap(seg, lambda i: float(np.mean(vals[i])), 300)
    self.assertGreater(hi - lo, 1.0)

  def test_too_few_segments_gives_nan(self):
    lo, hi = bootstrap(np.array(["a", "a", "b"]), lambda i: 1.0, 10)
    self.assertTrue(np.isnan(lo) and np.isnan(hi))


class TestDiscover(unittest.TestCase):
  def test_prefers_rlog_and_orders_segments(self):
    with tempfile.TemporaryDirectory() as d:
      for sub, name in [("q/0000001a--dbe019e4db--20", "qlog.zst"), ("r/0000001a--dbe019e4db--20", "rlog.zst"),
                        ("q/0000001a--dbe019e4db--3", "qlog.zst"), ("q/boot", "qlog.zst")]:
        os.makedirs(os.path.join(d, sub))
        open(os.path.join(d, sub, name), "w").close()
      routes = discover([os.path.join(d, "q"), os.path.join(d, "r")])
      self.assertEqual(list(routes), ["0000001a--dbe019e4db"])
      segs = routes["0000001a--dbe019e4db"]
      self.assertEqual([(s, k) for s, k, _ in segs], [(3, "qlog"), (20, "rlog")])


class TestBandLagEstimator(unittest.TestCase):
  def _okay_at(self, v):
    est = BandLagEstimator(car.CarParams(steerActuatorDelay=0.1), 0.05, v_max=10.0, min_vego=5.0, min_recovery_buffer_sec=0.0)
    est.calibrator.calib_valid = True
    est.lat_active, est.pose_valid, est.v_ego, est.t = True, True, v, 10.0
    est.yaw_rate, est.desired_curvature = 0.01, 0.01 / v
    est.update_points()
    return est.points.okay[-1]

  def test_band_limits(self):
    self.assertTrue(self._okay_at(7.0))
    self.assertFalse(self._okay_at(4.0), "below the band floor (lagd's own min_vego gate)")
    self.assertFalse(self._okay_at(12.0), "above the band ceiling")


if __name__ == "__main__":
  unittest.main()
