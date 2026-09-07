#!/usr/bin/env python3
"""Regression tests for lat_metrics.

Both cases below are real defects that shipped once and produced plausible but
wrong numbers in docs/lat/baseline.md. They are cheap to reintroduce, so they
are pinned here.
"""
import unittest

import numpy as np

from openpilot.tools.lat_metrics.lat_metrics import _reversals, Seg, metrics, MPH


class TestReversalsDoNotSpanGaps(unittest.TestCase):
    """A reversal must not be paired across frames outside the band.

    Original defect: bend detection stopped at the band edge but the pair loop
    then joined the bend before the gap to the bend after it, counting the time
    spent outside the band as reversal duration.
    """

    def _trace(self):
        # 60 s at 100 Hz: + bend peaking at t=5 s, 40 s out of band, - bend at t=54 s
        n = 6000
        t = np.arange(n) / 100.0
        tq = np.zeros(n)
        tq[400:600] = 0.5     # + bend around t=5
        tq[5300:5500] = -0.5  # - bend around t=54
        mask = np.zeros(n, dtype=bool)
        mask[0:1000] = True   # in band early
        mask[5000:6000] = True  # in band late
        return t, tq, mask     # frames 1000..5000 are OUT of band

    def test_gap_spanning_pair_is_rejected(self):
        t, tq, mask = self._trace()
        durations, transits, pairs = _reversals(t, tq, mask)
        self.assertEqual(durations, [], "bends separated by out-of-band frames "
                                        "must not form a reversal")
        self.assertEqual(transits, [])
        self.assertEqual(pairs, [])

    def test_contiguous_pair_is_still_detected(self):
        # same bends, but the whole window is in band -> a real reversal
        t, tq, _ = self._trace()
        mask = np.ones(len(t), dtype=bool)
        durations, transits, pairs = _reversals(t, tq, mask)
        self.assertEqual(len(durations), 1)
        self.assertAlmostEqual(durations[0], 49.0, places=1)
        # transit excludes dwell in each bend: peaks are 49 s apart but the bends
        # themselves span 2 s each, so the actual crossing is much shorter
        self.assertEqual(len(transits), 1)
        self.assertLess(transits[0], durations[0])
        self.assertAlmostEqual(transits[0], 47.0, places=1)


class TestClearanceUsesBandMembership(unittest.TestCase):
    """Model frames must be selected by band membership, not by time range.

    Original defect: model frames were selected with (mt >= t[band].min()) &
    (mt <= t[band].max()), i.e. the convex hull of the band. Speed bands are not
    contiguous, so a low clearance recorded at 30 mph was reported as the worst
    clearance of the <15 mph band too.
    """

    def _seg(self):
        n = 6000
        s = Seg(name="synthetic")
        s.t = (np.arange(n) / 100.0).tolist()
        # creep <15 mph for the first and last 5 s, 30 mph in between
        v = np.full(n, 30 * MPH)
        v[:500] = 5 * MPH
        v[5500:] = 5 * MPH
        s.v = v.tolist()
        s.asked = [0.1] * n
        s.got = [0.1] * n
        s.lat_active = [True] * n
        s.pressed = [False] * n
        s.angle = [10.0] * n
        s.eps = [0.0] * n
        s.fault_tmp = [False] * n
        s.des_acc = [0.0] * n
        s.act_acc = [0.0] * n
        s.saturated = [False] * n
        # one model frame at t=30 s (i.e. at 30 mph) with a tight clearance
        s.mt = [30.0]
        s.mcurv = [0.01]
        s.clear_min = [-0.19]
        s.clear_x0 = [0.5]
        s.bound_edge = [0]
        return s

    def test_low_speed_band_excludes_a_30mph_clearance(self):
        rows, pooled = metrics(self._seg())
        lt15 = next(r for r in rows if r["band"] == "lt15")
        self.assertEqual(
            lt15["min_clearance_m"], "",
            "a clearance recorded at 30 mph must not appear in the <15 mph band")

    def test_correct_band_receives_it(self):
        rows, pooled = metrics(self._seg())
        b = next(r for r in rows if r["band"] == "25_40")
        self.assertEqual(b["min_clearance_m"], -0.19)


if __name__ == "__main__":
    unittest.main()
