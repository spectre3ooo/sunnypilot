#!/usr/bin/env python3
"""Tests for nnlc_gain: the gain inversion and the steady-state input layout."""
import json
import os
import tempfile
import unittest

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel
from openpilot.tools.lat_metrics.nnlc_gain import implied_gain, steady_state_torque

N_INPUTS = 18


def _linear_model(weights):
  """An NNLC-format model whose output is weights . inputs (one identity layer)."""
  # shapes as in the shipped models: inputs (n, 1), W (out, in), b (out, 1)
  params = {"input_size": N_INPUTS, "output_size": 1, "input_mean": [[0.0]] * N_INPUTS, "input_std": [[1.0]] * N_INPUTS,
            "layers": [{"dense_W": [list(map(float, weights))], "dense_b": [[0.0]], "activation": "identity"}]}
  f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
  json.dump(params, f)
  f.close()
  model = NNTorqueModel(f.name)
  os.unlink(f.name)
  return model


class TestImpliedGain(unittest.TestCase):
  def test_inverts_a_linear_model(self):
    # torque = 0.25 * lateral_accel  =>  latAccelFactor 4
    w = np.zeros(N_INPUTS)
    w[1] = 0.25
    gain, tq0 = implied_gain(_linear_model(w), 20.0, 1.0)
    self.assertAlmostEqual(gain, 4.0, places=4)
    self.assertAlmostEqual(tq0, 0.0, places=6)

  def test_steady_state_fills_history_and_roll(self):
    # weight only the past/future lat-accel slots (4..10) and the roll slots (11..17)
    w = np.zeros(N_INPUTS)
    w[4:11] = 1.0
    w[11:18] = 10.0
    tq = steady_state_torque(_linear_model(w), 10.0, 0.5, roll=0.01)
    self.assertAlmostEqual(tq, 7 * 0.5 + 7 * 10.0 * 0.01, places=4)


if __name__ == "__main__":
  unittest.main()
