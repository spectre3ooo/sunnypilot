"""
Steering torque feedforward from steering angle and speed (LatAngleFeedforward).

Below ~30 mph the torque that holds and moves the wheel follows steering angle, not lateral acceleration: the
steering geometry's return force grows with angle, while lateral accel (speed^2 x curvature) stays small. The
lateral-accel torque model therefore under-asks at low speed, and the high-gain feedback that makes up the rest
drives the wheel back toward centre on corner exits. This model is fitted per car from its own logs:

  torque = sign(angle) * hold(v, |angle|) + damping(v) * rate + friction(v) * sign(rate) + gain(v) * lat_accel + bias

hold is a bilinear table over speed and |angle| nodes; the other terms interpolate linearly over the speed nodes.
Units: torque is normalized like actuators.torque (1.0 = STEER_MAX); angle and rate in degrees; lat_accel in m/s^2,
gravity adjusted (desired - roll * g). See docs/lat/angle-feedforward.md for the fit and its validation.
"""
import json
import os

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RATE_SCALE = 100.0  # deg/s; keeps the damping weights O(1)


def hat_weights(x: np.ndarray, nodes: np.ndarray) -> np.ndarray:
  """Linear interpolation weights of x over nodes, held flat outside them. Shape (len(x), len(nodes))."""
  x = np.clip(np.asarray(x, dtype=float), nodes[0], nodes[-1])
  i = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
  w = (x - nodes[i]) / (nodes[i + 1] - nodes[i])
  H = np.zeros((len(x), len(nodes)))
  H[np.arange(len(x)), i] = 1 - w
  H[np.arange(len(x)), i + 1] = w
  return H


def features(v, angle, rate, lat_accel, speed_nodes, angle_nodes) -> np.ndarray:
  """Design matrix shared by the offline fit and the runtime evaluator, so the two cannot disagree."""
  v, angle, rate, lat_accel = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (v, angle, rate, lat_accel))
  Hv, Ha = hat_weights(v, speed_nodes), hat_weights(np.abs(angle), angle_nodes)
  hold = (Hv[:, :, None] * Ha[:, None, :]).reshape(len(v), -1) * np.sign(angle)[:, None]
  return np.hstack([hold, Hv * (rate / RATE_SCALE)[:, None], Hv * np.sign(rate)[:, None], Hv * lat_accel[:, None],
                    np.ones((len(v), 1))])


class AngleFeedforwardModel:
  def __init__(self, path: str):
    with open(path) as f:
      p = json.load(f)
    self.name = p["name"]
    self.speed_nodes = np.array(p["speed_nodes"], dtype=float)
    self.angle_nodes = np.array(p["angle_nodes"], dtype=float)
    self.weights = np.array(p["weights"], dtype=float)
    n_expected = len(self.speed_nodes) * len(self.angle_nodes) + 3 * len(self.speed_nodes) + 1
    if len(self.weights) != n_expected:
      raise ValueError(f"{path}: {len(self.weights)} weights, expected {n_expected}")
    self.meta = p.get("meta", {})

  def evaluate(self, v: float, angle: float, rate: float, lat_accel: float) -> float:
    return float(features(v, angle, rate, lat_accel, self.speed_nodes, self.angle_nodes)[0] @ self.weights)

  def hold(self, v: float, angle: float) -> float:
    """The angle-only part: torque that holds the wheel still at this angle and speed (no lat accel, no motion)."""
    return self.evaluate(v, angle, 0.0, 0.0) - self.weights[-1]


def model_path(fingerprint: str) -> str:
  return os.path.join(MODEL_DIR, f"{fingerprint}.json")


def load_model(fingerprint: str) -> AngleFeedforwardModel | None:
  path = model_path(fingerprint)
  return AngleFeedforwardModel(path) if os.path.exists(path) else None
