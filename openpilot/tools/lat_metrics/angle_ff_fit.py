#!/usr/bin/env python3
"""Fit the LatAngleFeedforward model (sunnypilot/selfdrive/controls/lib/angle_ff) for one car from its own logs.

Input is a steer_dataset.py directory. The fit learns what torque the car needs at each speed, steering angle,
steering rate and lateral accel, from two kinds of frames that reveal the car rather than the controller:
  - steady engaged holds: the command that keeps the wheel still is what the car needs there
  - manual driving: the column torque that actually turned the wheel, hands (driver sensor) plus EPS assist,
    converted to command units, minus the torque the EPS adds by itself when it is the one being commanded

Calibration, all from the same logs:
  counts -> Nm  steady engaged holds, EPS output torque vs command
  driver Nm     scale of the raw driver sensor, from matching engaged and manual holds of the same state
  EPS-own       what the EPS adds on top of an engaged command (active return, damping), as a residual model

Sparse grid cells borrow from their neighbours through a second-difference smoothness penalty; its strength is
chosen by error on held-out routes. Usage:
  angle_ff_fit.py <dataset dir> --fingerprint HYUNDAI_IONIQ_5 [--out DIR]
"""
import argparse
import datetime
import glob
import json
import os
import zlib

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.sunnypilot.selfdrive.controls.lib.angle_ff.model import MODEL_DIR, features
from openpilot.tools.lat_metrics.steer_dataset import COLS

C = {c: i for i, c in enumerate(COLS)}
MPH = 0.44704
STEER_MAX = 270  # counts; the model is stored normalized to this, like actuators.torque
SPEED_NODES = np.array([0.5, 2.2, 4.5, 6.7, 8.9, 11.2, 15.6, 22.0, 30.0, 38.0])        # m/s
ANGLE_NODES = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 90.0, 150.0, 250.0, 400.0, 600.0])  # |deg|
LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
BANDS = [(0.5, 5 * MPH, "<5"), (5 * MPH, 15 * MPH, "5-15"), (15 * MPH, 25 * MPH, "15-25"), (25 * MPH, 45 * MPH, "25-45"), (45 * MPH, 60, ">45")]


def load(dataset_dir):
  segs = []
  for f in sorted(glob.glob(os.path.join(dataset_dir, "*.npz"))):
    x = np.load(f)["data"].astype(np.float64)
    if len(x) < 20:
      continue
    route = os.path.basename(f)[:-4].rsplit("--", 1)[0]
    rate = np.gradient(x[:, C["angle"]], x[:, C["t"]])
    segs.append(np.column_stack([x, rate, np.full(len(x), zlib.crc32(route.encode()) % 2)]))
  X = np.vstack(segs)
  d = {c: X[:, i] for c, i in C.items()}
  d["rate_calc"], d["half"] = X[:, -2], X[:, -1]
  d["routes"] = len({os.path.basename(f)[:-4].rsplit("--", 1)[0] for f in glob.glob(os.path.join(dataset_dir, "*.npz"))})
  return d


def smoothness(n_v, n_a):
  """Second differences of the hold grid along speed and |angle|, and of each per-speed vector along speed."""
  n_hold = n_v * n_a
  n = n_hold + 3 * n_v + 1
  rows = []
  def second(idx):
    for a, b, c in zip(idx, idx[1:], idx[2:], strict=False):
      r = np.zeros(n)
      r[a], r[b], r[c] = 1, -2, 1
      rows.append(r)
  for i in range(n_v):
    second([i * n_a + j for j in range(n_a)])
  for j in range(n_a):
    second([i * n_a + j for i in range(n_v)])
  for k in range(3):
    second([n_hold + k * n_v + i for i in range(n_v)])
  return np.array(rows)


def solve(A, y, D, lam):
  return np.linalg.solve(A.T @ A + lam * D.T @ D + 1e-6 * np.eye(A.shape[1]), A.T @ y)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("dataset")
  ap.add_argument("--fingerprint", required=True)
  ap.add_argument("--out", default=MODEL_DIR)
  args = ap.parse_args()

  d = load(args.dataset)
  v, ang, rate, drv, eps, cmd = d["v"], d["angle"], d["rate_calc"], d["drv"], d["eps"], d["cmd"]
  s = np.sign(ang)
  nnlc = np.abs(np.nan_to_num(d["nnlc_err"])) > 1e-4
  moving = v > 0.5
  eng = (d["lat"] > 0.5) & (d["pressed"] < 0.5) & (np.abs(drv) < 80) & moving & ~nnlc
  man = (d["lat"] < 0.5) & moving
  steady = np.abs(rate) < 3
  ok_la = ~np.isnan(d["la_meas"])
  la = np.nan_to_num(d["la_meas"])
  if np.corrcoef(la[ok_la & eng], d["act_la"][ok_la & eng])[0, 1] < 0:
    la = -la  # yaw sign convention of the pose: align with the controller's lateral accel
  la_adj = la - ACCELERATION_DUE_TO_GRAVITY * d["roll"]
  print(f"frames {len(v)} ({len(v) / 20 / 3600:.1f} h, {d['routes']} routes): engaged hands-off {eng.sum() / 1200:.0f} min, manual {man.sum() / 1200:.0f} min")

  # counts -> Nm
  k = eng & steady & (np.abs(cmd) > 10)
  kc, bc = np.polyfit(cmd[k], eps[k], 1)
  # driver sensor scale: engaged and manual holds of the same state need the same column torque
  cells = []
  for lo, hi, _ in BANDS:
    for alo, ahi in ((2, 15), (15, 45), (45, 90), (90, 720)):
      sel = (v >= lo) & (v < hi) & (np.abs(ang) >= alo) & (np.abs(ang) < ahi) & steady
      ke, km = eng & sel, man & sel
      if ke.sum() > 100 and km.sum() > 100:
        cells.append((np.median(eps[ke] * s[ke]), np.median(eps[km] * s[km]), np.median(drv[km] * s[km])))
  cells = np.array(cells)
  cdrv = float(np.linalg.lstsq(cells[:, 2:3], cells[:, 0] - cells[:, 1], rcond=None)[0][0])
  print(f"calibration: {kc:.4f} Nm/count ({STEER_MAX} counts = {kc * STEER_MAX:.1f} Nm), driver sensor {cdrv:.4f} Nm/unit from {len(cells)} cells")

  D = smoothness(len(SPEED_NODES), len(ANGLE_NODES))
  def F(idx):
    return features(v[idx], ang[idx], rate[idx], la_adj[idx], SPEED_NODES, ANGLE_NODES)
  # EPS-own torque on top of an engaged command (Nm), so it is not counted twice
  ie = np.flatnonzero(eng & ok_la)
  w_own = solve(F(ie), eps[ie] - kc * cmd[ie] - bc, D, 10.0)
  im = np.flatnonzero(man & ok_la)
  own_m = F(im) @ w_own
  y = np.full(len(v), np.nan)
  y[im] = (eps[im] + cdrv * drv[im] - own_m - bc) / kc / STEER_MAX
  ih = np.flatnonzero(eng & steady & ok_la)
  y[ih] = cmd[ih] / STEER_MAX
  use = ~np.isnan(y) & (np.abs(y) < 2.6)
  unwind = (np.sign(rate) == -s) & (np.abs(rate) > 5) & (np.abs(ang) > 2)

  def evaluate(w, idx):
    p, yt = F(idx) @ w * STEER_MAX, y[idx] * STEER_MAX
    r = p - yt
    out = {"rmse": float(np.sqrt(np.mean(r ** 2))), "r2": float(1 - np.sum(r ** 2) / np.sum((yt - yt.mean()) ** 2))}
    for lo, hi, lab in BANDS:
      b = (v[idx] >= lo) & (v[idx] < hi)
      if b.sum() > 200:
        rb = r[b]
        out[f"rmse_{lab}"] = float(np.sqrt(np.mean(rb ** 2)))
        out[f"r2_{lab}"] = float(1 - np.sum(rb ** 2) / np.sum((yt[b] - yt[b].mean()) ** 2))
    u = unwind[idx]
    out["reverse_on_unwind_model"] = float(np.mean(p[u] * s[idx][u] < -10))
    out["reverse_on_unwind_target"] = float(np.mean(yt[u] * s[idx][u] < -10))
    return out

  # held-out routes: fit on one half, score on the other half's manual frames
  tr = np.flatnonzero(use & (d["half"] == 0))
  te = np.flatnonzero(use & man & (d["half"] == 1))
  A_tr = F(tr)
  cv = {lam: evaluate(solve(A_tr, y[tr], D, lam), te) for lam in LAMBDAS}
  print("held-out manual frames, counts:")
  for lam, m in cv.items():
    print(f"  lambda {lam:6.1f}: RMSE {m['rmse']:5.1f}  R2 {m['r2']:+.3f}  " +
          "  ".join(f"{b} {m.get('rmse_' + b, float('nan')):5.1f}" for *_, b in BANDS) +
          f"  reverse on unwind {m['reverse_on_unwind_model']:.0%} (target {m['reverse_on_unwind_target']:.0%})")
  lam = min(cv, key=lambda k_: cv[k_]["rmse"])

  ia = np.flatnonzero(use)
  w = solve(F(ia), y[ia], D, lam)
  out = {
    "name": args.fingerprint,
    "speed_nodes": SPEED_NODES.tolist(),
    "angle_nodes": ANGLE_NODES.tolist(),
    "weights": w.tolist(),
    "meta": {
      "fitted": datetime.date.today().isoformat(),
      "tool": "openpilot/tools/lat_metrics/angle_ff_fit.py",
      "frames_used": int(len(ia)), "manual_frames": int(np.isin(ia, im).sum()), "steady_engaged_frames": int(np.isin(ia, ih).sum()),
      "routes": int(d["routes"]),
      "nm_per_count": float(kc), "driver_nm_per_unit": cdrv, "lambda": lam,
      "heldout": cv[lam],
    },
  }
  os.makedirs(args.out, exist_ok=True)
  path = os.path.join(args.out, f"{args.fingerprint}.json")
  with open(path, "w") as f:
    json.dump(out, f, indent=1)
  print(f"lambda {lam} chosen; wrote {path}")


if __name__ == "__main__":
  main()
