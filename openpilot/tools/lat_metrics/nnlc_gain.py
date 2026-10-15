#!/usr/bin/env python3
"""Steady-state torque gain implied by an NNLC model, per speed band (T-1).

learner_bands.py measured that the Ioniq 5's torque -> lateral-accel response roughly
halves below 25 mph, while the stock torque controller uses one latAccelFactor at every
speed. NNLC takes vEgo as an input, so it could already model that. This evaluates the
model's feedforward in steady state (constant lateral accel over the past and future
inputs, zero jerk and roll) across a lat-accel sweep at each band's speed. It fits the
implied latAccelFactor (lat accel per unit torque) and sets it beside the measured
median slope.

With --logs it also checks the model against driving. In steady-state frames it takes
the torque actually commanded for a desired lateral accel. The car settles on whatever
torque the plant needs, so the slope of gravity-adjusted lat accel against commanded torque
is a direct, torqued-free estimate of the plant's gain. The same slope against the NNLC
model's feedforward for the same frames is the gain the model believes in. NNLC state isn't
logged on official builds, so each segment is classified from torqueState.error: the
stock controller logs the lat-accel error (desired − actual) there, and NNLC overwrites
it with a torque-space error.

Usage:
  nnlc_gain.py [--model HYUNDAI_IONIQ_5] [--bands learner_bands.json] [--max-lat-acc 1.0]
               [--logs <rlog/qlog files or dirs...>]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import TORQUE_NN_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel
from openpilot.tools.lat_metrics.lat_metrics import BANDS, MPH

# Representative speed per learner_bands band: the band's middle, 55 mph for the open band.
BAND_SPEEDS = {"lt15": 10 * MPH, "15_25": 20 * MPH, "25_40": 32.5 * MPH, "gt40": 55 * MPH}
N_PAST, N_FUTURE = 3, 4  # lateral_accel_m03..m01, p03..p15 (and the same for roll)


STEADY_MAX_JERK = 0.3       # |desiredLateralJerk|, m/s^3
STEADY_MAX_LAT_ACC = 2.5    # |desiredLateralAccel|, m/s^2
STEADY_MAX_ERROR = 0.2      # |desired - actual| lat accel, m/s^2: only frames that are tracking
NNLC_ERROR_MISMATCH = 1e-4   # median |error - (desired - actual)|: stock ~1e-8 (float32), NNLC >= 1e-2


def steady_state_torque(model, v, lat_acc, roll=0.0):
  """NNLC feedforward for a car holding lat_acc: every past and future lateral accel
  equals the current one, jerk is zero and roll is constant."""
  n = N_PAST + N_FUTURE
  return model.evaluate([v, lat_acc, 0.0, roll] + [lat_acc] * n + [roll] * n)


def implied_gain(model, v, max_lat_acc, n=41):
  """(latAccelFactor, torque at zero) from a line through lat accel vs torque, the same
  orientation learner_bands' median slope is measured in."""
  la = np.linspace(-max_lat_acc, max_lat_acc, n)
  tq = np.array([steady_state_torque(model, v, a) for a in la])
  slope, _ = np.polyfit(tq, la, 1)
  return float(slope), float(steady_state_torque(model, v, 0.0))


def segment_frames(path):
  """-> (segment, is_nnlc, frames[v, desired_la, commanded steer, roll]) for steady frames."""
  from openpilot.tools.lib.logreader import LogReader
  rows, mismatch = [], []
  roll = 0.0
  cs = None
  try:
    msgs = list(LogReader(path))
  except Exception as e:  # a corrupt segment should not sink the run
    print(f"skip {path}: {e}", file=sys.stderr)
    return os.path.basename(os.path.dirname(path)), None, np.zeros((0, 4))
  for m in msgs:
    which = m.which()
    if which == "vehicleParameters":
      roll = m.vehicleParameters.roll
    elif which == "carState":
      cs = m.carState
    elif which == "controlsState" and cs is not None:
      lcs = m.controlsState.lateralControlState
      if lcs.which() != "torqueState":
        continue
      ts = lcs.torqueState
      if not ts.active or cs.steeringPressed or ts.saturated:
        continue
      mismatch.append(abs(ts.error - (ts.desiredLateralAccel - ts.actualLateralAccel)))
      if abs(ts.desiredLateralJerk) < STEADY_MAX_JERK and abs(ts.desiredLateralAccel) <= STEADY_MAX_LAT_ACC and \
         abs(ts.desiredLateralAccel - ts.actualLateralAccel) < STEADY_MAX_ERROR:
        rows.append((cs.vEgo, ts.desiredLateralAccel, -ts.output, roll))  # output is logged negated (left-positive torque)
  is_nnlc = bool(np.median(mismatch) > NNLC_ERROR_MISMATCH) if len(mismatch) > 100 else None
  return os.path.basename(os.path.dirname(path)), is_nnlc, np.array(rows).reshape(-1, 4)


def log_check(model, paths, jobs):
  from openpilot.tools.lat_metrics.learner_bands import bootstrap, discover
  files = [f for segs in discover(paths).values() for _, _, f in segs]
  with ProcessPoolExecutor(jobs) as ex:
    segs = list(ex.map(segment_frames, files, chunksize=4))
  n_on = sum(1 for _, k, _ in segs if k)
  n_off = sum(1 for _, k, _ in segs if k is False)
  print(f"\n## Check against driving\n\n{len(files)} segments: {n_on} classified NNLC on, {n_off} off, " +
        f"{len(segs) - n_on - n_off} too little engaged driving to classify. Steady, tracking frames: " +
        f"\\|desired jerk\\| < {STEADY_MAX_JERK:g}, \\|desired lat accel\\| ≤ {STEADY_MAX_LAT_ACC:g}, " +
        f"\\|desired − actual\\| < {STEADY_MAX_ERROR:g}, engaged, no driver torque, not saturated.\n")
  print("*plant gain* = slope of gravity-adjusted lat accel (desired − g·roll) against the torque actually commanded: " +
        "what the car does. *NNLC gain* = the same slope against the NNLC model's steady-state feedforward for the " +
        "same frames: what the model believes. Ratio below 1 = the model expects more lat accel per torque than the car " +
        "gives, so its feedforward under-asks and feedback has to make up the rest.\n")
  print("| mode | band | frames | segs | plant gain | 90 % CI | NNLC gain | plant / NNLC | 90 % CI |")
  print("|---|---|---|---|---|---|---|---|---|")
  for mode, flag in (("NNLC on", True), ("NNLC off", False)):
    seg_ids, data = [], []
    for name, k, rows in segs:
      if k is flag and len(rows):
        seg_ids += [name] * len(rows)
        data.append(rows)
    if not data:
      print(f"| {mode} | all | 0 | 0 | | | | | |")
      continue
    seg_ids, data = np.array(seg_ids), np.vstack(data)
    la_adj = data[:, 1] - ACCELERATION_DUE_TO_GRAVITY * data[:, 3]
    ff = np.array([steady_state_torque(model, v, la, r) for v, la, _, r in data])
    for band, lo, hi in BANDS:
      idx = np.flatnonzero((data[:, 0] >= lo) & (data[:, 0] < hi))
      n_seg = len(set(seg_ids[idx]))
      if len(idx) < 100 or n_seg < 3:
        print(f"| {mode} | {band} | {len(idx)} | {n_seg} | too few | | | | |")
        continue

      def slope(xs, i, idx=idx, y=la_adj):
        return float(np.polyfit(xs[idx][i], y[idx][i], 1)[0])
      all_i = np.arange(len(idx))
      plant, nn = slope(data[:, 2], all_i), slope(ff, all_i)
      cmd = data[:, 2]
      p_ci = bootstrap(seg_ids[idx], lambda i, cmd=cmd: slope(cmd, i), 300)
      r_ci = bootstrap(seg_ids[idx], lambda i, cmd=cmd, ff=ff: slope(ff, i) / slope(cmd, i), 300)
      print(f"| {mode} | {band} | {len(idx)} | {n_seg} | **{plant:.2f}** | {p_ci[0]:.2f}–{p_ci[1]:.2f} | {nn:.2f} | " +
            f"**{plant / nn:.2f}** | {1 / r_ci[1]:.2f}–{1 / r_ci[0]:.2f} |")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--model", default="HYUNDAI_IONIQ_5", help="model name under neural_network_lateral_control/")
  ap.add_argument("--bands", help="learner_bands.json to compare against")
  ap.add_argument("--max-lat-acc", type=float, default=1.0, help="sweep ±this, m/s² (learner_bands' stock gate is 1)")
  ap.add_argument("--logs", nargs="+", help="also check against driving in these rlog/qlog files or dirs")
  ap.add_argument("--jobs", type=int, default=os.cpu_count())
  args = ap.parse_args()

  model = NNTorqueModel(os.path.join(TORQUE_NN_MODEL_PATH, f"{args.model}.json"))
  measured = {}
  if args.bands:
    with open(args.bands) as f:
      torque = json.load(f)["torque"]
    measured = {k.split("@")[0]: v for k, v in torque.items() if k.endswith(f"@{args.max_lat_acc:g}") and "medianSlope" in v}

  print(f"## NNLC implied gain — {args.model}\n")
  print(f"Steady-state sweep over ±{args.max_lat_acc:g} m/s², flat road. latAccelFactor = lat accel per unit torque.\n")
  print("| band | speed mph | NNLC latAccelFactor | measured median slope | 90 % CI | NNLC / measured |")
  print("|---|---|---|---|---|---|")
  for band, v in BAND_SPEEDS.items():
    gain, _ = implied_gain(model, v, args.max_lat_acc)
    m = measured.get(band)
    if m:
      lo, hi = m["medianSlope_ci"]
      print(f"| {band} | {v / MPH:.1f} | **{gain:.3f}** | {m['medianSlope']:.3f} | {lo:.3f}–{hi:.3f} | {gain / m['medianSlope']:.2f} |")
    else:
      print(f"| {band} | {v / MPH:.1f} | **{gain:.3f}** | | | |")

  print("\nFine speed sweep (NNLC latAccelFactor):\n")
  print("| mph | " + " | ".join(str(s) for s in range(5, 80, 5)) + " |")
  print("|---|" + "---|" * len(range(5, 80, 5)))
  print("| gain | " + " | ".join(f"{implied_gain(model, s * MPH, args.max_lat_acc)[0]:.2f}" for s in range(5, 80, 5)) + " |")

  if args.logs:
    log_check(model, args.logs, args.jobs)


if __name__ == "__main__":
  main()
