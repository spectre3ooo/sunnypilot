#!/usr/bin/env python3
"""Per-frame steering dataset from rlogs for angle_ff_fit.py: one npz per segment, sampled on carState and
decimated to 20 Hz. Also records the stock camera's LKAS request (bus 2, 0x50) at full rate.

Usage:
  steer_dataset.py <out dir> <rlog files or dirs...>
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

COLS = ["t", "v", "a", "angle", "rate", "drv", "eps", "lat", "pressed", "cmd", "asked", "des_la", "act_la",
        "roll", "la_meas", "cam_req", "cam_act", "nnlc_err", "fault"]
DECIMATION = 5  # carState is 100 Hz


def seg(path, out_dir):
  from opendbc.can.parser import CANParser
  from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
  from openpilot.tools.lib.logreader import LogReader

  name = os.path.basename(os.path.dirname(path))
  dst = os.path.join(out_dir, name + ".npz")
  if os.path.exists(dst):
    return name, "cached"
  cam = CANParser("hyundai_canfd_generated", [("LKAS", 100)], 2)
  calib = PoseCalibrator()
  cur = {"lat": 0.0, "cmd": 0.0, "asked": 0.0, "des_la": 0.0, "act_la": 0.0, "roll": 0.0, "la_meas": np.nan,
         "cam_req": 0.0, "cam_act": 0.0, "nnlc_err": np.nan}
  rows, cam_peaks = [], []
  n, v, t0 = 0, 0.0, None
  try:
    for m in LogReader(path):
      w = m.which()
      t = m.logMonoTime * 1e-9
      if w == "carControl":
        cur["lat"] = float(m.carControl.latActive)
        cur["asked"] = m.carControl.actuators.torque * 270
      elif w == "carOutput":
        cur["cmd"] = m.carOutput.actuatorsOutput.torque * 270
      elif w == "controlsState":
        lcs = m.controlsState.lateralControlState
        if lcs.which() == "torqueState":
          ts = lcs.torqueState
          cur["des_la"], cur["act_la"] = ts.desiredLateralAccel, ts.actualLateralAccel
          # stock logs exactly desired - actual as the error; NNLC overwrites it (see nnlc_gain.py)
          cur["nnlc_err"] = ts.error - (ts.desiredLateralAccel - ts.actualLateralAccel)
      elif w == "vehicleParameters":
        cur["roll"] = m.vehicleParameters.roll
      elif w == "extrinsicsCalibration":
        calib.feed_extrinsics_calibration(m.extrinsicsCalibration)
      elif w == "deviceMotion":
        if calib.calib_valid:
          yaw = calib.build_calibrated_pose(Pose.from_device_motion(m.deviceMotion)).angular_velocity.yaw
          cur["la_meas"] = -yaw * v  # left positive, matching the controller's lateral accel
      elif w == "can":
        frames = [(c.address, bytes(c.dat), c.src) for c in m.can if c.src == 2 and c.address == 0x50]
        if frames and 0x50 in cam.update([m.logMonoTime, frames]):
          cur["cam_req"], cur["cam_act"] = cam.vl["LKAS"]["StrTqReqVal"], cam.vl["LKAS"]["ActToiSta"]
          cam_peaks.append((v, abs(cur["cam_req"]), cur["cam_act"]))
      elif w == "carState":
        cs = m.carState
        v = cs.vEgo
        if t0 is None:
          t0 = t
        n += 1
        if n % DECIMATION:
          continue
        rows.append((t - t0, v, cs.aEgo, cs.steeringAngleDeg, cs.steeringRateDeg, cs.steeringTorque, cs.steeringTorqueEps,
                     cur["lat"], float(cs.steeringPressed), cur["cmd"], cur["asked"], cur["des_la"], cur["act_la"],
                     cur["roll"], cur["la_meas"], cur["cam_req"], cur["cam_act"], cur["nnlc_err"],
                     float(cs.steerFaultTemporary)))
  except Exception as e:  # a truncated segment still yields what was read before the error
    if not rows:
      return name, f"error {e}"
  np.savez_compressed(dst, data=np.array(rows, dtype=np.float32), cam=np.array(cam_peaks, dtype=np.float32).reshape(-1, 3))
  return name, len(rows)


def main():
  out_dir, paths = sys.argv[1], sys.argv[2:]
  os.makedirs(out_dir, exist_ok=True)
  files = sorted({f for p in paths for f in ([p] if os.path.isfile(p) else glob.glob(os.path.join(p, "**", "rlog*"), recursive=True))})
  with ProcessPoolExecutor() as ex:
    res = list(ex.map(seg, files, [out_dir] * len(files), chunksize=4))
  bad = [r for r in res if isinstance(r[1], str) and r[1].startswith("error")]
  print(f"{len(res)} segments, {sum(r[1] for r in res if isinstance(r[1], int))} frames, errors {len(bad)}", bad[:3])


if __name__ == "__main__":
  main()
