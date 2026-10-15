#!/usr/bin/env python3
"""Offline per-speed-band fits of the online lateral learners (T-1).

Every online learner publishes ONE value set for all speeds, and two of them only
learn at highway speed:

  torqued  latAccelFactor / friction / latAccelOffset   vEgo > 15 m/s, |lat accel| <= 1
  lagd     lateral actuator delay                       vEgo > 50 mph
  paramsd  steerRatio (+ stiffness, offset, roll)       vEgo > 1 m/s, averaged over all

This replays logs through the upstream learners with only the speed gate moved, tags
every accepted sample with its speed, and fits each band separately. It answers the
question docs/lat/torqued-lowspeed.md defers to an offline fit: do these values
actually vary with speed on this car?

  torque  upstream TorqueEstimator (track_all_points), MIN_VEL lowered; samples tagged
          with vEgo and fitted per band with torqued's own total-least-squares
  lag     upstream LateralLagEstimator, one instance per band (min_vego = band floor,
          samples above the band ceiling masked); rlogs only, it needs 20 Hz deviceMotion
  steer   not a KF replay. Regresses measured curvature (calibrated yaw rate / v, roll
  ratio   removed) on VehicleModel.curvature_factor(v) * steer angle per band, using the
          logged stiffness, roll and average angle offset. slope = 1 / steerRatio

Each learner also gets a STOCK row using the on-device gates. It should land near the
value the car logged; if it does not, distrust the band rows too.

Confidence intervals are bootstrapped over segments, not samples: samples within a
segment are strongly correlated and a per-sample CI would be far too narrow.

Usage:
  learner_bands.py <rlog/qlog files or dirs...> [--out DIR] [--min-speed 2.0]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.locationd import torqued, lagd
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.sunnypilot.selfdrive.locationd import torqued_ext
from openpilot.tools.lat_metrics.lat_metrics import BANDS

STOCK = "stock"
TORQUED_MIN_VEL = torqued.MIN_VEL                  # read before _isolate_learners lowers it
TORQUE_LAT_ACC_STOCK = torqued.LAT_ACC_THRESHOLD   # 1 m/s^2, torqued's linear-region gate
TORQUE_LAT_ACC_WIDE = 3.0                          # covers low-speed corners
SR_MAX_ANGLE = 45.0         # paramsd's linear-region gate, deg
SR_MAX_STEER_RATE = 10.0    # deg/s; steady-state only, yaw lags steer in transients
SR_MAX_LAT_ACC = 4.0        # paramsd's LATERAL_ACC_SENSOR_THRESHOLD
SR_ANGLE_BINS = [(2, 10), (10, 20), (20, 45)]  # |steer angle| deg; separates rack nonlinearity from speed
SR_MIN_SPEED = 1.0         # paramsd MIN_ACTIVE_SPEED
LAG_DT_RLOG = 1. / 20       # deviceMotion rate in rlogs
LAG_ESTIMATE_EVERY = 5      # lagd runs update_estimate every 5 deviceMotion frames
CI = (5, 95)


class _NoParams:
  """Stand-in for Params so a replay never reads or writes the developer's own params."""
  def __init__(self, *a, **k): pass
  def get(self, *a, **k): return None
  def get_bool(self, *a, **k): return False
  def put(self, *a, **k): pass
  def remove(self, *a, **k): pass


def _isolate_learners(min_speed):
  torqued.Params = _NoParams
  torqued_ext.Params = _NoParams
  torqued.get_lat_delay = lambda _params, lag: lag  # use the delay the car logged
  torqued.MIN_VEL = min_speed


def band_of(v):
  for name, lo, hi in BANDS:
    if lo <= v < hi:
      return name
  return None


# ---------------------------------------------------------------- fits (pure, tested)

def fit_torque(steer, lat_acc):
  """torqued.estimate_params on explicit arrays: (latAccelFactor, latAccelOffset, friction)."""
  points = np.column_stack([steer, np.ones(len(steer)), lat_acc])
  _, _, v = np.linalg.svd(points, full_matrices=False)
  slope, offset = -v.T[0:2, 2] / v.T[2, 2]
  _, spread = np.matmul(points[:, [0, 2]], torqued.slope2rot(slope)).T
  return float(slope), float(offset), float(np.std(spread) * torqued.FRICTION_FACTOR)


def balance_buckets(steer, rng, per_bucket=torqued.POINTS_PER_BUCKET):
  """Indices torqued would fit on once every steer bucket is full: at most per_bucket
  points from each STEER_BUCKET_BOUNDS bucket, nothing outside them. Without this the
  fit is dominated by near-zero steer and lands well above what the car learns."""
  keep = []
  for lo, hi in torqued.STEER_BUCKET_BOUNDS:
    idx = np.flatnonzero((steer >= lo) & (steer < hi))
    keep.append(idx if len(idx) <= per_bucket else rng.choice(idx, per_bucket, replace=False))
  return np.sort(np.concatenate(keep))


def fit_torque_balanced(steer, lat_acc, seed=0):
  idx = balance_buckets(steer, np.random.default_rng(seed))
  return fit_torque(steer[idx], lat_acc[idx])


def response_curve(steer, lat_acc, min_points=30):
  """Median lateral accel per torqued steer bucket: the torque response with no line
  fitted through it, so curvature in it is visible rather than averaged into a slope."""
  out = []
  for lo, hi in torqued.STEER_BUCKET_BOUNDS:
    m = (steer >= lo) & (steer < hi)
    out.append(float(np.median(lat_acc[m])) if m.sum() >= min_points else None)
  return out


def fit_response_slope(steer, lat_acc, min_points=30):
  """Robust latAccelFactor: OLS through per-bucket (median steer, median lat accel).
  torqued's TLS assumes equal noise on both axes, so when the lat-accel spread is large
  next to the signal (low speed) it rotates the line steep; bucket medians do not."""
  xs, ys = [], []
  for lo, hi in torqued.STEER_BUCKET_BOUNDS:
    m = (steer >= lo) & (steer < hi)
    if m.sum() >= min_points:
      xs.append(np.median(steer[m]))
      ys.append(np.median(lat_acc[m]))
  if len(xs) < 5:
    raise ZeroDivisionError("too few populated steer buckets")
  slope, _ = np.polyfit(xs, ys, 1)
  return float(slope)


def fit_steer_ratio(x, y):
  """OLS y = x / SR + b. The intercept soaks up residual angle offset. Returns (SR, b)."""
  A = np.column_stack([x, np.ones(len(x))])
  (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
  return float(1.0 / a), float(b)


def fit_steer_ratio_through(x, y, b):
  """SR with the intercept held at b (from a wider fit): OLS of (y - b) on x through 0."""
  return float(np.dot(x, x) / np.dot(x, y - b))


def bootstrap(seg_ids, fn, n_boot, seed=0):
  """Percentile CI of fn(index_array), resampling whole segments with replacement."""
  rng = np.random.default_rng(seed)
  uniq = np.unique(seg_ids)
  if len(uniq) < 3:
    return float("nan"), float("nan")
  by_seg = {s: np.flatnonzero(seg_ids == s) for s in uniq}
  vals = []
  for _ in range(n_boot):
    idx = np.concatenate([by_seg[s] for s in rng.choice(uniq, len(uniq), replace=True)])
    try:
      vals.append(fn(idx))
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
      continue
  if not vals:
    return float("nan"), float("nan")
  lo, hi = np.percentile(vals, CI)
  return float(lo), float(hi)


def bootstrap_many(seg_ids, fn, n_boot, seed=0):
  """bootstrap() for an fn returning a tuple: one CI per element, from the same resamples."""
  rng = np.random.default_rng(seed)
  uniq = np.unique(seg_ids)
  by_seg = {s: np.flatnonzero(seg_ids == s) for s in uniq}
  vals = []
  for _ in range(n_boot if len(uniq) >= 3 else 0):
    idx = np.concatenate([by_seg[s] for s in rng.choice(uniq, len(uniq), replace=True)])
    try:
      vals.append(fn(idx))
    except (np.linalg.LinAlgError, ZeroDivisionError, FloatingPointError):
      continue
  if not vals:
    return [(float("nan"), float("nan"))] * 3
  return [tuple(float(q) for q in np.percentile(col, CI)) for col in np.array(vals).T]


# ---------------------------------------------------------------- replay

class BandLagEstimator(lagd.LateralLagEstimator):
  """lagd restricted to [min_vego, v_max): samples above the band are not okay."""
  def __init__(self, CP, dt, v_max, **kw):
    super().__init__(CP, dt, **kw)
    self.v_max = v_max
    self.accepted = []
    update = self.block_avg.update

    def record(value):
      self.accepted.append(float(value))
      update(value)
    self.block_avg.update = record

  def update_points(self):
    super().update_points()
    if self.v_ego >= self.v_max:
      self.points.okay[-1] = False


def _new_lag_estimators(CP, okay_window):
  ests = {name: BandLagEstimator(CP, LAG_DT_RLOG, hi, min_vego=lo, okay_window_sec=okay_window) for name, lo, hi in BANDS}
  ests[STOCK] = BandLagEstimator(CP, LAG_DT_RLOG, np.inf, okay_window_sec=okay_window)  # lagd's default floor: > 50 mph
  return ests


@dataclass
class RouteResult:
  fingerprint: str = ""
  torque: list = field(default_factory=list)   # (seg, v, steer, lat_acc)
  sr: list = field(default_factory=list)       # (seg, v, x, y, steer angle deg net of offset)
  lag: list = field(default_factory=list)      # (seg, band, delay)
  lag_okay_s: dict = field(default_factory=dict)  # band -> seconds of samples lagd accepted as okay
  logged: list = field(default_factory=list)   # per-segment final on-device values
  segments: list = field(default_factory=list)


LAG_INPUTS = lagd.LateralLagEstimator.inputs
TORQUE_INPUTS = {"carControl", "carOutput", "carState", "extrinsicsCalibration", "deviceMotion", "lateralDelay"}


def process_route(files, min_speed, lag_okay_window=lagd.MIN_OKAY_WINDOW_SEC):
  """files: [(seg_no, kind, path)] for one route, sorted. Estimators carry across
  consecutive segments and reset at a gap, as they would on the car."""
  from openpilot.tools.lib.logreader import LogReader
  _isolate_learners(min_speed)
  res = RouteResult()
  CP = VM = torque = lag_ests = None
  calib = PoseCalibrator()
  prev_seg = prev_kind = None

  for seg_no, kind, path in files:
    seg_id = f"{os.path.basename(os.path.dirname(path))}"
    contiguous = prev_seg is not None and seg_no == prev_seg + 1
    if not contiguous and CP is not None:
      torque = torque.__class__(CP, decimated=(kind == "qlog"), track_all_points=True)
    if CP is not None and (not contiguous or kind != "rlog" or prev_kind != "rlog"):
      lag_ests = _new_lag_estimators(CP, lag_okay_window) if kind == "rlog" else None
    prev_seg, prev_kind = seg_no, kind
    res.segments.append((seg_id, kind))

    cs = vp = None
    lag_frame = 0
    last = {}
    try:
      msgs = list(LogReader(path))
    except Exception as e:  # a corrupt segment should not sink the run
      print(f"skip {path}: {e}", file=sys.stderr)
      continue

    for m in msgs:
      which = m.which()
      if which == "carParams" and CP is None:
        CP = m.carParams
        res.fingerprint = CP.carFingerprint
        VM = VehicleModel(CP)
        torque = torqued.TorqueEstimator(CP, decimated=(kind == "qlog"), track_all_points=True)
        lag_ests = _new_lag_estimators(CP, lag_okay_window) if kind == "rlog" else None
        continue
      if CP is None or not m.valid:
        continue
      t = m.logMonoTime * 1e-9
      msg = getattr(m, which)

      if which in ("vehicleParameters", "lateralTorqueParameters", "lateralDelay"):
        last[which] = msg
      if which == "carState":
        cs = msg
      elif which == "vehicleParameters":
        vp = msg
      elif which == "extrinsicsCalibration":
        calib.feed_extrinsics_calibration(msg)

      # torque: upstream estimator, tag each new point with its speed
      if which in TORQUE_INPUTS:
        n0 = len(torque.all_torque_points)
        torque.handle_log(t, which, msg)
        if len(torque.all_torque_points) > n0:
          steer, lat_acc = torque.all_torque_points[-1]
          v = float(np.interp(msg.timestamp * 1e-9, torque.raw_points["carState_t"], torque.raw_points["vego"]))
          res.torque.append((seg_id, v, steer, lat_acc))

      # lag: one upstream estimator per band, driven at deviceMotion rate like lagd.main
      if lag_ests is not None and which in LAG_INPUTS:
        for est in lag_ests.values():
          est.handle_log(t, which, msg)
        if which == "deviceMotion":
          lag_frame += 1
          for name, est in lag_ests.items():
            est.update_points()
            if est.points.okay[-1]:
              res.lag_okay_s[name] = res.lag_okay_s.get(name, 0.0) + LAG_DT_RLOG
            if lag_frame % LAG_ESTIMATE_EVERY == 0:
              n0 = len(est.accepted)
              est.update_estimate()
              res.lag.extend((seg_id, name, d) for d in est.accepted[n0:])

      # steer ratio: steady-state curvature regression sample
      if which == "deviceMotion" and cs is not None and vp is not None and calib.calib_valid:
        v = cs.vEgo
        if v < SR_MIN_SPEED or abs(cs.steeringAngleDeg) >= SR_MAX_ANGLE or abs(cs.steeringRateDeg) > SR_MAX_STEER_RATE:
          continue
        if str(cs.gearShifter) == "reverse" or not (msg.angularVelocityDevice.valid and msg.posenetOK):
          continue
        yaw = calib.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity.yaw
        if abs(yaw) >= 1.0 or abs(yaw * v) > SR_MAX_LAT_ACC:
          continue
        VM.update_params(max(vp.stiffnessFactor, 0.1), 1.0)
        angle = cs.steeringAngleDeg - vp.angleOffsetAverageDeg
        x = VM.curvature_factor(v) * np.radians(angle)
        y = -yaw / v - VM.roll_compensation(vp.roll, v)
        res.sr.append((seg_id, v, float(x), float(y), float(angle)))

    if last:
      row = {"seg": seg_id}
      if (p := last.get("vehicleParameters")) is not None:
        row.update(steerRatio=p.steerRatio, stiffness=p.stiffnessFactor, angleOffsetAvg=p.angleOffsetAverageDeg)
      if (p := last.get("lateralTorqueParameters")) is not None:
        row.update(latAccelFactor=p.latAccelFactorFiltered, friction=p.frictionCoefficientFiltered,
                   latAccelOffset=p.latAccelOffsetFiltered, torqueValid=p.valid)
      if (p := last.get("lateralDelay")) is not None:
        row.update(lateralDelay=p.lateralDelay, lagStatus=str(p.status))
      res.logged.append(row)
  return res


# ---------------------------------------------------------------- inputs

SEG_RE = re.compile(r"^(?P<route>[0-9a-f]{8}--[0-9a-f]{10})--(?P<seg>\d+)$")


def discover(paths):
  """-> {fingerprint-agnostic route: [(seg_no, kind, path)]}, preferring rlog over qlog
  when both exist for a segment (they overlap; counting both double-weights it)."""
  found = {}
  for p in paths:
    cands = [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
    for f in cands:
      base = os.path.basename(f)
      kind = "rlog" if base.startswith("rlog") else "qlog" if base.startswith("qlog") else None
      m = SEG_RE.match(os.path.basename(os.path.dirname(os.path.abspath(f))))
      if kind is None or m is None:
        continue
      key = (m["route"], int(m["seg"]))
      if key not in found or (kind == "rlog" and found[key][0] == "qlog"):
        found[key] = (kind, f)
  routes = defaultdict(list)
  for (route, seg), (kind, f) in sorted(found.items()):
    routes[route].append((seg, kind, f))
  return dict(routes)


# ---------------------------------------------------------------- report

def _fmt(x, nd=3):
  return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def summarize(results, n_boot):
  lag_okay = defaultdict(float)
  for res in results:
    for k, sec in res.lag_okay_s.items():
      lag_okay[k] += sec
  torque = [r for res in results for r in res.torque]
  sr = [r for res in results for r in res.sr]
  lag = [r for res in results for r in res.lag]
  out = {"torque": {}, "steer_ratio": {}, "lag": {}}

  # torque
  if torque:
    seg = np.array([r[0] for r in torque])
    v, steer, la = (np.array([r[i] for r in torque]) for i in (1, 2, 3))
    groups = [(STOCK, TORQUE_LAT_ACC_STOCK, v > TORQUED_MIN_VEL)]
    for name, lo, hi in BANDS:
      for lim in (TORQUE_LAT_ACC_STOCK, TORQUE_LAT_ACC_WIDE):
        groups.append((name, lim, (v >= lo) & (v < hi)))
    for name, lim, vmask in groups:
      mask = vmask & (np.abs(la) <= lim)
      idx = np.flatnonzero(mask)
      key = f"{name}@{lim:g}"
      if len(idx) < 50 or len(np.unique(seg[idx])) < 3:
        out["torque"][key] = {"n": int(len(idx)), "segments": int(len(np.unique(seg[idx])))}
        continue
      laf, off, fr = fit_torque_balanced(steer[idx], la[idx])
      boot = bootstrap_many(seg[idx], lambda i, idx=idx: fit_torque_balanced(steer[idx][i], la[idx][i]), n_boot)
      ci_laf, ci_fr = boot[0], boot[2]
      try:
        robust = fit_response_slope(steer[idx], la[idx])
        ci_robust = bootstrap(seg[idx], lambda i, idx=idx: fit_response_slope(steer[idx][i], la[idx][i]), n_boot)
      except ZeroDivisionError:
        robust, ci_robust = float("nan"), (float("nan"), float("nan"))
      counts = [int(np.sum((steer[idx] >= a) & (steer[idx] < b))) for a, b in torqued.STEER_BUCKET_BOUNDS]
      out["torque"][key] = {"n": int(len(idx)), "segments": int(len(np.unique(seg[idx]))),
                            "median_lat_acc": response_curve(steer[idx], la[idx]),
                            "latAccelFactor": laf, "latAccelFactor_ci": ci_laf,
                            "medianSlope": robust, "medianSlope_ci": ci_robust, "latAccelOffset": off,
                            "friction": fr, "friction_ci": ci_fr, "steer_bucket_counts": counts}

  # steer ratio
  if sr:
    seg = np.array([r[0] for r in sr])
    v, x, y, ang = (np.array([r[i] for r in sr]) for i in (1, 2, 3, 4))
    groups = [(STOCK, v >= SR_MIN_SPEED)] + [(name, (v >= lo) & (v < hi)) for name, lo, hi in BANDS]
    for name, mask in groups:
      idx = np.flatnonzero(mask)
      if len(idx) < 50 or len(np.unique(seg[idx])) < 3:
        out["steer_ratio"][name] = {"n": int(len(idx))}
        continue
      s, b = fit_steer_ratio(x[idx], y[idx])
      ci = bootstrap(seg[idx], lambda i, idx=idx: fit_steer_ratio(x[idx][i], y[idx][i])[0], n_boot)
      resid = y[idx] - (x[idx] / s + b)
      out["steer_ratio"][name] = {"n": int(len(idx)), "segments": int(len(np.unique(seg[idx]))),
                                  "steerRatio": s, "steerRatio_ci": ci, "intercept": b,
                                  "resid_std": float(np.std(resid)),
                                  "angle_deg_p95": float(np.percentile(np.abs(ang[idx]), 95))}

    # A variable-ratio rack makes SR depend on steer angle, and low speed means big angles.
    # Speed x angle cells, intercept held at the stock fit's, tell the two effects apart.
    if "intercept" in out["steer_ratio"].get(STOCK, {}):
      b0 = out["steer_ratio"][STOCK]["intercept"]
      grid = {}
      for name, lo, hi in [(STOCK, SR_MIN_SPEED, np.inf)] + BANDS:
        for alo, ahi in SR_ANGLE_BINS:
          idx = np.flatnonzero((v >= lo) & (v < hi) & (np.abs(ang) >= alo) & (np.abs(ang) < ahi))
          cell = {"n": int(len(idx)), "segments": int(len(np.unique(seg[idx])))}
          if len(idx) >= 50 and cell["segments"] >= 3:
            cell["steerRatio"] = fit_steer_ratio_through(x[idx], y[idx], b0)
            cell["steerRatio_ci"] = bootstrap(seg[idx], lambda i, idx=idx: fit_steer_ratio_through(x[idx][i], y[idx][i], b0), n_boot)
          grid[f"{name}@{alo}-{ahi}"] = cell
      out["steer_ratio_by_angle"] = grid

  # lag
  seg = np.array([r[0] for r in lag], dtype=object)
  band = np.array([r[1] for r in lag], dtype=object)
  d = np.array([r[2] for r in lag], dtype=float)
  for name in [STOCK] + [b[0] for b in BANDS]:
    idx = np.flatnonzero(band == name)
    entry = {"n": int(len(idx)), "segments": int(len(set(seg[idx]))), "okay_s": float(lag_okay[name])}
    if len(idx) >= 3:
      entry.update(median=float(np.median(d[idx])), iqr=[float(q) for q in np.percentile(d[idx], [25, 75])],
                   median_ci=bootstrap(seg[idx], lambda i, idx=idx: float(np.median(d[idx][i])), n_boot))
    out["lag"][name] = entry

  # what the car logged
  logged = [row for res in results for row in res.logged]
  out["logged"] = {k: float(np.median([r[k] for r in logged if k in r]))
                   for k in ("steerRatio", "stiffness", "angleOffsetAvg", "latAccelFactor", "friction", "latAccelOffset", "lateralDelay")
                   if any(k in r for r in logged)}
  return out


BAND_LABEL = {STOCK: "stock gates", "lt15": "< 15 mph", "15_25": "15–25 mph", "25_40": "25–40 mph", "gt40": "> 40 mph"}


def _ci(e, key, nd=3):
  lo, hi = e[key]
  return f"{_fmt(lo, nd)}–{_fmt(hi, nd)}"


def render(summary, meta):
  L = summary["logged"]
  lines = [
    f"# Learner values per speed band — {meta['fingerprint']}", "",
    (f"Generated by `openpilot/tools/lat_metrics/learner_bands.py` over {meta['segments']} segments " +
     f"({meta['rlog']} rlog, {meta['qlog']} qlog) from {meta['routes']} routes. " +
     f"Torque points taken from vEgo > {meta['min_speed']:g} m/s. CIs are {CI[0]}–{CI[1]} % " +
     f"segment-bootstrap ({meta['n_boot']} resamples)."), "",
    ("`stock gates` rows use the on-device gates and should land near the logged value; " +
     "if they do not, distrust the band rows."), "",
  ]

  lines += [
    "## Torque (`torqued`)", "",
    (f"Logged on device: latAccelFactor **{_fmt(L.get('latAccelFactor'))}**, friction **{_fmt(L.get('friction'))}**, " +
     f"latAccelOffset {_fmt(L.get('latAccelOffset'))}. Fits are balanced like torqued's full buckets " +
     f"(≤ {torqued.POINTS_PER_BUCKET} points per steer bucket, \\|steer\\| < {torqued.STEER_BUCKET_BOUNDS[-1][1]})."), "",
    ("*latAccelFactor* is torqued's own TLS fit (the like-for-like check against the car). *median slope* " +
     "is a line through per-bucket medians, which TLS's equal-noise assumption cannot bias; compare bands on it."), "",
    "| band | \\|lat acc\\| ≤ | points | segs | latAccelFactor (TLS) | 90 % CI | median slope | 90 % CI | friction | 90 % CI | offset |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
  ]
  for key, e in summary["torque"].items():
    name, lim = key.split("@")
    if "latAccelFactor" not in e:
      lines.append(f"| {BAND_LABEL[name]} | {lim} | {e['n']} | {e['segments']} | too few | | | | | | |")
      continue
    lines.append(f"| {BAND_LABEL[name]} | {lim} | {e['n']} | {e['segments']} | **{_fmt(e['latAccelFactor'])}** | " +
                 f"{_ci(e, 'latAccelFactor_ci')} | **{_fmt(e['medianSlope'])}** | {_ci(e, 'medianSlope_ci')} | " +
                 f"{_fmt(e['friction'])} | {_ci(e, 'friction_ci')} | {_fmt(e['latAccelOffset'])} |")
  bounds = torqued.STEER_BUCKET_BOUNDS
  lines += ["", (f"Points per torqued steer bucket (bounds {bounds[0][0]}…{bounds[-1][1]}; torqued wants " +
                 f"≥ {torqued.MIN_BUCKET_POINTS.tolist()} on rlogs, a tenth of that on qlogs):"), "", "```"]
  lines += [f"{key:>12}  {e['steer_bucket_counts']}" for key, e in summary["torque"].items() if "steer_bucket_counts" in e]
  lines += ["```", ""]

  bounds_hdr = " | ".join(f"{lo:+g}…{hi:+g}" for lo, hi in bounds)
  lines += [
    ("Median lateral accel (m/s²) per steer bucket, \\|lat acc\\| ≤ " + f"{TORQUE_LAT_ACC_WIDE:g}. " +
     "A constant latAccelFactor means each row scales linearly with the bucket centre; " +
     "rows that bend, or differ in shape between bands, are what one slope per band cannot capture."), "",
    f"| band | {bounds_hdr} |", "|---|" + "---|" * len(bounds),
  ]
  for key, e in summary["torque"].items():
    name, lim = key.split("@")
    if "median_lat_acc" in e and float(lim) == TORQUE_LAT_ACC_WIDE:
      lines.append(f"| {BAND_LABEL[name]} | " + " | ".join(_fmt(x, 2) for x in e["median_lat_acc"]) + " |")
  lines.append("")

  lines += [
    "## Steer ratio (`paramsd` substitute)", "",
    (f"Logged on device: steerRatio **{_fmt(L.get('steerRatio'), 2)}**, stiffness {_fmt(L.get('stiffness'))}, " +
     f"angle offset {_fmt(L.get('angleOffsetAvg'), 2)}°. Steady-state frames only " +
     f"(\\|steer rate\\| ≤ {SR_MAX_STEER_RATE:g}°/s, \\|angle\\| < {SR_MAX_ANGLE:g}°)."), "",
    "| band | samples | segs | steerRatio | 90 % CI | intercept (1/m) | resid std (1/m) | p95 \\|steer angle\\| ° |",
    "|---|---|---|---|---|---|---|---|",
  ]
  for name, e in summary["steer_ratio"].items():
    if "steerRatio" not in e:
      lines.append(f"| {BAND_LABEL[name]} | {e['n']} | | too few | | | | |")
      continue
    lines.append(f"| {BAND_LABEL[name]} | {e['n']} | {e['segments']} | **{_fmt(e['steerRatio'], 2)}** | " +
                 f"{_ci(e, 'steerRatio_ci', 2)} | {e['intercept']:.5f} | {e['resid_std']:.5f} | {_fmt(e['angle_deg_p95'], 1)} |")
  if grid := summary.get("steer_ratio_by_angle"):
    lines += [
      "",
      ("By speed and steer angle, intercept held at the stock fit's. If SR changes across a row but not down a " +
       "column, it is the rack's angle-dependent ratio, not speed."), "",
      "| band | " + " | ".join(f"{lo}–{hi}°" for lo, hi in SR_ANGLE_BINS) + " |",
      "|---|" + "---|" * len(SR_ANGLE_BINS),
    ]
    for name in [STOCK] + [b[0] for b in BANDS]:
      cells = []
      for lo, hi in SR_ANGLE_BINS:
        c = grid[f"{name}@{lo}-{hi}"]
        cells.append(f"**{_fmt(c['steerRatio'], 2)}** ({_ci(c, 'steerRatio_ci', 1)}, n={c['n']})" if "steerRatio" in c
                     else f"n={c['n']}")
      lines.append(f"| {BAND_LABEL[name]} | " + " | ".join(cells) + " |")
  lines.append("")

  lines += [
    "## Lateral delay (`lagd`, rlogs only)", "",
    (f"Logged on device: lateralDelay **{_fmt(L.get('lateralDelay'))} s**. An estimate needs ≥ " +
     f"{meta['lag_okay_window']:g} s of okay samples inside a {lagd.MOVING_WINDOW_SEC:g} s window, plus " +
     "lagd's correlation and confidence checks; *okay s* shows how much qualifying driving each band saw." +
     ("" if meta["lag_okay_window"] == lagd.MIN_OKAY_WINDOW_SEC else
      f" **Non-stock:** lagd itself requires {lagd.MIN_OKAY_WINDOW_SEC:g} s; these rows use a relaxed window.")), "",
    "| band | okay s | estimates | segs | median s | IQR | 90 % CI of median |",
    "|---|---|---|---|---|---|---|",
  ]
  for name, e in summary["lag"].items():
    if "median" not in e:
      lines.append(f"| {BAND_LABEL[name]} | {e['okay_s']:.0f} | {e['n']} | {e['segments']} | too few | | |")
      continue
    lines.append(f"| {BAND_LABEL[name]} | {e['okay_s']:.0f} | {e['n']} | {e['segments']} | **{_fmt(e['median'])}** | " +
                 f"{_fmt(e['iqr'][0])}–{_fmt(e['iqr'][1])} | {_ci(e, 'median_ci')} |")
  lines.append("")
  return "\n".join(lines)


def lag_okay_seconds(path):
  """Seconds per band that would pass lagd's okay test, approximated from a qlog's 10 Hz
  controls messages, so rlogs worth fetching can be chosen before paying for them.
  Approximations: lat accel from the controller's actual (not pose), and no calib check."""
  from openpilot.tools.lib.logreader import LogReader
  okay = dict.fromkeys([STOCK] + [b[0] for b in BANDS], 0.0)
  lat_active = pressed = False
  v = 0.0
  last_bad_t = -1e9
  prev_t = None
  for m in LogReader(path):
    which = m.which()
    if which == "carControl":
      lat_active = m.carControl.latActive
    elif which == "carState":
      pressed = m.carState.steeringPressed
      v = m.carState.vEgo
    elif which == "controlsState":
      t = m.logMonoTime * 1e-9
      dt, prev_t = (0.0 if prev_t is None else min(t - prev_t, 0.5)), t
      lcs = m.controlsState.lateralControlState
      ts = getattr(lcs, lcs.which())
      la_ok = lcs.which() == "torqueState" and abs(ts.actualLateralAccel) <= lagd.MAX_LAT_ACCEL and \
        abs(ts.desiredLateralAccel - ts.actualLateralAccel) <= lagd.MAX_LAT_ACCEL_DIFF
      if not lat_active or pressed or ts.saturated or not la_ok:
        last_bad_t = t
      if t - last_bad_t >= lagd.MIN_RECOVERY_BUFFER_SEC:
        if v > lagd.MIN_VEGO:
          okay[STOCK] += dt
        if (band := band_of(v)) is not None:
          okay[band] += dt
  return okay


def screen_lag(paths, jobs, min_okay):
  routes = discover(paths)
  files = [(f"{r}--{seg}", f) for r, segs in routes.items() for seg, _, f in segs]
  with ProcessPoolExecutor(jobs) as ex:
    results = list(ex.map(lag_okay_seconds, [f for _, f in files], chunksize=4))
  names = [STOCK] + [b[0] for b in BANDS]
  print("segment," + ",".join(names))
  totals = dict.fromkeys(names, 0.0)
  picked = 0
  for (seg, _), okay in zip(files, results, strict=True):
    for k in names:
      totals[k] += okay[k]
    if max(okay.values()) >= min_okay:
      picked += 1
      print(seg + "," + ",".join(f"{okay[k]:.0f}" for k in names))
  print(f"# {picked}/{len(files)} segments with >= {min_okay:g} s okay in some band; totals (s): " +
        ", ".join(f"{k} {totals[k]:.0f}" for k in names), file=sys.stderr)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("logs", nargs="+", help="rlog/qlog files, or directories containing <route>--<seg>/{rlog,qlog}*")
  ap.add_argument("--out", default="learner_bands_out")
  ap.add_argument("--min-speed", type=float, default=2.0, help="torque point speed floor, m/s (torqued uses 15)")
  ap.add_argument("--bootstrap", type=int, default=200)
  ap.add_argument("--jobs", type=int, default=os.cpu_count())
  ap.add_argument("--lag-okay-window", type=float, default=lagd.MIN_OKAY_WINDOW_SEC,
                  help=f"okay seconds lagd needs per {lagd.MOVING_WINDOW_SEC:g} s window (stock {lagd.MIN_OKAY_WINDOW_SEC:g}); " +
                       "lowering it reaches low-speed bands but is no longer lagd's estimator")
  ap.add_argument("--screen-lag", type=float, metavar="SECONDS",
                  help="instead of fitting, list segments with at least this many lagd-okay seconds in some band " +
                       "(run on qlogs to choose which rlogs to fetch)")
  args = ap.parse_args()

  if args.screen_lag is not None:
    screen_lag(args.logs, args.jobs, args.screen_lag)
    return

  routes = discover(args.logs)
  if not routes:
    sys.exit("no <route>--<seg>/rlog|qlog files found")
  print(f"{sum(map(len, routes.values()))} segments in {len(routes)} routes", file=sys.stderr)

  with ProcessPoolExecutor(args.jobs) as ex:
    results = list(ex.map(process_route, routes.values(), [args.min_speed] * len(routes), [args.lag_okay_window] * len(routes)))

  fps = {r.fingerprint for r in results if r.fingerprint}
  if len(fps) > 1:
    sys.exit(f"logs span several cars {sorted(fps)}; pass one car's logs at a time")

  summary = summarize(results, args.bootstrap)
  kinds = [k for r in results for _, k in r.segments]
  meta = {"fingerprint": next(iter(fps), "?"), "segments": len(kinds), "rlog": kinds.count("rlog"),
          "qlog": kinds.count("qlog"), "routes": len(routes), "min_speed": args.min_speed, "n_boot": args.bootstrap,
          "lag_okay_window": args.lag_okay_window}

  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, "learner_bands.json"), "w") as f:
    json.dump({"meta": meta, **summary}, f, indent=1)
  report = render(summary, meta)
  with open(os.path.join(args.out, "learner_bands.md"), "w") as f:
    f.write(report)
  print(report)


if __name__ == "__main__":
  main()
