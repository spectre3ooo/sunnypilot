#!/usr/bin/env python3
"""Baseline lateral-limit metrics for the Ioniq 5 low-speed authority work (T0.3).

Answers the Phase 0 question: below 40 mph, in tight corners, is openpilot's steering
request limited by the TORQUE CAP or by the RATE LIMIT? Everything downstream (the
derived envelope in Phase 2) depends on that split.

The key comparison needs no new telemetry:
  carControl.actuators.torque        -> what the controller ASKED for (unlimited)
  carOutput.actuatorsOutput.torque   -> what the Python limiter LET THROUGH
Frames where they differ are limited; if the output is pinned at +/-1.0 the cap binds,
otherwise the rate does.

Usage:
  lat_metrics.py <rlog...> [--geofence f.json] [--out DIR]
"""
import argparse
import collections
import csv
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.lib.logreader import LogReader
from opendbc.can.parser import CANParser

MPH = 0.44704
DBC = "hyundai_canfd_generated"

# Speed bands from the plan (mph). Sub-40 is the regime of interest; >40 is the
# control band that must NOT change when the envelope moves.
BANDS = [("lt15", 0.0, 15 * MPH), ("15_25", 15 * MPH, 25 * MPH),
         ("25_40", 25 * MPH, 40 * MPH), ("gt40", 40 * MPH, 1e9)]

LIMITED_EPS = 0.01     # |asked - delivered| above this counts as limited
CAP_FRAC = 0.98        # |delivered| at/above this means the cap is binding
REVERSAL_TQ = 0.30     # |torque| a bend must reach to count as a real bend
CURVE_K = 0.005        # |desiredCurvature| defining a curve, per the plan
CUTTING_X = 30.0       # look 0..30 m ahead for the inside-cutting clearance
LANE_PROB_MIN = 0.6    # below this, fall back to road edges
EDGE_STD_MAX = 0.3     # road edge usable only if std is at least this tight


@dataclass
class Seg:
    """Time-aligned per-frame arrays, sampled on carOutput (the limiter's output)."""
    name: str = ""
    t: list = field(default_factory=list)          # seconds from segment start
    v: list = field(default_factory=list)          # vEgo, m/s
    asked: list = field(default_factory=list)      # carControl torque
    got: list = field(default_factory=list)        # carOutput torque
    lat_active: list = field(default_factory=list)
    pressed: list = field(default_factory=list)
    angle: list = field(default_factory=list)
    eps: list = field(default_factory=list)        # steeringTorqueEps
    fault_tmp: list = field(default_factory=list)
    des_acc: list = field(default_factory=list)    # torqueState.desiredLateralAccel
    act_acc: list = field(default_factory=list)    # torqueState.actualLateralAccel
    saturated: list = field(default_factory=list)
    # model frames (20 Hz, own clock)
    mt: list = field(default_factory=list)
    mcurv: list = field(default_factory=list)
    clear_min: list = field(default_factory=list)  # min clearance 0..30 m
    clear_x0: list = field(default_factory=list)   # clearance at x=0
    bound_edge: list = field(default_factory=list) # 1 if road edge used, else 0
    # CAN: raw LKAS series, sampled per sendcan frame and indexed onto t
    act_idx: list = field(default_factory=list)    # frame index of each LKAS TX
    act_val: list = field(default_factory=list)    # ActToiSta
    act_tq: list = field(default_factory=list)     # StrTqReqVal (counts)
    # derived steer-request cuts (populated by _find_cuts)
    cut_idx: list = field(default_factory=list)
    cut_angle: list = field(default_factory=list)
    cut_len: list = field(default_factory=list)
    cut_tq_held: list = field(default_factory=list)
    cut_eps_dip: list = field(default_factory=list)


def _interp_y(xs, ys, x):
    if len(xs) < 2:
        return None
    return float(np.interp(x, xs, ys))


def _clearance(mv):
    """Min clearance from the planned path to the INNER boundary over 0..30 m.

    Sign conventions in the model frame are easy to get wrong (laneLines[1] sits at
    negative y here, and positive desiredCurvature drives the path positive), so the
    inner side is taken from the path itself: whichever way the plan bends is the
    side it can cut toward. The boundary is then whichever lane line / road edge
    lies on that same side, chosen by sign rather than by array index.

    Returns (min_clearance, clearance_at_x0, used_road_edge). Positive clearance
    means the path is inside the boundary; negative means it has crossed it.
    """
    k = mv.action.desiredCurvature
    if abs(k) < CURVE_K:
        return None, None, 0
    px, py = list(mv.position.x), list(mv.position.y)
    if len(px) < 2:
        return None, None, 0

    x_far = min(CUTTING_X, max(px))
    y_far = _interp_y(px, py, x_far)
    if y_far is None or abs(y_far) < 1e-3:
        return None, None, 0
    side = 1.0 if y_far > 0 else -1.0   # the direction the plan bends

    probs = list(mv.laneLineProbs)
    stds = list(mv.roadEdgeStds)
    bx = by = None
    use_edge = 0

    # prefer a confident lane line on the inner side
    best = None
    for i in (1, 2):
        if i >= len(mv.laneLines) or i >= len(probs) or probs[i] < LANE_PROB_MIN:
            continue
        ln = mv.laneLines[i]
        y0 = _interp_y(list(ln.x), list(ln.y), 0.0)
        if y0 is None or (y0 * side) <= 0:
            continue
        if best is None or abs(y0) < best[0]:
            best = (abs(y0), ln)
    if best is not None:
        bx, by = list(best[1].x), list(best[1].y)
    else:
        # fall back to a road edge on the inner side (the rural no-line case)
        best = None
        for i in (0, 1):
            if i >= len(mv.roadEdges) or i >= len(stds) or stds[i] > EDGE_STD_MAX:
                continue
            ed = mv.roadEdges[i]
            y0 = _interp_y(list(ed.x), list(ed.y), 0.0)
            if y0 is None or (y0 * side) <= 0:
                continue
            if best is None or abs(y0) < best[0]:
                best = (abs(y0), ed)
        if best is not None:
            bx, by = list(best[1].x), list(best[1].y)
            use_edge = 1

    if bx is None or len(bx) < 2:
        return None, None, use_edge

    worst = None
    for x in [v for v in px if 0.0 <= v <= CUTTING_X]:
        pv = _interp_y(px, py, x)
        bv = _interp_y(bx, by, x)
        if pv is None or bv is None:
            continue
        c = side * (bv - pv)          # positive => path inside the boundary
        worst = c if worst is None else min(worst, c)
    p0 = _interp_y(px, py, 0.0)
    b0 = _interp_y(bx, by, 0.0)
    c0 = side * (b0 - p0) if (p0 is not None and b0 is not None) else None
    return worst, c0, use_edge


def load(path):
    s = Seg(name=os.path.basename(os.path.dirname(path)))
    cp = CANParser(DBC, [("LKAS", 100)], 0)
    cur = dict(v=0.0, angle=0.0, eps=0.0, pressed=False, fault=False,
               asked=0.0, lat_active=False, des=0.0, act=0.0, sat=False)
    t0 = None
    for m in LogReader(path):
        w = m.which()
        mono = m.logMonoTime
        if w == "carState":
            cs = m.carState
            cur.update(v=cs.vEgo, angle=cs.steeringAngleDeg, eps=cs.steeringTorqueEps,
                       pressed=bool(cs.steeringPressed), fault=bool(cs.steerFaultTemporary))
        elif w == "carControl":
            cc = m.carControl
            cur.update(asked=cc.actuators.torque, lat_active=bool(cc.latActive))
        elif w == "controlsState":
            lcs = m.controlsState.lateralControlState
            if lcs.which() == "torqueState":
                ts = lcs.torqueState
                cur.update(des=ts.desiredLateralAccel, act=ts.actualLateralAccel,
                           sat=bool(ts.saturated))
        elif w == "carOutput":
            if t0 is None:
                t0 = mono
            s.t.append((mono - t0) / 1e9)
            s.v.append(cur["v"]); s.asked.append(cur["asked"])
            s.got.append(m.carOutput.actuatorsOutput.torque)
            s.lat_active.append(cur["lat_active"]); s.pressed.append(cur["pressed"])
            s.angle.append(cur["angle"]); s.eps.append(cur["eps"])
            s.fault_tmp.append(cur["fault"]); s.des_acc.append(cur["des"])
            s.act_acc.append(cur["act"]); s.saturated.append(cur["sat"])
        elif w == "modelV2":
            if t0 is None:
                continue
            mv = m.modelV2
            cmin, c0, edge = _clearance(mv)
            if cmin is not None:
                s.mt.append((mono - t0) / 1e9)
                s.mcurv.append(mv.action.desiredCurvature)
                s.clear_min.append(cmin)
                s.clear_x0.append(c0 if c0 is not None else float("nan"))
                s.bound_edge.append(edge)
        elif w == "sendcan":
            if t0 is None:
                continue
            cp.update([mono, [(c.address, bytes(c.dat), c.src) for c in m.sendcan]])
            v = cp.vl["LKAS"]
            s.act_idx.append(len(s.t) - 1)
            s.act_val.append(float(v["ActToiSta"]))
            s.act_tq.append(float(v["StrTqReqVal"]))
    _find_cuts(s)
    return s


def _find_cuts(s):
    """Identify genuine >85 deg steer-request cuts in the ActToiSta series.

    common_fault_avoidance drops the actuation bit for two frames while KEEPING
    lateral control active and holding the torque value. A lateral disengagement
    also drops the bit -- but latActive goes false with it. Counting every falling
    edge conflates the two, so require lateral control to stay active across the
    drop. Without this the metric reports disengagements as EPS faults.
    """
    la = s.lat_active
    n = len(la)
    av, ai = s.act_val, s.act_idx
    for k in range(1, len(av)):
        if not (av[k - 1] == 1.0 and av[k] == 0.0):
            continue
        i = ai[k]
        if i < 1 or i >= n:
            continue
        # how long the bit stays low
        j = k
        while j < len(av) and av[j] == 0.0:
            j += 1
        dur = j - k
        end_i = ai[min(j, len(ai) - 1)]
        # lateral must remain active across the whole drop: otherwise it is a
        # disengagement, not a fault-avoidance cut
        if not all(la[max(0, i - 1):min(n, end_i + 2)]):
            continue
        s.cut_idx.append(i)
        s.cut_angle.append(s.angle[i])
        s.cut_len.append(dur)
        # torque should be HELD (non-zero) through the cut, not zeroed
        s.cut_tq_held.append(1 if abs(s.act_tq[k]) > 1e-6 else 0)
        # EPS response: how far steeringTorqueEps moves in the 15 frames after
        w0, w1 = i, min(n, i + 15)
        if w1 > w0 + 1:
            seg = s.eps[w0:w1]
            s.cut_eps_dip.append(round(float(max(seg) - min(seg)), 2))
        else:
            s.cut_eps_dip.append(0.0)


def _reversals(t, tq, mask):
    """Time from a + torque peak to the following - peak, in seconds.

    A torque reversal is what an S-turn demands, and at stock rates it is the
    slowest thing in the loop - the reason this metric exists.

    Two bends only form a reversal if the frames BETWEEN them are also in the
    band. A speed band has holes (the car crosses the band edge, or lateral
    control disengages), and pairing across a hole would silently count the time
    spent outside the band as reversal duration -- e.g. a low-speed bend, forty
    seconds at 30 mph, then another low-speed bend would be recorded as a single
    40+ second sub-15 mph reversal.

    Returns (durations, pairs) where pairs are the contiguous (a, b) bend pairs
    that produced them, so callers measuring "the second bend" use the same
    filtered set rather than re-deriving it.
    """
    bends = []  # (sign, peak_idx, peak_t, start_idx, end_idx)
    i, n = 0, len(tq)
    while i < n:
        if not mask[i] or abs(tq[i]) < REVERSAL_TQ:
            i += 1
            continue
        sgn = 1 if tq[i] > 0 else -1
        j = i
        best, bi = abs(tq[i]), i
        while j < n and mask[j] and (tq[j] * sgn) > 0:
            if abs(tq[j]) > best:
                best, bi = abs(tq[j]), j
            j += 1
        if best >= REVERSAL_TQ:
            bends.append((sgn, bi, t[bi], i, j - 1))
        i = max(j, i + 1)

    out, trans, pairs = [], [], []
    for a, b in zip(bends, bends[1:]):
        if a[0] == b[0]:
            continue
        # every frame between the two bends must still be in the band
        gap_lo, gap_hi = a[4], b[3]
        if gap_hi > gap_lo and not all(mask[gap_lo:gap_hi + 1]):
            continue
        out.append(b[2] - a[2])          # peak-to-peak, as the plan specifies
        trans.append(t[b[3]] - t[a[4]])  # transit only: bend exit -> opposite bend entry
        pairs.append((a, b))
    return out, trans, pairs


def metrics(s):
    """Per-band and whole-segment metrics. Only frames with lateral control active
    are counted -- the limiter does nothing when openpilot is not steering."""
    t = np.array(s.t); v = np.array(s.v)
    asked = np.array(s.asked); got = np.array(s.got)
    active = np.array(s.lat_active, dtype=bool)
    rows = []
    pooled = {}
    if len(t) == 0:
        return rows, pooled

    diff = np.abs(asked - got)
    limited = (diff > LIMITED_EPS) & active
    cap = limited & (np.abs(got) >= CAP_FRAC)
    rate = limited & ~cap
    err = np.abs(np.array(s.des_acc) - np.array(s.act_acc))

    mt = np.array(s.mt) if s.mt else np.array([])

    for label, lo, hi in BANDS + [("all", 0.0, 1e9)]:
        band = active & (v >= lo) & (v < hi)
        n = int(band.sum())
        row = {"segment": s.name, "band": label, "n_frames": n,
               "seconds": round(n / 100.0, 1)}
        pool = pooled.setdefault(label, {"err": [], "clear": [], "edge": [],
                                         "n_frames": 0, "n_limited": 0,
                                         "n_cap": 0, "n_rate": 0, "revs": [], "trans": []})
        if n == 0:
            rows.append({**row, **{k: "" for k in (
                "frac_limited", "frac_cap_limited", "frac_rate_limited",
                "rate_share_of_limited", "p95_track_err", "peak_err_2nd_bend",
                "frac_saturated", "n_faults", "n_interventions",
                "n_reversals", "median_reversal_s", "median_transition_s", "max_abs_angle",
                "n_cuts", "min_clearance_m", "frac_edge_bound")}})
            continue
        pool["n_frames"] += n
        pool["n_limited"] += int(limited[band].sum())
        pool["n_cap"] += int(cap[band].sum())
        pool["n_rate"] += int(rate[band].sum())
        pool["err"].extend(err[band].tolist())
        row["frac_limited"] = round(float(limited[band].mean()), 4)
        row["frac_cap_limited"] = round(float(cap[band].mean()), 4)
        row["frac_rate_limited"] = round(float(rate[band].mean()), 4)
        nl = int(limited[band].sum())
        row["rate_share_of_limited"] = round(float(rate[band].sum()) / nl, 3) if nl else ""
        row["p95_track_err"] = round(float(np.percentile(err[band], 95)), 3)
        row["frac_saturated"] = round(float(np.array(s.saturated)[band].mean()), 4)
        row["n_faults"] = int(np.array(s.fault_tmp)[band].sum())
        row["n_interventions"] = int((np.array(s.pressed) & band).sum())
        row["max_abs_angle"] = round(float(np.abs(np.array(s.angle))[band].max()), 1)

        revs, trans, pairs = _reversals(t, got, band)
        pool["revs"].extend(revs)
        pool["trans"].extend(trans)
        row["n_reversals"] = len(revs)
        row["median_reversal_s"] = round(float(np.median(revs)), 3) if revs else ""
        row["median_transition_s"] = round(float(np.median(trans)), 3) if trans else ""

        # peak tracking error in the SECOND bend of each S-turn: the bend after a
        # reversal is where running wide shows up, because the rate limit has just
        # spent its budget crossing zero.
        peaks = []
        for _a, b in pairs:
            lo_i, hi_i = b[1], min(b[1] + 150, len(err))
            if hi_i > lo_i:
                peaks.append(float(err[lo_i:hi_i].max()))
        row["peak_err_2nd_bend"] = round(max(peaks), 3) if peaks else ""

        cuts_in = [i for i in s.cut_idx if i < len(band) and band[i]]
        row["n_cuts"] = len(cuts_in)

        if len(mt):
            # Map each 20 Hz model frame onto the nearest 100 Hz state frame and
            # test band membership THERE. Selecting by the band's [min, max] time
            # range instead would take the convex hull of the band: speed bands
            # are not contiguous (the car crosses 15/25/40 mph repeatedly within
            # one 60 s recording), so the hull sweeps in all the out-of-band time
            # between the first and last in-band frame, and drops the latActive
            # requirement that `band` carries.
            idx = np.clip(np.searchsorted(t, mt), 0, len(t) - 1)
            left = np.clip(idx - 1, 0, len(t) - 1)
            nearest = np.where(np.abs(mt - t[left]) < np.abs(t[idx] - mt), left, idx)
            msel = band[nearest] & (np.abs(mt - t[nearest]) <= 0.05)
            cm = np.array(s.clear_min)[msel]
            pool["clear"].extend(cm.tolist())
            row["min_clearance_m"] = round(float(cm.min()), 2) if len(cm) else ""
            be = np.array(s.bound_edge)[msel]
            pool["edge"].extend(be.tolist())
            row["frac_edge_bound"] = round(float(be.mean()), 3) if len(be) else ""
        else:
            row["min_clearance_m"] = ""
            row["frac_edge_bound"] = ""
        rows.append(row)
    return rows, pooled


def cut_report(segs):
    """Validate the >85 deg steer-request-cut mechanism (plan section 5)."""
    tot = ok = 0
    lens, held, dips = [], 0, []
    for s in segs:
        for ang, ln, h, d in zip(s.cut_angle, s.cut_len, s.cut_tq_held, s.cut_eps_dip):
            tot += 1
            if abs(ang) >= 85.0:
                ok += 1
            lens.append(ln)
            held += h
            dips.append(d)
    return tot, ok, lens, held, dips


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rlogs", nargs="+")
    ap.add_argument("--geofence", help="JSON of named lat/lon polygons (optional)")
    ap.add_argument("--out", default="lat_metrics_out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows, segs = [], []
    POOL = {}
    for i, p in enumerate(args.rlogs, 1):
        try:
            s = load(p)
        except Exception as e:
            print(f"  [{i}/{len(args.rlogs)}] {p}: FAILED {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        segs.append(s)
        r, pooled = metrics(s)
        all_rows += r
        for band, d in pooled.items():
            acc = POOL.setdefault(band, {"err": [], "clear": [], "edge": [],
                                         "n_frames": 0, "n_limited": 0,
                                         "n_cap": 0, "n_rate": 0, "revs": [], "trans": []})
            for k in ("err", "clear", "edge", "revs", "trans"):
                acc[k].extend(d[k])
            for k in ("n_frames", "n_limited", "n_cap", "n_rate"):
                acc[k] += d[k]
        print(f"  [{i}/{len(args.rlogs)}] {s.name}: {len(s.t)} frames, "
              f"{len(s.cut_idx)} cuts", file=sys.stderr)

    if not all_rows:
        print("no data", file=sys.stderr)
        return 1
    csv_path = os.path.join(args.out, "lat_metrics.csv")
    keys = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    md = os.path.join(args.out, "summary.md")
    tot, ok, lens, held, dips = cut_report(segs)
    with open(md, "w") as fh:
        fh.write("# Lateral limit baseline\n\n")
        fh.write(f"Segments: {len(segs)}\n\n")
        fh.write("## Which limit binds, by speed band\n\n")
        fh.write("All figures are POOLED over segments, not averages of per-segment\n"
                 "values: a frame-weighted mean of per-segment p95s is not the p95 of\n"
                 "the combined sample, and a rate share must be weighted by limited\n"
                 "frames rather than total frames.\n\n")
        fh.write("| band | frames | frac limited | cap | rate | rate share | "
                 "p95 err | 2nd-bend peak | reversals | median rev s | median transit s |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for label, _, _ in BANDS + [("all", 0, 0)]:
            d = POOL.get(label)
            if not d or not d["n_frames"]:
                continue
            nf = d["n_frames"]
            rs = [r for r in all_rows if r["band"] == label and r["n_frames"]]
            pk = [float(r["peak_err_2nd_bend"]) for r in rs
                  if r["peak_err_2nd_bend"] != ""]
            fh.write(
                f"| {label} | {nf} | {d['n_limited']/nf:.3f} | "
                f"{d['n_cap']/nf:.3f} | {d['n_rate']/nf:.3f} | "
                f"{(d['n_rate']/d['n_limited'] if d['n_limited'] else 0):.2f} | "
                f"{(np.percentile(d['err'], 95) if d['err'] else 0):.3f} | "
                f"{(max(pk) if pk else 0):.3f} | {len(d['revs'])} | "
                f"{(np.median(d['revs']) if d['revs'] else 0):.2f} | "
                f"{(np.median(d['trans']) if d['trans'] else 0):.2f} |\n")

        fh.write("\n## Inside-cutting clearance (pooled)\n\n")
        fh.write("| band | curve frames | median | p5 | worst | frames crossing | "
                 "road edge used |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for label, _, _ in BANDS:
            d = POOL.get(label)
            if not d or not d["clear"]:
                continue
            c = np.array(d["clear"])
            e = np.array(d["edge"]) if d["edge"] else np.array([0.0])
            fh.write(f"| {label} | {len(c)} | {np.median(c):+.2f} m | "
                     f"{np.percentile(c, 5):+.2f} m | {c.min():+.2f} m | "
                     f"{int((c < 0).sum())} | {e.mean():.0%} |\n")
        fh.write(f"\n## Steer-request cuts (plan section 5)\n\n")
        fh.write(f"- cut events: {tot}\n")
        fh.write(f"- at |steeringAngleDeg| >= 85: {ok}/{tot}\n")
        fh.write(f"- cut durations (frames): {dict(collections.Counter(lens))}\n")
        fh.write(f"- torque value held (not zeroed) through cut: {held}/{tot}\n")
        if dips:
            fh.write(f"- steeringTorqueEps swing in 150 ms after cut: "
                     f"median {float(np.median(dips)):.1f}, max {max(dips):.1f}\n")
        fh.write(f"- ACCEPTANCE {'PASS' if tot == ok and tot > 0 else 'FAIL'}: "
                 f"cuts occur only above 85 deg\n")
    print(f"wrote {csv_path}\nwrote {md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
