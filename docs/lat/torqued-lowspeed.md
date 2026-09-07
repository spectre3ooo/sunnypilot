# Sketch: extending `torqued` below 15 m/s

Status: **design sketch, not implemented.** Sequencing argument at the end — this
should almost certainly wait for T1.6's offline fit.

## What `torqued` does today

`openpilot/selfdrive/locationd/torqued.py` learns two scalars online and publishes
them as `liveTorqueParameters`:

- `latAccelFactor` — the slope of lateral accel vs. commanded steer torque
- `friction` — from the standard deviation of the residual spread, × `FRICTION_FACTOR` (1.5)

It collects points into 8 buckets over steer ∈ [−0.5, 0.5] (`STEER_BUCKET_BOUNDS`,
`MIN_BUCKET_POINTS = [100, 300, 500, 500, 500, 500, 300, 100]`) and fits **one**
line across all of them.

## The two gates that exclude our regime

From `handle_log`, a point is only collected when:

```python
if all(lat_active) and not any(steer_override) and (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD):
  if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
```

with `MIN_VEL = 15` m/s (33.5 mph) and `LAT_ACC_THRESHOLD = 1` m/s².

**Both gates exclude exactly the regime this project cares about.** Tight corners
below 40 mph are under 15 m/s *and* above 1 m/s² lateral accel. So the controller
extrapolates a highway, low-lateral-accel linear fit into a low-speed, high-lateral-accel
regime it has never observed. That is a plausible contributor to the tracking error
`docs/lat/baseline.md` measures (pooled p95 0.733 m/s² below 15 mph vs 0.221 above 40).

Those gates are not arbitrary, and anyone changing them should know why they exist:

- **`MIN_VEL`** — lateral accel is derived from vehicle pose. At low speed the signal
  shrinks while sensor noise does not, so SNR degrades. Low-speed data is also
  contaminated by parking manoeuvres with large angles and dynamics that do not
  represent driving.
- **`LAT_ACC_THRESHOLD`** — keeps the fit inside the linear region. Above ~1 m/s²
  the tire/EPS relationship bends, and a single slope stops describing it.

Relaxing either one weakens an assumption the current estimator depends on.

## Minimum viable change

1. **Bucket by speed as well as steer.** Keep `PointBuckets`, add a speed dimension:
   bands at roughly <8, 8–15, 15–25, >25 m/s. Fit each band independently to get
   `K(v)` and `friction(v)` instead of two scalars. This reuses the existing
   structure rather than replacing it.
2. **Raise `LAT_ACC_THRESHOLD` per band**, or drop it in favour of a nonlinearity
   check. It cannot stay at 1 m/s² for a band whose whole purpose is 2–3 m/s² corners.
3. **Per-band sanity bounds.** `FACTOR_SANITY = 0.3` allows only ±30 % deviation from
   the `CarParams` default. If low-speed `K` genuinely differs by more than that, the
   sanity check silently clamps it and the estimate is a lie. Bounds must widen per
   band, which means the sanity check is no longer protecting what it used to.
4. **Bump `VERSION`** (currently 1) to invalidate cached parameters.
5. **New message shape.** `liveTorqueParameters` carries scalars. A schedule needs a
   list or a new struct — a cereal change, and a consumer change in `LatControlTorque`,
   which reads `latAccelFactor` as a scalar today.

## Risks

- **It changes the controller's model of the car at low speed.** Not the safety layer,
  but directly upstream of torque requests. Must be param-gated and validated with the
  same ABAB methodology as everything else in plan §8.1.
- **Sanity-bound widening is the dangerous part.** A bad low-speed fit that used to be
  clamped would now pass through. Consider making low-speed bands *advisory* first —
  logged but not consumed — until the offline fit confirms they agree.
- **NNLC may make it redundant.** sunnypilot's neural feedforward already takes speed
  as an input and is trained on fleet data, including low speed. If NNLC is enabled
  (T0.6), a hand-rolled speed-scheduled linear fit may add nothing. **T0.6 should be
  read before this work starts.**

## Sequencing: do T1.6 first

T1.6 already performs this fit offline, with more data per point, no real-time budget,
and the ability to iterate. It will answer the question that decides whether any of
this is worth doing:

> Does `K(v)` actually vary enough below 15 m/s to matter?

If it varies little, extending `torqued` buys nothing and the envelope work alone is
the answer. If it varies a lot, T1.6 gives ground truth to validate an online learner
against — which is the only honest way to know the online version works.

Building the online learner first means shipping an estimator with nothing to check it
against, in a regime chosen precisely because nobody has measured it.
