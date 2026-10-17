# Steering-angle feedforward (`LatAngleFeedforward`)

Opt-in, off by default. Code: `openpilot/sunnypilot/selfdrive/controls/lib/angle_ff/`. The fit tool is
`openpilot/tools/lat_metrics/angle_ff_fit.py`, and the dataset comes from `steer_dataset.py`. Tests:
`sunnypilot/selfdrive/controls/tests/test_angle_ff.py`.

## Why

At low speed, the torque that holds and moves the wheel follows **steering angle**, not lateral acceleration. The
steering geometry's return force (plus the EPS's own active return) grows with angle, while lateral accel stays
small (speed² × curvature). The torque controller converts lateral accel to torque. So at low speed it under-asks,
and its high-gain feedback (KP up to 250 near standstill) makes up the rest, slamming the torque toward centre on
corner exits. From the Ioniq 5's logs (931 rlogs, 15 h, 53 min of manual driving):

- Hold torque grows with angle *and* speed. Steady engaged holds reach the 270 cap at 25–45 mph past 45° and at
  5–15 mph past 90°.
- With the wheel nearly free, natural return speed is 1–36 °/s below 5 mph and up to 138–201 °/s at 5–25 mph past
  90°.
- The EPS adds torque of its own: active return toward centre (−1 to −4.5 Nm when the driver lets go at low
  speed) and damping (about −1.4 Nm against motion).
- Calibration: 270 counts ≈ 14 Nm (0.052 Nm/count near the top of the range). The driver sensor reads 0.0047 Nm
  per unit, from matching engaged and manual holds of the same state (the DBC note says 0.005).

## Model

    torque = sign(angle)·hold(v, |angle|) + damping(v)·rate + friction(v)·sign(rate) + gain(v)·lat_accel + bias

`hold` is a bilinear table over 10 speed × 10 |angle| nodes, and the other terms are linear over the speed
nodes. It is fitted per car from two kinds of frames that reveal the car rather than the controller:

- **steady engaged holds:** the command that keeps the wheel still is what the car needs there;
- **manual driving:** hands plus EPS assist, converted to command counts, minus the torque the EPS adds by
  itself when it is the one being commanded (fitted as a residual model), so that torque isn't counted twice.

A second-difference smoothness penalty lets sparse cells borrow from their neighbours. Its strength is chosen on
held-out routes.

Two kinds of frames are left out:

- **within 25° of full lock** (lock is taken from the data, about ±488° on the Ioniq 5): holding the wheel against
  the stop says nothing about the torque the angle needs.
- **engaged holds while the angle feedforward was on** (`aff_w` column, from `lateralControlSP.angleFfWeight`):
  a still wheel is held by any torque inside the friction band, so those holds would teach the model its own
  output. They still feed the EPS-own model. Without this rule, one drive with the feature on pulled the
  driver-sensor scale from 0.0048 to 0.0032 Nm/unit.

**Held-out fit (Ioniq 5, refit 2026-10-16 with the first arm C drive and its parking-lot circles, 43 routes):**
R² 0.94 on manual driving from routes the fit never saw. Error by speed band, in counts: 59 below 5 mph, then 37,
25, 19 and 15 above 45 mph. On unwinds the fitted model pushes the wheel back toward centre in 44 % of frames;
your manual driving does so in 46 %. The first fit (40 routes) scored R² 0.93. For comparison, against the same target the
current lateral-accel feedforward scores R² −1.0, the latAccelFactor schedule −1.75, and NNLC −0.69.

A small neural net with the same inputs and angle history reached R² 0.94. With NNLC's inputs alone, retrained
on this car, it reached only 0.45. **Steering angle is the missing input; the network's extra flexibility adds
little.** With only 53 minutes of manual data, the inspectable table is the safer first step.

## Sign conventions (checked in the data, pinned by tests)

- `steeringAngleDeg` and the actuator torque share a sign. The controller's lateral accel has the opposite sign
  (controlsd measures curvature as `-VM.calc_curvature(angle)`).
- Desired angle: `degrees(VM.get_steer_from_curvature(-desired_curvature, v, roll)) + angleOffsetDeg`, so road
  roll is included.
- The model's output is in actuator convention, and is negated before it is added to the controller's
  pre-negation output torque.

## Runtime

Below 30 mph the angle model replaces the lateral-accel feedforward entirely. Between 30 and 45 mph it fades back
to it, and above 45 mph the controller is bit-for-bit unchanged. The model was fitted at every speed, but its
highway torque is about 20–30 % below the current controller's, so the first on-car test keeps the proven highway
behaviour. PID feedback is unchanged. It is skipped when NNLC is active or no model exists for the car. Every
frame logs `lateralControlSP.angleFfWeight` and `angleFfTorque`.

**The rate terms are off at runtime** (`USE_RATE_TERM = False`). Recorded drives were replayed through the real
controller with process replay, feeding `selfdriveStateSP` and `lateralDelay`, which stock controlsd replay
omits; with the feature off, the replay reproduces the logged torque at correlation 0.98. On the 10 segments with
the most engaged turning below 25 mph:

| variant | reverse torque on unwind | step size p50 / p95 / p99 (counts) | at the steer_max clip |
|---|---|---|---|
| off (current controller) | 52 % | 0.54 / 26.3 / 60.4 | 23.5 % |
| on, rate terms driven by the planned angle | 58 % | 1.17 / 37.9 / 95.8 | 24.5 % |
| on, 1 Hz filter on the desired angle | 57 % | 0.74 / 21.2 / 48.7 | 23.9 % |
| **on, no rate terms (shipped)** | **45 %** | **0.42 / 22.1 / 51.7** | **18.7 %** |
| on, 1 Hz filter + no rate terms | 43 % | 0.69 / 20.1 / 48.9 | 18.2 % |

Driving the damping term from the planned angle rate pushes the wheel toward centre on every planned unwind, on
top of the return the EPS and geometry already give, and passes planner noise straight into torque steps. The
1 Hz filter is marginally smoother but adds about 0.16 s of lag on top of the steering's own 0.25–0.37 s.

**Limits of that replay:** it is open-loop, because the logged wheel doesn't respond to the new commands. It shows
what would be requested, not what the car would do. The expected closed-loop benefit, less work for the high-gain
feedback when the feedforward is right, only shows on the road.

## Online learning (not built)

A torqued-style learner using only steady engaged holds was replayed over the 15 hours in drive order. Cells near
centre at normal speeds settle within 2–35 minutes, like torqued. **The low-speed, large-angle cells, which matter
most, barely fill**: several had data from only 0–4 routes, because tight low-speed turns are dynamic, not steady.
An online version therefore has to learn from moving frames too, by fitting the whole model, or use a short
calibration drive (parking-lot circles) to fill those cells.

## Refit

    python3 openpilot/tools/lat_metrics/steer_dataset.py /tmp/ds ~/sunnypilot-experiments/logs/devices/ioniq5 ~/sunnypilot-experiments/logs/rlogs
    python3 openpilot/tools/lat_metrics/angle_ff_fit.py /tmp/ds --fingerprint HYUNDAI_IONIQ_5

## First drive (route `00000044`, night, arm C)

Against official dev and the rate-schedule build, overrides per engaged minute dropped to 1.4 at 15–25 mph
(5.6 and 15.3 before) and 1.7 at 25–45 mph (3.2 and 7.8), with about half the driver torque. Below 15 mph there
was no gain, and the request sat at the steer_max clip 24.5 % of the time. The parking-lot circles show why: at
250–400° below 15 mph the model asks about 178 counts where manual driving needed about 122. The refit didn't
close that gap, because one lot session is outweighed by the older data there.

The driver felt the car hug the right side. Lane lines say that comes from the plan, not from this controller:
the model's path cuts the inside of right curves by 0.3–0.7 m on every build, dev included, and the car follows
the plan to within a few centimetres. Arm C sat about 0.1 m further right than the plan on 30–45 mph straights,
which is within the noise of one night drive.

## Open

- The PID's low-speed gains are unchanged. With a better feedforward they could come down, which needs on-road A/B.
- The 270 cap still binds on about 19 % of low-speed requests. Raising it needs an EPS-acceptance test.
- More manual data (baseline drives) would firm up the low-speed cells and allow a fair NN comparison.
