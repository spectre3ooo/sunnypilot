# Upstreamable contributions found along the way

A living list. Work on this branch is scoped to one Ioniq 5, but some of it is
generic and would stand on its own merits upstream — to sunnypilot, and in a couple
of cases to comma. Kept updated as we find things.

Each entry records **what**, **why it is generic**, **evidence**, and **status**.
Status is one of: `verified` (we ran it), `observed` (we saw it but have not proved
it), `speculative` (idea only).

Last updated: 2026-09-07.

---

## 1. Speed-scheduled torque rate limit in panda — `verified`

**What.** `TorqueSteeringLimits` gains `dynamic_max_rate_up` + `max_rate_up_lookup`,
and `steer_torque_cmd_checks` interpolates the rate on speed, clamped by the existing
scalar as a hard ceiling so a schedule can only ever tighten.

**Why generic.** panda already schedules the *angle* rate on speed
(`angle_rate_up_lookup` / `angle_rate_down_lookup` in `AngleSteeringLimits`) and the
torque *cap* on speed for Rivian (`dynamic_max_torque`). Torque *rate* on speed is the
missing third case, built from the same `lookup_t` and `safety_interpolate`. No brand
is affected unless it opts in.

**Evidence.** T1.0. Gates run on the branch.

**Two findings worth carrying upstream with it:**

- **`max_rate_down` must NOT be scheduled.** In `driver_limit_check`,
  `MAX_RATE_DOWN` is a *minimum required wind-down*, not an allowance:
  `highest_allowed = SAFETY_MIN(highest_allowed_rl, SAFETY_MAX(val_last - MAX_RATE_DOWN, ...))`
  is monotonically **decreasing** in it, so a larger value is stricter. Taking
  `SAFETY_MIN(schedule, scalar)` on rate-down would **loosen** a safety mandate. The
  original plan asked for both lookups; shipping only rate-up is the correct deviation.
- **panda's rate-down is not what makes reversals slow.** With `val_last = +270`,
  `lowest_allowed_rl = SAFETY_MIN(val_last, 0) - MAX_RATE_UP = -MAX_RATE_UP`, so panda
  permits an immediate drop from full positive torque to just below zero. The 2.25 s
  reversal is imposed by openpilot's Python limiter
  (`apply_driver_steer_torque_limits`), not by panda. Anyone reasoning about reversal
  speed from panda's constants alone will get this wrong.

---

## 2. Per-car torque envelope derivation — `speculative`

**What.** Generalise `derive_limits.py` (T1.6) into a car-agnostic tool: feed it logs,
get a speed-scheduled torque envelope plus a written justification page.

**Why generic.** comma's current porting guidance is "find the maximum value that the
stock LKAS will request" — once, at one speed, by hand. Doing it per speed bin,
automatically, against four explicit bounds (need / EPS acceptance / overridability /
ISO hazard) with a plot per number is strictly more defensible, and it is exactly what
a fork policy demanding written derivations wants to see.

**Feasibility note.** Three of the four bounds are automatable from data already being
collected: *need* from what `torqued` learns, *EPS acceptance* passively from
`StrTqReqVal` while stock LKAS steers, *hazard* by simulation once the plant is fitted.
**Overridability is the holdout** — it needs a human physically pushing the wheel, and
it is safety-critical, so it cannot be inferred from incidental intervention data
without a lot of validation.

**Sequencing.** Must prove itself on one car first. Generalising before the Ioniq 5
envelope is derived risks baking in assumptions that turn out to be wrong on the only
car we can test.

---

## 3. `torqued` below 15 m/s — `speculative`

See `docs/lat/torqued-lowspeed.md`. Gated on T1.6 answering whether `K(v)` varies
enough below 15 m/s to be worth learning online, and on T0.6 showing whether NNLC
already covers it.

---

## 4. `setup_dependencies.sh` misses two build requirements — `verified`

**What.** On a clean Ubuntu 24.04 box, `tools/op.sh setup` completes but the tree
cannot be built or fully checked out.

- **`git-lfs`** is not in the apt list, yet `op.sh` runs `git lfs pull` (line ~225)
  and *checks* for LFS files (line ~105). The pull fails silently and the run ends with
  "git lfs files not found!". 9 LFS patterns including `*.onnx` stay as pointer stubs.
- **`clang`** is not in the apt list, yet `scons` hard-requires it: tinygrad shells out
  to `clang` to compile the driving model, failing at
  `driving_tinygrad.pkl.chunkmanifest` with `FileNotFoundError: 'clang'`.

The installed list is `ca-certificates build-essential curl libcurl4-openssl-dev
locales git xclip wl-clipboard`.

**Why generic.** Affects any fresh Ubuntu checkout, not this branch.

**Evidence.** Hit both on devbook; both resolved by `apt-get install`. `verified`.

---

## 5. `process_replay` does not cover any sunnypilot SP message — `verified`

**What.** controlsd's `ProcessConfig` in
`openpilot/selfdrive/test/process_replay/process_replay.py` has
`subs=["carControl", "controlsState"]`. Neither `carControlSP` nor the new
`lateralControlSP` is captured, so a green `process_replay` says nothing about whether
those messages are correct — only that publishing them did not perturb the two
messages that *are* compared.

**Why generic.** Applies to every SP message sunnypilot publishes from controlsd, not
just ours.

**Caveat.** Adding them to `subs` requires regenerating reference logs
(`--update-refs`), which we deliberately did not do on a fork. Someone with upstream
standing should make that call.

**Evidence.** T0.5. `verified` by reading the config and confirming a passing run.

---

## 6. `mutation.py` may schedule tests that never run — `observed`, unverified

**What.** In `_discover_test_catalog`:

```python
catalog[test_file.name] = [t.id() for group in suite for t in group]
```

This appears to include abstract base classes (`...Base`) and `@parameterized_class`
templates carrying `__test__ = False`, both of which are skipped at runtime. If a
mutant is scheduled only against those, nothing actually executes and the mutant may be
scored as covered when it was never tested — under-testing mutants in the shared
non-mode files.

**Status.** Surfaced by a review agent working on T1.0 and **not independently
verified by us**. It was in a design variant we discarded, and we did not adopt the
change: modifying the mutation tester inside a safety commit is exactly what an
upstream reviewer should object to. Recorded here so it is not lost. Worth confirming
separately and, if real, fixing in its own commit.

---

## 7. `lat_metrics.py` as general lateral-limit analysis — `speculative`

**What.** `openpilot/tools/lat_metrics/lat_metrics.py` answers "is this car limited by
its torque cap or its rate limit, per speed band" from ordinary rlogs, with no new
telemetry, for any torque-controlled car.

**Why generic.** The cap-vs-rate question is the first thing anyone tuning lateral
control on any car should ask, and today it is answered by feel. Nothing in the tool is
Ioniq-5-specific except the DBC used for the 0x50 decode.

**Before offering it:** two metric defects were found by adversarial review after the
first version shipped (band selection by time hull; reversals paired across
out-of-band gaps). Both are fixed and pinned by regression tests. A third — reversal
time being peak-to-peak, so it includes dwell — is documented but **not** fixed, and
should be before this is offered to anyone.
