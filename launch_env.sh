#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="19.7"
fi

export STAGING_ROOT="/data/safe_staging"

# Start Tailscale installed by spectre3ooo/tailscale-comma3x. Tracked here because the installer's
# own hook is appended to this file, and every openpilot update resets it.
if [ -x /data/tailscale/start_tailscale.sh ] && ! pgrep -x tailscaled > /dev/null; then
  sudo -n /data/tailscale/start_tailscale.sh &
fi
