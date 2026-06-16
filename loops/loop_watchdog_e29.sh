#!/usr/bin/env bash
# loop_watchdog_e29.sh — week-long SELF-RECOVERY for the Effort-29 CUDA-kernels driver.
# Restarts the driver on EXIT (finished/crash) OR HANG (driver alive but its tee-log
# stale > STALE_SECS — a wedged `claude -p` blocks the driver without it exiting). On
# restart it cleans the worktree to the last COMMITTED+PUSHED branch state (hung/dirty
# cycle discarded; committed wins kept). Kills SCOPED to e29 (branch string in the agent
# prompt) so e27/e26 untouched. Auto-stops ~8 days. LOGS every check (with the log age)
# to /tmp/e29_watchdog.log so its behaviour is verifiable.
#   nohup bash /Users/stepan/Workspace/zinc-e29/loops/loop_watchdog_e29.sh >/dev/null 2>&1 &
# Stop:  pkill -f loop_watchdog_e29 ; then pkill -f run_perf_effort_e29 ; pkill -f 'perf/e29-cuda-kernels'
set -u
WORKTREE=/Users/stepan/Workspace/zinc-e29
BRANCH=perf/e29-cuda-kernels
DRIVER=loops/run_perf_effort_e29.sh
EFFORT=loops/efforts/MULTI_HOUR_EFFORT_29_CUDA_KERNEL_EFFICIENCY.md
LOG=/tmp/perf_effort_MULTI_HOUR_EFFORT_29_CUDA_KERNEL_EFFICIENCY.log
WLOG=/tmp/e29_watchdog.log
STALE_SECS=${STALE_SECS:-5400}     # 90m — > the slowest legit cycle (full-catalog commit gate ~68m)
CHECK_SECS=${CHECK_SECS:-120}      # check every 2m
DEADLINE=$(( $(date +%s) + 8*86400 ))
DRIVER_PID=0
wlog(){ printf '[e29-wd %s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" >>"$WLOG"; }
start_driver(){
  ( cd "$WORKTREE" 2>/dev/null || exit 1
    git fetch -q origin 2>/dev/null
    git checkout -f "$BRANCH" 2>/dev/null
    git reset --hard "origin/$BRANCH" 2>/dev/null
    git clean -fdq 2>/dev/null
    exec bash "$DRIVER" "$EFFORT" ) >/dev/null 2>&1 &
  DRIVER_PID=$!
  wlog "started driver pid $DRIVER_PID (worktree reset to origin/$BRANCH)"
}
wlog "=== e29 watchdog up (stale>${STALE_SECS}s, check ${CHECK_SECS}s, deadline +8d) ==="
start_driver
while true; do
  sleep "$CHECK_SECS"
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    wlog "8-day deadline reached — stopping"
    kill -9 "$DRIVER_PID" 2>/dev/null; pkill -9 -f run_perf_effort_e29 2>/dev/null; pkill -9 -f "perf/e29-cuda-kernels" 2>/dev/null
    exit 0
  fi
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    wlog "driver pid $DRIVER_PID exited — restarting"
    start_driver; continue
  fi
  now=$(date +%s)
  m=$(stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  age=$(( now - m ))
  wlog "check: driver $DRIVER_PID alive, tee-log age ${age}s"
  if [ "$age" -gt "$STALE_SECS" ]; then
    wlog "HANG detected: tee-log stale ${age}s > ${STALE_SECS}s — killing e29 driver+agent, restarting"
    kill -9 "$DRIVER_PID" 2>/dev/null
    pkill -9 -f run_perf_effort_e29 2>/dev/null
    pkill -9 -f "perf/e29-cuda-kernels" 2>/dev/null
    sleep 5
    start_driver
  fi
done
