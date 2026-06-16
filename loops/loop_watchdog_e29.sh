#!/usr/bin/env bash
# loop_watchdog_e29.sh — week-long SELF-RECOVERY for the Effort-29 CUDA-kernels driver.
# Restarts the driver on EXIT (finished MAX_CYCLES / crash) OR HANG (stale tee-log — a
# wedged `claude -p` blocks the driver without it exiting). On each restart it cleans the
# worktree to the last COMMITTED branch state (a hung/dirty cycle is discarded; committed+
# pushed wins are kept — so the loop never corrupts itself). Kills are SCOPED to e29 (the
# branch name is in the agent's prompt) so e27/e26 loops are never touched. Auto-stops
# after ~8 days. Start:
#   nohup bash /Users/stepan/Workspace/zinc-e29/loops/loop_watchdog_e29.sh >/tmp/e29_watchdog.log 2>&1 &
# Stop:  pkill -f loop_watchdog_e29   (then: kill the driver + pkill -f 'perf/e29-cuda-kernels')
set -u
WORKTREE=/Users/stepan/Workspace/zinc-e29
BRANCH=perf/e29-cuda-kernels
DRIVER=run_perf_effort_e29.sh
EFFORT=loops/efforts/MULTI_HOUR_EFFORT_29_CUDA_KERNEL_EFFICIENCY.md
LOG=/tmp/perf_effort_MULTI_HOUR_EFFORT_29_CUDA_KERNEL_EFFICIENCY.log
STALE_SECS=${STALE_SECS:-2700}     # 45m — keep > the slowest legit silent cycle (gemma reload + A/B)
CHECK_SECS=${CHECK_SECS:-300}
DEADLINE=$(( $(date +%s) + 8*86400 ))   # auto-stop after ~8 days
wlog(){ echo "[e29-watchdog $(date '+%m-%d %H:%M:%S')] $*"; }
start_driver(){
  ( cd "$WORKTREE" || exit 1
    git fetch -q origin 2>/dev/null
    git checkout -f "$BRANCH" 2>/dev/null
    git reset --hard "$BRANCH" 2>/dev/null
    git clean -fdq 2>/dev/null
    exec bash "loops/$DRIVER" "$EFFORT" ) &
  echo $!
}
PID=$(start_driver)
wlog "started driver pid $PID (worktree $WORKTREE, branch $BRANCH, stale>${STALE_SECS}s, deadline +8d)"
while true; do
  sleep "$CHECK_SECS"
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    wlog "8-day deadline reached — stopping driver + watchdog"
    kill -9 "$PID" 2>/dev/null; pkill -9 -f "perf/e29-cuda-kernels" 2>/dev/null
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    wlog "driver exited (MAX_CYCLES or crash) — restarting"
    PID=$(start_driver); wlog "restarted driver pid $PID"; continue
  fi
  if [ -f "$LOG" ]; then
    now=$(date +%s); m=$(stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo "$now")
    age=$(( now - m ))
    if [ "$age" -gt "$STALE_SECS" ]; then
      wlog "HANG: tee-log stale ${age}s (> ${STALE_SECS}) — killing e29 driver+agent, restarting"
      kill -9 "$PID" 2>/dev/null
      pkill -9 -f "perf/e29-cuda-kernels" 2>/dev/null   # scoped: only the e29 claude -p agent
      sleep 5
      PID=$(start_driver); wlog "restarted driver pid $PID"
    fi
  fi
done
