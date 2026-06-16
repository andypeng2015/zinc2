#!/usr/bin/env bash
# Effort 29 (CUDA kernel efficiency — close the llama gap) autonomous driver.
# 5090-pinned, accumulates on perf/e29-cuda-kernels, ONE validated kernel increment
# per cycle. Wrapped by loop_watchdog_e29.sh for week-long self-recovery (restart on
# exit/hang). Each cycle spawns a fresh `claude -p` that reads the effort file + memory,
# lands one validated increment, commits to the branch + pushes, then stops.
#   bash loops/run_perf_effort_e29.sh loops/efforts/MULTI_HOUR_EFFORT_29_CUDA_KERNEL_EFFICIENCY.md
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
EFFORT="${1:?usage: run_perf_effort_e29.sh <effort-file.md>}"
[ -f "$EFFORT" ] || { echo "no effort file: $EFFORT"; exit 1; }
MAX=${MAX_CYCLES:-200}
LOG="/tmp/perf_effort_$(basename "$EFFORT" .md).log"

PROMPT="You are autonomously advancing the CUDA KERNEL-EFFICIENCY effort (close the llama.cpp gap on the RTX 5090) on the ZINC inference engine (Zig), in ${ROOT} (a DEDICATED git worktree on the PERSISTENT branch perf/e29-cuda-kernels — work HERE, never in /Users/stepan/Workspace/zinc [main checkout] or other worktrees), then STOPPING after ONE validated increment.

STEP 0 — READ: your memory (MEMORY.md + project_effort29_cuda_kernels.md if present + project_effort26_beat_llama.md + project_effort28_batching.md + project_cuda_perf_blog.md + project_zinc_cuda_backend.md) AND the effort file ${EFFORT} IN FULL. It is the spec: the measured gap vs llama (prefill ~5-30x, decode 30-85%, serving ~6x), the LEVERS in priority order (T1b persistent fp16 weight cache FIRST = simplest real prefill win; T3 shared-mem higher-B decode GEMM + per-step fusion; T2 MoE expert GEMM; T1a cp.async/wgmma fused-dequant GEMM = the hard one, isolated-microbench-gated, LAST), the DEAD ENDS you must NOT re-litigate (pure int8 MMQ — epilogue tax kills it; FP8; m128/normf16/micro-opts; prefill CUDA graphs; async gemma decode), the validation contract, the HARD RULES, and the '## CURRENT STATE' pointer (done-so-far + exact next step). Honor all of it. This is an ACCUMULATING BUILD — reuse the existing cuBLAS GEMM / btok / gemmDispatch machinery; do NOT rewrite from scratch.

HARD CONSTRAINTS (override the generic playbook): persistent branch perf/e29-cuda-kernels (accumulate; push EVERY cycle; NEVER main). Pin the RTX 5090: export CUDA_VISIBLE_DEVICES=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350 and run validate_catalog + measurements with ZINC_GPU=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350. The 4090 (GPU-e59a6fce-…) is Effort-27's. Use the isolated box dir ~/zinc-e29 (rsync source there; never ~/workspace/zinc). Isolated-cache builds (ZIG_LOCAL_CACHE_DIR+ZIG_GLOBAL_CACHE_DIR; verify the binary md5 actually changed or you are measuring stale code). THE 5090 IS SHARED WITH EFFORT-27 WHICH LEAKS FOREIGN COMPUTE ONTO IT — util-gate EVERY A/B round (nvidia-smi --query-gpu=utilization.gpu on idx0; a round with foreign compute is GARBAGE, skip it); if contention blocks progress, log it and stop the cycle. Box gotchas: PREFILL/DECODE tok/s print to STDERR (2>&1); always 'nohup CMD >FILE 2>&1 &' on the box and poll FILE; gemma reloads ~17GB/call (~45s); models at ~/workspace/models/.

THIS CYCLE: from '## CURRENT STATE' pick the exact next lever/step (or continue an in-progress one on perf/e29-cuda-kernels), implement ONE focused kernel change, build with isolated caches (md5 changed?). VALIDATE: (1) scripts/validate_catalog.sh stays 5/5 token-correct (ZINC_GPU=5090 UUID; bit-identical for T1b, token-correctness-tolerance for reduction-order changes); (2) interleaved util-gated A/B of the new kernel vs the prior ZINC base (tok/s, medians, clear the ±10% boost floor); (3) on a real win, also A/B vs ~/workspace/llama.cpp/build/bin/llama-bench (pp256/tg128, same 5090+gguf) to track gap-closure. If correctness breaks, FIX or REVERT. If it is a VALIDATED WIN, commit ONLY this change to perf/e29-cuda-kernels and push it; then UPDATE the effort file's '## CURRENT STATE' (done + exact next step + risks) AND append a dated cycle-log entry AND your memory project_effort29_cuda_kernels.md. If NEGATIVE/contended, revert the code and log the finding. Clean up box scratch. STOP — do not loop yourself.

NEVER: break catalog correctness, commit unvalidated/swept code (commit ONLY your scoped change), push to main, async gemma decode, re-litigate the DEAD ENDS, disturb /Users/stepan/Workspace/zinc or other worktrees, or trust a single boost-noisy measurement."

echo "=== e29 cuda-kernels driver: $EFFORT  (5090, root=$ROOT, branch perf/e29-cuda-kernels, max $MAX cycles)  $(date) ===" | tee -a "$LOG"
i=0
while [ "$i" -lt "$MAX" ]; do
  i=$((i + 1))
  echo "===== e29 cycle $i / $MAX  —  $(date) =====" | tee -a "$LOG"
  claude -p --permission-mode bypassPermissions --effort high "$PROMPT" 2>&1 | tee -a "$LOG"
  echo "===== e29 cycle $i done — $(date); sleeping 60s =====" | tee -a "$LOG"
  sleep 60
done
echo "=== e29 driver finished after $i cycles $(date) ===" | tee -a "$LOG"
