# Effort 29 — CUDA kernel efficiency: close the llama.cpp gap (prefill + decode + serving)

> **Status:** 🔬 OPEN (spawned 2026-06-15, week-long unattended). Goal: make ZINC's CUDA kernels competitive with llama.cpp on the **5090**. This is a MULTI-CYCLE BUILD on the persistent branch **`perf/e29-cuda-kernels`** (NOT main). Each cycle: ONE validated kernel increment, A/B vs the prior ZINC base + periodically vs `llama-bench`, commit to the branch, push, update `## CURRENT STATE` + the cycle log + memory. **Watchdog-restart-safe:** all state lives in (a) this branch (committed+pushed every cycle) and (b) memory `project_effort29_cuda_kernels.md` — the worktree is reset to `origin/perf/e29-cuda-kernels` on every restart, so UNCOMMITTED work is lost; commit+push or it didn't happen.

Forward paths: `src/compute/forward_cuda_gemma.zig` (gemma4 dense+MoE), `src/compute/forward_cuda.zig` (qwen35/36 hybrid-SSM), kernels in `src/shaders/cuda/kernels.cu`, GEMM dispatch + cuBLAS shims (`cuda_cublas_hgemm`, `dequant_q4k_to_f16`, `gemmDispatch`/`gemmDispatchPrefill`).

## The measured gap (2026-06-15 clean 5090 head-to-head — the TARGETS to close)

| dimension | ZINC | llama.cpp | gap |
| --- | ---: | ---: | --- |
| **Prefill** (pp256) | ~200–500 t/s | 2.2k–6.0k | **~5–30× behind** (the biggest) |
| **Decode** (tg128, single-stream) | 43–107 (30–85% of llama) | 55–160 | behind on all 5 on the 5090 |
| **Serving** (multi-tenant, B=8) | ~26.5 (gemma-31b, btok) | ~155 | **~6× behind** |

The gap is **kernel efficiency**, not architecture: llama runs MMQ-class fused-dequant tensor-core GEMM + tuned matvec; ZINC runs cuBLAS-with-a-dequant-round-trip (prefill) + btok matvec (decode). On the 4090 ZINC is closer (wins small-model decode) — the 5090's higher bandwidth favors llama's kernels more, so 5090 is the hardest bar.

## Levers (priority order — TRACTABLE first so the week banks real wins; the hard one last)

**T1b — persistent fp16 weight cache (FIRST: simplest clearly-real prefill win).** Today the cuBLAS prefill path (`dequant_q4k_to_f16`/`dequant_q6k_to_f16` → `cuda_cublas_hgemm`) re-dequants the FULL weight to an fp16 scratch on EVERY GEMM (a fixed per-GEMM round-trip llama avoids). **Dequant each dense weight to fp16 ONCE per model load, cache it, and have prefill GEMMs read the cached fp16 directly** — trades ~2× VRAM on the dense weights for removing the per-GEMM dequant. Verify the win is REAL-USAGE (not warmup-only): measure pp256/512 across repeated prefills. Token-correct gate (fp16 cache is bit-identical to the current per-GEMM dequant). A/B vs the current cuBLAS path. Gate VRAM: if a model OOMs, cache only the largest GEMMs.

**T3 — decode/serving: shared-mem higher-B GEMM for B>27 + per-step fusion (Effort-28's unfinished lever).** `btok` token-batch matvecs win ~2–3× up to B≤27 then hit `float sum[B]` register pressure. For the serving regime (B>27) build a **shared-memory-tiled batched GEMM** — hold the B activation columns in smem (not per-thread regs), many M-blocks for occupancy = btok's bandwidth-bound design decoupled from register count. Measure vs btok27 AND the 64-pad tile GEMM. Also: per-step fusion on the batched decode step (gemma FFN/norm adjacencies, FFN gate+up dual-btok at low B where 2·B accumulators fit — mind Effort-27's block-count-preservation lesson). Token-identical gate (`dbg_cuda batch`/`sched`). This directly moves the **serving** number (~6× → less).

**T2 — MoE expert GEMM (gemma-26b prefill ~26× behind — its FLOPs are in the experts).** The routed-expert matvecs run hand-written Q4_K-dequant matvecs vs llama's MMQ expert GEMM. Route the batched routed-expert matvecs through the **cuBLAS / fp16-TC path** (per-expert or grouped GEMM) like the dense path; for DECODE drop the router host round-trip (GPU-side gather, already partly done in Effort-27) + wider expert kernels. Token-correct gate.

**T1a — cp.async/wgmma FUSED-DEQUANT fp16 GEMM (THE hard one, the real llama-MMQ approach — do LAST, isolated-microbench-gated).** Kill the dequant round-trip entirely: dequant Q4_K→fp16 INSIDE the GEMM K-loop (no scratch), pipelined with `cp.async` double-buffering + `wgmma`/`wmma` tensor-core math (keeps the per-(row,subblock) `d·sc`/`dmin·mn` scaling baked into the fp16 operand FOR FREE — the thing int8 MMQ couldn't do). **Mandatory: isolated `dbg_cuda gemm` microbench FIRST (reuse Effort-28's harness pattern), must beat the current `cuda_cublas_hgemm`/`gemm_q4k_tc` at the gemma prefill shape (M≈K≈3584–4608, T=512) before wiring into the prefill chain.** Multi-cycle. If it can't clear the bar after a fair attempt, log the negative and fall back to T1b/T3.

## DEAD ENDS — do NOT re-litigate (Effort 26/28 already killed these)
- **Pure int8 MMQ** (`s8.s8.s32` mma): the Q4_K-asymmetric epilogue forces a per-32-subblock store-s32-to-shared + fp32-rescale that the fp16 path avoids (fp16 bakes the scaling into the operand free) → microbench was 0.99–1.02× at T=512, **<1.3× kill-criterion** (Effort-26 c8). int8's 2× TC rate + half operand bits do NOT survive the epilogue tax. T1a (cp.async/wgmma fp16) is the correct path BECAUSE it keeps the free scaling.
- **FP8 (e4m3) cuBLAS** (Effort-26 c11): token-correct but in-noise/slower — dense GEMM is weight-traffic-bound, 2× TC rate buys nothing.
- **m128 A-traffic knob, normf16 recast-elision, grouped-experts reorder** (Effort-26 c2/c7): all in-noise (±1%).
- **CUDA graphs over PREFILL** (Effort-26 c1): prefill is COMPUTE-bound (~100% util) on the 5090 → no launch bubbles. (Decode/MoE graphs are already default-on where they help — Effort-25/27/28.)
- **async gemma decode** (Effort-23/25): boost-saturated, proven regression.

## Validation contract (every cycle)
- Build with ISOLATED caches (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`); verify the binary md5 actually changed or you measured stale code.
- **Token-correct — FAST per-cycle gate (do NOT run the full catalog every cycle: `validate_catalog.sh` REBUILDS cuda-dbg + loads all 5 ~17 GB models = ~68 min/cycle, far too slow).** Per cycle: build cuda-dbg ONCE, then a 1-MODEL spot-check (qwen35-9b via `dbg_cuda gen`/`batch`, ~3 min) to confirm the kernel didn't break correctness + the A/B. Run the FULL `scripts/validate_catalog.sh` (5/5, `ZINC_GPU`=5090 UUID; bit-identical for T1b, token-correctness-tolerance for T1a/T2/T3) ONLY in the cycle where you COMMIT a win — the final pre-commit gate. A divergence = bug → fix or revert. This keeps cycles ~15 min, not ~68.
- **Perf A/B:** interleaved (ABBA) new-kernel vs prior-base ZINC, util-gated (skip CONTENDED rounds — the 5090 is shared with Effort-27 which leaks; `--query-gpu=utilization.gpu`, a foreign-compute round is GARBAGE). Take medians; a "win" must robustly clear the ±~10% boost floor.
- **The BAR is llama:** on a real win, also A/B vs `~/workspace/llama.cpp/build/bin/llama-bench` on the SAME 5090 + same gguf (pp256/tg128) to track gap-closure. Don't trust a single boost-noisy number.

## HARD RULES
- **5090-pinned** (`GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350`): `CUDA_VISIBLE_DEVICES`+`ZINC_GPU`. The 4090 is Effort-27's. Branch **`perf/e29-cuda-kernels`**, NEVER main. Isolated box dir `~/zinc-e29` (rsync source; never `~/workspace/zinc`).
- **Commit ONLY the validated scoped change** + push EVERY cycle (watchdog resets uncommitted work). Update `## CURRENT STATE` (done/next/risks) + a cycle-log entry + memory `project_effort29_cuda_kernels.md` every cycle. A negative is a logged finding (valuable) → revert the code, keep the note.
- Box gotchas: `PREFILL/DECODE/GEN_IDS` print on STDERR (`2>&1`); `nohup … >FILE 2>&1 &` + poll FILE; gemma reloads ~17 GB/call (~45 s); models at `~/workspace/models/` (qwen35-9b also `~/workspace/`); `dbg_cuda gen` prompt cap [256] (bump+revert to measure pp512); `zig build cuda-dbg` ALSO RUNS the binary.
- Never break catalog correctness, push to main, async gemma decode, or re-litigate the DEAD ENDS.

## CURRENT STATE (read FIRST; update LAST every cycle)
**In progress:** nothing yet — week just spawned (branch = origin/main `b6494ebf` + this effort file + driver + watchdog).
**Exact next step:** START **T1b (persistent fp16 weight cache)** — the simplest clearly-real prefill win. Locate the per-GEMM `dequant_q4k_to_f16`→scratch→`cuda_cublas_hgemm` path in `forward_cuda_gemma.zig` (`gemmDispatch`/`gemmDispatchPrefill`), add a one-time-per-model fp16 dequant cache for the dense weights, point prefill GEMMs at it, A/B pp256/512 vs the per-GEMM path, catalog 5/5.
**Open risks/notes:** (1) 5090 shared with Effort-27 (4090) which leaks foreign compute onto idx0 → util-gate every A/B (skip CONTENDED); if it blocks progress for many cycles, log it. (2) **USE THE FAST PER-CYCLE GATE** (1-model spot-check; full `validate_catalog.sh` only in the commit cycle) — full catalog is ~68 min/cycle (rebuild + 5 model loads), far too slow to run every cycle. (3) wrapped in `loop_watchdog_e29.sh` (self-recovery, 90-min stale threshold, auto-restart on exit/hang) — **commit+push your win PROMPTLY** so a restart can't discard it (uncommitted work is reset to origin/perf/e29-cuda-kernels on restart). (4) 2026-06-15: cycle-1 began T1b (fp16 weight cache) and it passed catalog 5/5 once but wasn't committed before a watchdog/setup reset — re-do T1b (it's correctness-confirmed), commit it fast.

## Cycle log
(append per cycle: cycle | lever | change | built+md5? | catalog 5/5? | A/B (ZINC new vs base, util-gated) [+ vs llama] | branch/sha or revert+why | next)
