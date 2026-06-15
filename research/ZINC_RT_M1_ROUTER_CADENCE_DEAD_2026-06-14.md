# ZINC_RT M1 Router/MoE Cadence Dead End - 2026-06-14

## Context

The M1 default path already consumes a real AMDGPU CS DMMV model slice:

- first decode step: `lm_head_q4_0_argmax_prefix_resident`
- periodic router/MoE proof: `router_q8_0_row_range_parallel64_primary` plus Q4_0 MoE gate/up and down prefixes

The 2026-06-14 benchmark sample before this check was around 35.2 tok/s with
`direct_compute_ops=31`, `direct_decode_model_slices=31`, coherent output, and
`benchmark_shortcuts=none`.

## Remote Evidence

All runs used the R9700 ZINC_RT RDNA node, the Qwen3.6 35B-A3B Q4_K_XL model,
prompt `The capital of France is`, `--max-tokens 96`, and
`RADV_PERFTEST=coop_matrix`.

| Variant | Decode tok/s | Direct ops | Direct slices | Evidence | Result |
|---|---:|---:|---:|---|---|
| Skip router/MoE after first LM-head DMMV proof | 34.04, 34.04 | 1 | 1 | `lm_head_q4_0_argmax_prefix_resident`, coherent output | Regressed |
| Router/MoE cadence 16 | 34.93 | 151 | 151 | repeated router/MoE DMMV slices, coherent output | Regressed |
| Router/MoE cadence 32 | 35.09 | 61 | 61 | repeated router/MoE DMMV slices, coherent output | Flat/slightly down |
| Router/MoE cadence 48 | 35.11 | 31 | 31 | one router/MoE DMMV slice, coherent output | Flat |
| Router/MoE cadence 64 | 35.11 | 31 | 31 | one router/MoE DMMV slice, coherent output | Current default behavior |

The router/MoE slice is not just validation overhead: removing it slowed the
96-token decode despite preserving a consumed LM-head DMMV proof. However,
running that slice more often did not improve throughput. The extra CS work and
validation traffic cancel the CPU rows saved at this granularity.

## Verdict

Default router/MoE cadence tuning is measured-dead for M1 throughput:

- do not suppress the cadence-64 router/MoE proof while the path is still
  host-assisted
- do not lower the default cadence to 48, 32, 16, or 0 without a same-cycle
  material tok/s gain
- more consumed slices alone are not progress after the current coverage point

The next plausible direct-execution work should change the cost model, not the
cadence:

- make recurring Q4_0 row ranges use resident/device-local weights
- batch more same-input ranges into fewer CS fences only with measured tok/s gain
- replace a larger on-path CPU matvec section for every token, not just add more
  proof slices
