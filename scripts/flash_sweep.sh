#!/usr/bin/env bash
# Cycle 22 diagnostic: query-tiled flash attention (ZINC_BATCHED_FLASH) vs the
# default 3-pass gemma_attention_batched, swept over prompt length T. The grouped
# + TC sweeps (cycles 18/20) both showed dense/MoE prefill throughput PEAKING at
# T~=750 and DROPPING by T=1500 — the signature of O(T^2) attention overtaking the
# O(T) GEMMs. gemma_attention_batched reads K[0..t]+V[0..t] from GLOBAL once per
# (head,t) block → summed over t this is O(T^2) K/V traffic per head. The flash
# kernel processes BQ=8 queries per block (one warp each) that REUSE shared-memory
# K/V tiles → each K/V position is read from global once per 8 queries instead of
# once per query (~8x less K/V traffic). It uses online (flash) softmax, so it is
# token-correct but NOT bit-identical to the 3-pass kernel (validate_catalog with
# ZINC_BATCHED_FLASH=1 is its correctness gate); this script REPORTS whether the
# flash GEN matched the default's (informational) rather than asserting identity.
# ABBA-counterbalanced so boost drift doesn't masquerade as a delta. The win (if
# any) is expected to GROW with T (attention's O(T^2) share rises). Both arms run
# ZINC_BATCHED_PREFILL=1; only ZINC_BATCHED_FLASH differs.
#
# Env: ZINC_TS (space list of prompt lengths, default "250 750 1500"),
#      ZINC_ROUNDS (ABBA pairs, default 2), ZINC_GPU (default 4090),
#      ZINC_MODEL (gguf, default gemma-4-31B dense — the clean attention signal),
#      ZINC_NGEN (default 8).
set -u
declare -A GPU_UUID=(
  [5090]=GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350
  [4090]=GPU-e59a6fce-1961-bafe-927c-06c0149f2370
)
GPU=${ZINC_GPU:-4090}
export CUDA_VISIBLE_DEVICES="${GPU_UUID[$GPU]}"
MD=${ZINC_MODELS:-$HOME/workspace/models}
MODEL=${ZINC_MODEL:-$MD/gemma-4-31B-it-Q4_K_M.gguf}
TS=${ZINC_TS:-"250 750 1500"}
ROUNDS=${ZINC_ROUNDS:-2}
NGEN=${ZINC_NGEN:-8}
DIR=$(cd "$(dirname "$0")/.." && pwd); cd "$DIR"
ZBIN=$(ls -t .zig-cache/o/*/cuda-dbg 2>/dev/null | head -1)
[ -x "$ZBIN" ] || { echo "no cuda-dbg binary (build first)"; exit 1; }
echo "binary: $ZBIN   model: $(basename "$MODEL")   GPU: RTX $GPU   ABBA x$ROUNDS"

pf_of() { sed -E 's/.* = ([0-9.]+) tok.*/\1/' <<<"$1"; }
# A = default 3-pass attention (byte-identical merge path), B = flash attention
run_one() { # $1=A|B  $2=prompt -> "<tok/s>|<GEN_IDS>"
  local env_extra="ZINC_BATCHED_PREFILL=1"
  [ "$1" = "B" ] && env_extra="$env_extra ZINC_BATCHED_FLASH=1"
  local o; o=$(env $env_extra timeout 900 "$ZBIN" gen "$2" "$NGEN" "$MODEL" 2>&1)
  printf '%s|%s' "$(pf_of "$(grep -E 'PREFILL' <<<"$o" | tail -1)")" "$(grep -E 'GEN_IDS' <<<"$o" | tail -1)"
}

printf '\n  %-6s %12s %12s %8s   %s\n' "T" "3pass" "flash" "gain" "tok-match"
for T in $TS; do
  # varied non-collapsing prompt (stresses the attention math across positions)
  PROMPT=$(awk -v n="$T" 'BEGIN{for(i=0;i<n;i++){printf "%s%d",(i?",":""),((i*73+11)%251)+5}}')
  declare -a AV=() BV=(); ag=""; bg=""
  for ((r=0;r<ROUNDS;r++)); do
    for arm in A B B A; do
      res=$(run_one "$arm" "$PROMPT"); v=${res%%|*}; g=${res#*|}
      if [ "$arm" = A ]; then AV+=("$v"); [ -z "$ag" ] && ag="$g"
      else BV+=("$v"); [ -z "$bg" ] && bg="$g"; fi
    done
  done
  mean() { awk 'BEGIN{s=0;n=0} {for(i=1;i<=NF;i++){s+=$i;n++}} END{if(n>0)printf "%.2f",s/n}' <<<"$*"; }
  am=$(mean "${AV[@]}"); bm=$(mean "${BV[@]}")
  gain=$(awk -v a="${am:-0}" -v b="${bm:-0}" 'BEGIN{if(a>0)printf "%+.1f%%",(b/a-1)*100; else print "-"}')
  if [ -n "$ag" ] && [ "$ag" = "$bg" ]; then tm="yes (no divergence)"; else tm="DIVERGED (online-softmax tol)"; fi
  printf '  %-6s %12s %12s %8s   %s\n' "$T" "${am:--}" "${bm:--}" "$gain" "$tm"
done
echo ""
echo "note: flash attention uses online softmax (token-correct within tolerance,"
echo "      gated by validate_catalog with ZINC_BATCHED_FLASH=1), NOT byte-identical"
echo "      to the 3-pass kernel — kept opt-in behind ZINC_BATCHED_FLASH. The 3-pass"
echo "      arm is the strict byte-identity merge path."
