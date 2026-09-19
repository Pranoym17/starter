
## Offline kernel lab (built after run 3) - `lab/`
- WSL Ubuntu, no GPU. `lab/compile_check.py` (venv ~/lab: the platform's torch 2.5.1 + **Triton 3.1.0**):
  a fake driver reports sm_90, so the engine's own wrappers trigger exactly the H100 compilations;
  ptxas -v reports registers/spills. Current: **1246 specialisations, 0 failures**, spills <= 12 B.
- `lab/interp_test.py` (venv ~/lab2: Triton 3.4 interpreter + `lab/interp_shims.py`, which fixes the
  interpreter's BF16 handling to GPU semantics: exact fp32 dot/products, RNE casts) - every kernel vs a
  torch reference incl. split-K tickets, cache writes only at `pos`, lowest-index ties; plus end-to-end
  tiny Qwen3 (real head geometry) through T5/T4/T3/T2 vs native HF greedy. **All pass.**
- Run-3 root cause: kernels were correct (lab-verified) -> T4's fused-norm prologue (every GEMV block
  re-reads the whole x row before streaming weights) was a latency *loss*. Removed; T4 = run 2's form.
- Bug caught by the lab before shipping: `_State.capacity` was never set -> T5 would have crashed.

## Run 4 (fadc392): T5
T5 = 1 + 5x36 + 1 = 182 launches/step (T4 ~ 330): RMSNorm statistics accumulated in the producing
epilogue (fp32 atomics), split-K reduced by the last CTA (ticket), attention with fused q/k-norm +
RoPE + KV write + merge, LM head + argmax + pos++ in one launch, `evict_first` weight streams, per-B
autotune of (tile, warps, stages, split-K). Warmup times T5 vs T4 and keeps the faster. Prefill:
GQA-native flash (`enable_gqa`, probed for bitwise equality with repeat_kv at warmup). Spec: off.

## Run 4 result (fadc392, T5 + GQA-native flash prefill)
Score **901.97** (new best; attempt 1 was a platform harness_error, attempt 2 ranked).
public-0 229.8 tok/s (TTFT 11.2/21.9, TPOT 4.13/17.54) · public-1 477.8 (TTFT 125.6/202.1, TPOT 4.59/20.80)
· public-2 2895.5 (TTFT 114.9/191.7, TPOT 4.66/22.47); spreads < 0.5%; peak 13-17 GB.
Reading: TTFT -5% on long prompts (GQA flash). TPOT unchanged vs run 2 -> T5 was at best equal to T4:
cutting launches 330 -> 182 did not move decode, so the remaining gap is GEMV bandwidth efficiency, not
launch overhead. Next: forced-tier diagnostic run(s) to measure T5/T6 in isolation.

## Run 5 (pushed): T6 megakernel + Phase 4 v2 tree speculation
- T6: one persistent kernel per decode step (lab: compiles for sm_90, 255 regs / 0 spills / 74 KB smem,
  end-to-end tokens == native with P=1). Warmup compares up to 3 passing tiers (T6/T5/T4), keeps fastest.
- Tree speculation (kernels/tree.py + engine/spec.py): token tree of 16 (B=1) / 8 (B<=8) nodes verified
  in one forward; candidates = Jacobi lookahead window + multi-occurrence n-grams + model-prediction pool;
  ancestor-mask attention, per-node RoPE, KV compaction of the accepted path. Lab: kernels pass vs brute
  force; host logic exact on 6 mock cases (frozen sequences, N=1/2, B=8). NOT yet run end-to-end with real
  kernels in the interpreter. Guarded: setup/warmup/capture failure disables only speculation; enabled
  only if warmup teacher-forcing is exact and it is >= 8% faster. Risk to watch: 25% spread gate.

Current best: 901.97 · target 1200 · gap 1.33x
