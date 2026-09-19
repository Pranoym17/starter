
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
