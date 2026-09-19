# NOTES — Qwen3-4B decode engine

## Platform baseline (unchanged starter)
Run `ad765af3-729d-411f-bfcf-209c96dd2af7` (official, commit 3e18b72), succeeded, **score 127.29 tok/s** (geomean over 6 hidden workloads; every run is official).
Container: NVIDIA H100 80GB HBM3, driver 580.95.05, Python 3.11.5, gVisor kernel.

| shape | tok/s | total ms (p50) | TTFT ms | TPOT ms | native TTFT | native TPOT | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| public-0 b1×512→32 | 27.12 | 1180.0 | 43.47 | 36.64 | 42.00 | 34.88 | 10.28 GB |
| public-1 b4×2048→32 | 92.38 | 1385.6 | 200.73 | 38.24 | 200.37 | 37.77 | 12.28 GB |
| public-2 b16×512→128 | 410.26 | 4991.9 | 190.22 | 37.81 | 189.90 | 37.98 | 12.28 GB |

Observations: TPOT ≈ 37 ms regardless of batch → decode is purely overhead bound on the platform
(bandwidth floor is ~2.5 ms). Gates: our TTFT ≤ 1.10× native TTFT (~46 ms at b1×512, ~220 ms at b4×2048).

Note: the dryft CLI needs `DRYFT_API=https://htn.dryft.ai` (default endpoint returns HTTP 403).

## Setup status
- No GPU: the Baseten H100 workstation (job wlvve1q) never left PENDING and was stopped; no jobs active.
- `scripts/remote_setup.sh`, `localjudge/` (judge.py, bench.py, profile_step.py) and `scripts/hotpath_dryft_qwen.yaml`:
  **optional, not used (no GPU)**. Kept for when a GPU is available.
- Every Dryft run is the test: the engine self-checks during the untimed warmup and logs `[engine]` lines.
- Pre-push: `scripts/cpu_sanity.py` (CPU venv with torch 2.5.1+cpu, transformers 4.51.3; tiny random Qwen3,
  exercises T1/T0, self-check bookkeeping, yield contract) + py_compile + `dryft validate engine`.

## Engine tiers (self-guard, chosen per workload during warmup)
T3 graphs + fused Triton (qk-norm+RoPE+cache write, SiLU×up) · T2 graphs + Triton norm/decode-attn ·
T1 eager torch-only (reference SDPA on cache) · T0 native starter loop.
Each tier is validated on the warmup prompt (first 32 steps, teacher-forced through native HF);
margin > 1.0, exception or non-finite → `FALLBACK T<n>→T<n-1>`.

## Runs
| # | commit | change | score | b1×512→32 | b4×2048→32 | b16×512→128 | TTFT/TPOT ratio (worst) | tier | margin | verdict |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| 0 | 3e18b72 | unchanged starter | 127.29 | 27.1 | 92.4 | 410.3 | 1.03/1.05 | native | – | baseline |

Current best: 127.29 · target 1200 · gap 9.4×

## Phase 4 design — exact prompt-lookup speculative decoding (T4 + spec)
**Draft (host, per sequence).** Prompt-lookup n-grams over prompt + generated tokens. Two dicts per
sequence (n=3, n=2) map an n-gram → index right after its most recent *earlier* occurrence (an n-gram
is registered only once its continuation token exists, so the current suffix never matches itself).
Draft = the k tokens following the match (n=3 first, then n=2); short/no match → pad with the last
token (padding is simply rejected; the graph shape is fixed). Index build is O(S) dict inserts per
sequence (~1 ms at S=2048), done after the first token is yielded, so TTFT is untouched.

**Verify (device, one CUDA graph, fixed k).** Input per sequence: T = k+1 tokens
`[last accepted token, d1..dk]` at absolute positions `pos_b .. pos_b+k` (per-sequence `pos` vector).
Forward = the T4 decode path generalised to M = B·T rows: fused q/k-norm + RoPE + KV write at
`pos_b + t`, a causal multi-query split-K GQA attention (row (t, head) sees keys `<= pos_b + t`;
fully-masked splits are NaN-safe), skinny GEMMs tuned for M = B·T, LM head + argmax for all B·T rows.
Output g[b, 0..k] = our greedy token after each prefix.

**Accept (host).** a_b = longest prefix with d_i == g[b, i-1]; emit g[b, 0..a_b] (a_b+1 tokens).
Every emitted token is the model's argmax on our own prefix → exact by construction (only reduction
order differs from T=1 decode). KV at `pos_b .. pos_b+a_b` came from accepted inputs; rejected slots
are overwritten by the next verify, and attention never reads past `pos_b + t`.

**Batch.** Per-sequence acceptance; yield step t once *every* sequence has token t. Sequences that
reach N tokens are frozen (same input, same pos) so capacity S+N+T is never exceeded.
Throughput per verify ≈ min_b(a_b)+1 → useful mainly for small B; M = B·T must be ≤ 64.
k = 6 for B = 1, 4 for B ≤ 12 (B·T ≤ 64), off otherwise.

**Safety / auto-disable (warmup only).** After T4 is chosen: capture the verify graph, run the spec
stream on the warmup prompt for min(N, 128) tokens, teacher-force them through native (margin ≤ 1.0),
and time spec vs plain T4 on the same prompt. Keep spec only if it is exact and ≥ 8% faster.
**Risks:** (1) the 25% spread gate — acceptance varies by prompt; total time includes prefill, which
dilutes it; (2) one host sync per verify (no one-step-ahead pipelining) ~50–100 µs; (3) extra warmup
compile for the M = B·T GEMM plans (budget 300 s).
