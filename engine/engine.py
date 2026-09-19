"""Qwen3 4B greedy decode engine.

Loads the checkpoint with Transformers (so tied weights and dtypes are exactly the
reference's) and runs its own forward: packed QKV and gate/up weights, a static per-layer KV
cache, SDPA flash prefill, a Triton split-K GQA decode attention that reads a device-side
position, and CUDA graphs over both prefill and the decode step. Arithmetic follows
transformers 4.51.3 modeling_qwen3 operation by operation; only reduction order differs.

Tiers, fastest first. On the first call of a shape (the platform's untimed warmup) the engine
validates tiers against the loaded native model by teacher-forcing their own tokens, falling
back a tier on any exception, non-finite logit, or margin above MARGIN_LIMIT, and keeps the
faster of the two best passing tiers:
  T4  T3 + Triton skinny-GEMM decode (fused residual / SwiGLU epilogues, LM head + argmax)
  T3  CUDA graphs + fused Triton kernels
  T2  CUDA graphs + Triton norm / decode attention, torch elementwise ops
  T1  eager, torch ops only (reference-style SDPA over the cache)
  T0  the starter's native Transformers loop
Diagnostics go to stderr with an "[engine]" prefix, only during that warmup call.
"""

import gc
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kernels.decode_attn import DecodeAttention
    from kernels.rmsnorm import rms_norm_rows
    _KERNEL_ERR = None
except Exception as _exc:  # no Triton: the torch-only tiers still work
    _KERNEL_ERR = repr(_exc)[:200]
try:
    from kernels.fused import qk_norm_rope_cache, silu_mul
    _FUSED_ERR = None
except Exception as _exc:
    _FUSED_ERR = repr(_exc)[:200]
try:
    from kernels.gemv import EPI_NONE, EPI_RES, EPI_SILU, Gemv, LmHeadArgmax
    _GEMV_ERR = None
except Exception as _exc:
    _GEMV_ERR = repr(_exc)[:200]

DEVICE = os.environ.get("ENGINE_DEVICE", "cuda:0")
CUDA = DEVICE.startswith("cuda")
N_HEADS, N_KV, HEAD_DIM = 32, 8, 128
Q_SIZE, KV_SIZE = N_HEADS * HEAD_DIM, N_KV * HEAD_DIM
BF16 = torch.bfloat16

CHECK_STEPS = 32          # self-check length (tokens) on the warmup prompt
CHECK_ROWS = 4            # rows run through the native loop for the divergence report
MARGIN_LIMIT = 1.0        # native's own drift is <= 0.75; the judge's margin is 2.0
PREFILL_TOKENS = 16384    # prefill processes at most this many tokens per row-chunk
FORCE_TIER = int(os.environ.get("ENGINE_TIER", "-1"))

GEMV_MAX_B = 64          # T4's skinny-GEMM decode path covers batches up to this

#            graphs  triton  fused  gemv
TIERS = {4: (True, True, True, True), 3: (True, True, True, False),
         2: (True, True, False, False), 1: (False, False, False, False)}


def log(msg):
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _repeat_kv(x):
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, N_HEADS // N_KV, s, d).reshape(b, N_HEADS, s, d)


class _NullEvent:
    def record(self, *a):
        pass

    def synchronize(self):
        pass


def _event(timing=False):
    return torch.cuda.Event(enable_timing=timing) if CUDA else _NullEvent()


class _Layer:
    __slots__ = ("ln1", "w_qkv", "q_norm", "k_norm", "w_o", "ln2", "w_gu", "w_down")


class _State:
    """Everything shape-dependent for one (batch, prompt_len, max_new_tokens, tier)."""

    def __init__(self, eng, B, S, N, tier):
        self.B, self.S, self.N, self.tier = B, S, N, tier
        self.graphs, self.triton, self.fused, self.gemv = TIERS[tier]
        cap = S + N
        n_layers = len(eng.layers)
        self.k_cache = [torch.zeros((B, N_KV, cap, HEAD_DIM), dtype=BF16, device=DEVICE) for _ in range(n_layers)]
        self.v_cache = [torch.zeros((B, N_KV, cap, HEAD_DIM), dtype=BF16, device=DEVICE) for _ in range(n_layers)]
        with torch.inference_mode():
            dummy = torch.empty(1, dtype=BF16, device=DEVICE)
            cos, sin = eng.rotary(dummy, torch.arange(cap, device=DEVICE)[None])  # reference module
        self.cos, self.sin = cos[0].contiguous(), sin[0].contiguous()          # [cap, 128] BF16
        self.ids = torch.zeros((B, S), dtype=torch.int64, device=DEVICE)
        self.tok = torch.zeros((B,), dtype=torch.int64, device=DEVICE)
        self.pos = torch.zeros((1,), dtype=torch.int64, device=DEVICE)
        self.rows = max(1, min(B, PREFILL_TOKENS // S))
        self.attn = DecodeAttention(B, N_KV, cap, DEVICE, eng.sm_count) if self.triton else None
        self.attn_out = torch.empty((B, N_HEADS, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.q_buf = torch.empty((B, N_HEADS, 1, HEAD_DIM), dtype=BF16, device=DEVICE)
        self.host = torch.empty((max(N, 1), B), dtype=torch.int64, pin_memory=CUDA)
        self.ids_host = torch.empty((B, S), dtype=torch.int64, pin_memory=CUDA)
        self.events = [_event() for _ in range(max(N, 1))]
        self.prefill_graph = self.decode_graph = None
        self.prefill_logits = None
        self.compile_s = self.capture_s = 0.0
        if self.gemv:
            eng._gemv_plans(self)


class Engine:
    def __init__(self, model_path: str) -> None:
        t0 = time.perf_counter()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.hf = (
            AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=BF16, attn_implementation="sdpa", local_files_only=True,
            ).eval().to(DEVICE)
        )
        base = self.hf.model
        cfg = self.hf.config
        self.eps = cfg.rms_norm_eps
        self.inter = cfg.intermediate_size
        self.embed = base.embed_tokens.weight
        self.lm_head = self.hf.lm_head.weight                  # tied: same storage as embed
        self.final_norm = base.norm.weight
        self.rotary = base.rotary_emb
        self.layers = []
        with torch.no_grad():
            for layer in base.layers:
                at, mlp = layer.self_attn, layer.mlp
                L = _Layer()
                L.ln1 = layer.input_layernorm.weight
                L.ln2 = layer.post_attention_layernorm.weight
                L.q_norm, L.k_norm = at.q_norm.weight, at.k_norm.weight
                L.w_qkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
                L.w_o = at.o_proj.weight
                L.w_gu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
                L.w_down = mlp.down_proj.weight
                self.layers.append(L)
        self.sm_count = torch.cuda.get_device_properties(DEVICE).multi_processor_count if CUDA else 1
        self.state = None
        self.tier = None
        self._tuned = {}
        self.load_s = time.perf_counter() - t0
        log(f"loaded in {self.load_s:.1f}s; kernels {'ok' if _KERNEL_ERR is None else 'UNAVAILABLE ' + _KERNEL_ERR}; "
            f"fused {'ok' if _FUSED_ERR is None else 'UNAVAILABLE ' + _FUSED_ERR}")

    def _gemv_plans(self, st):
        """T4: skinny-GEMM plans for batch B, tuned once per B (compiles every config it keeps)."""
        B, H, I = st.B, self.embed.shape[1], self.inter
        L = self.layers[0]
        st.qkv_buf = torch.empty((B, L.w_qkv.shape[0]), dtype=BF16, device=DEVICE)
        st.act_buf = torch.empty((B, I), dtype=BF16, device=DEVICE)
        cached = self._tuned.get(B)
        plans = {
            "qkv": Gemv(B, L.w_qkv.shape[0], H, EPI_NONE, DEVICE, self.sm_count),
            "o": Gemv(B, H, L.w_o.shape[1], EPI_RES, DEVICE, self.sm_count),
            "gu": Gemv(B, I, H, EPI_SILU, DEVICE, self.sm_count),
            "down": Gemv(B, H, I, EPI_RES, DEVICE, self.sm_count),
        }
        lm = LmHeadArgmax(B, self.lm_head.shape[0], H, DEVICE)
        if cached is None:
            t0 = time.perf_counter()
            g = torch.Generator(device=DEVICE).manual_seed(0)
            xs = {k: torch.randn((B, n), generator=g, device=DEVICE).to(BF16) for k, n in
                  (("qkv", H), ("o", L.w_o.shape[1]), ("gu", H), ("down", I))}
            ws = {"qkv": L.w_qkv, "o": L.w_o, "gu": L.w_gu, "down": L.w_down}
            outs = {"qkv": st.qkv_buf, "o": torch.empty((B, H), dtype=BF16, device=DEVICE),
                    "gu": st.act_buf, "down": torch.empty((B, H), dtype=BF16, device=DEVICE)}
            res = torch.zeros((B, H), dtype=BF16, device=DEVICE)
            cached, desc = {}, []
            for k, plan in plans.items():
                t, cfg = plan.tune(xs[k], ws[k], outs[k], res if plan.epi == EPI_RES else None)
                cached[k] = cfg
                desc.append(f"{k}={cfg[0]}x{cfg[1]}/w{cfg[2]}s{cfg[3]}k{plan.split}:{t * 1e3:.0f}us")
            tok = torch.empty((B,), dtype=torch.int64, device=DEVICE)
            t, cfg = lm.tune(xs["qkv"], self.lm_head, tok)
            cached["lm"] = cfg
            desc.append(f"lm={cfg[0]}x{cfg[1]}:{t * 1e3:.0f}us")
            self._tuned[B] = cached
            log(f"gemv tuned B={B} in {time.perf_counter() - t0:.1f}s: {' '.join(desc)}")
        for k, plan in plans.items():
            plan.set_config(cached[k])
        lm.set_config(cached["lm"])
        st.plans, st.lm = plans, lm

    def _decode_gemv(self, st):
        """T4 decode step: Triton skinny GEMMs with fused epilogues; residual stream updated in place."""
        B = st.B
        p = st.plans
        x = F.embedding(st.tok, self.embed)
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            h = rms_norm_rows(x, L.ln1, self.eps).view(B, -1)
            qkv = p["qkv"](h, L.w_qkv, st.qkv_buf)
            qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, st.q_buf, kc, vc,
                               B, 1, self.eps, True)
            a = st.attn(st.q_buf.view(B, N_HEADS, HEAD_DIM), kc, vc, st.pos, st.attn_out)
            p["o"](a.view(B, Q_SIZE), L.w_o, x, res=x)
            h = rms_norm_rows(x, L.ln2, self.eps).view(B, -1)
            act = p["gu"](h, L.w_gu, st.act_buf)
            p["down"](act, L.w_down, x, res=x)
        h = rms_norm_rows(x, self.final_norm, self.eps).view(B, -1)
        st.lm(h, self.lm_head, st.tok)
        st.pos.add_(1)

    # ------------------------------------------------------------------ building blocks
    def _norm(self, st, x2d, w, heads=1):
        """Qwen3RMSNorm over rows of width w.numel(); x2d [M, >=heads*N] -> [M, heads, N]."""
        if st.triton:
            return rms_norm_rows(x2d, w, self.eps, heads)
        n = w.numel()
        xf = x2d[:, :heads * n].reshape(x2d.shape[0], heads, n).to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return w * xf.to(BF16)

    def _act(self, st, gu):
        if st.fused:
            return silu_mul(gu, self.inter)
        return F.silu(gu[:, :self.inter]) * gu[:, self.inter:]

    def _prefill(self, st):
        """Prompt forward in row chunks; fills KV [0, S), writes token 0 to st.tok, returns logits."""
        B, S = st.B, st.S
        cos = st.cos[:S][None, None]
        sin = st.sin[:S][None, None]
        outs = []
        for r0 in range(0, B, st.rows):
            r1 = min(B, r0 + st.rows)
            b, M = r1 - r0, (r1 - r0) * S
            x = F.embedding(st.ids[r0:r1], self.embed).view(M, -1)
            for i, L in enumerate(self.layers):
                kc, vc = st.k_cache[i][r0:r1], st.v_cache[i][r0:r1]
                h = self._norm(st, x, L.ln1).view(M, -1)
                qkv = F.linear(h, L.w_qkv)
                if st.fused:
                    q = torch.empty((b, N_HEADS, S, HEAD_DIM), dtype=BF16, device=DEVICE)
                    qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, q, kc, vc,
                                       b, S, self.eps, False)
                else:
                    q = self._norm(st, qkv, L.q_norm, N_HEADS).view(b, S, N_HEADS, HEAD_DIM).transpose(1, 2)
                    k = self._norm(st, qkv[:, Q_SIZE:], L.k_norm, N_KV).view(b, S, N_KV, HEAD_DIM).transpose(1, 2)
                    v = qkv[:, Q_SIZE + KV_SIZE:].view(b, S, N_KV, HEAD_DIM).transpose(1, 2)
                    q = (q * cos) + (_rotate_half(q) * sin)
                    k = (k * cos) + (_rotate_half(k) * sin)
                    kc[:, :, :S].copy_(k)
                    vc[:, :, :S].copy_(v)
                a = F.scaled_dot_product_attention(
                    q.contiguous(), _repeat_kv(kc[:, :, :S]), _repeat_kv(vc[:, :, :S]),
                    is_causal=True, scale=HEAD_DIM ** -0.5)
                x = x + F.linear(a.transpose(1, 2).reshape(M, Q_SIZE), L.w_o)
                h = self._norm(st, x, L.ln2).view(M, -1)
                x = x + F.linear(self._act(st, F.linear(h, L.w_gu)), L.w_down)
            last = x.view(b, S, -1)[:, -1]
            outs.append(F.linear(self._norm(st, last, self.final_norm).view(b, -1), self.lm_head))
        logits = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        return logits

    def _decode(self, st, host_pos=None):
        """One token per sequence at st.pos (device) or host_pos (eager T1); advances st.pos."""
        if st.gemv:
            return self._decode_gemv(st)
        B = st.B
        x = F.embedding(st.tok, self.embed)
        if host_pos is None:
            cos = st.cos.index_select(0, st.pos)[None]           # [1, 1, 128]
            sin = st.sin.index_select(0, st.pos)[None]
        else:
            cos = st.cos[host_pos:host_pos + 1][None]
            sin = st.sin[host_pos:host_pos + 1][None]
        for i, L in enumerate(self.layers):
            kc, vc = st.k_cache[i], st.v_cache[i]
            h = self._norm(st, x, L.ln1).view(B, -1)
            qkv = F.linear(h, L.w_qkv)
            if st.fused:
                qk_norm_rope_cache(qkv, L.q_norm, L.k_norm, st.cos, st.sin, st.pos, st.q_buf, kc, vc,
                                   B, 1, self.eps, True)
                q = st.q_buf.view(B, N_HEADS, HEAD_DIM)
            else:
                q = self._norm(st, qkv, L.q_norm, N_HEADS)
                k = self._norm(st, qkv[:, Q_SIZE:], L.k_norm, N_KV)
                v = qkv[:, Q_SIZE + KV_SIZE:].view(B, N_KV, HEAD_DIM)
                q = (q * cos) + (_rotate_half(q) * sin)
                k = (k * cos) + (_rotate_half(k) * sin)
                if host_pos is None:
                    kc.index_copy_(2, st.pos, k[:, :, None])
                    vc.index_copy_(2, st.pos, v[:, :, None])
                else:
                    kc[:, :, host_pos].copy_(k)
                    vc[:, :, host_pos].copy_(v)
            if st.triton:
                a = st.attn(q.contiguous(), kc, vc, st.pos, st.attn_out).view(B, Q_SIZE)
            else:
                n = host_pos + 1
                a = F.scaled_dot_product_attention(
                    q.view(B, N_HEADS, 1, HEAD_DIM).contiguous(), _repeat_kv(kc[:, :, :n]), _repeat_kv(vc[:, :, :n]),
                    scale=HEAD_DIM ** -0.5).transpose(1, 2).reshape(B, Q_SIZE)
            x = x + F.linear(a, L.w_o)
            h = self._norm(st, x, L.ln2).view(B, -1)
            x = x + F.linear(self._act(st, F.linear(h, L.w_gu)), L.w_down)
        logits = F.linear(self._norm(st, x, self.final_norm).view(B, -1), self.lm_head)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        st.pos.add_(1)
        return logits

    # ------------------------------------------------------------------ shapes, graphs
    def _build(self, B, S, N, tier):
        if tier == 4 and B > GEMV_MAX_B:
            tier = 3
        self.state = None
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()
        st = _State(self, B, S, N, tier)
        if st.graphs:
            with torch.inference_mode():
                t0 = time.perf_counter()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):          # JIT-compiles every Triton specialisation, warms cuBLAS
                        self._prefill(st)
                        if N > 1:
                            st.pos.fill_(S)
                            self._decode(st)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                st.compile_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                try:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        st.prefill_logits = self._prefill(st)
                    st.prefill_graph = g
                    if N > 1:
                        st.pos.fill_(S)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            self._decode(st)
                        st.decode_graph = g
                except Exception as exc:
                    log(f"graph capture failed, T{tier} runs eager: {repr(exc)[:160]}")
                    st.prefill_graph = st.decode_graph = None
                torch.cuda.synchronize()
                st.capture_s = time.perf_counter() - t0
        self.state = st
        return st

    def _stream(self, st, input_ids, N, timing=None):
        """Greedy stream of N steps on st. Keeps the GPU one step ahead of the host."""
        B, S = st.B, st.S
        if timing is not None:
            ev = [(_event(True), _event(True)) for _ in range(N)]
        with torch.inference_mode():
            st.ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
            st.ids.copy_(st.ids_host, non_blocking=True)
            st.pos.fill_(S)
            if timing is not None:
                ev[0][0].record()
            if st.prefill_graph is not None:
                st.prefill_graph.replay()
            else:
                st.prefill_logits = self._prefill(st)
            if timing is not None:
                ev[0][1].record()
            st.host[0].copy_(st.tok, non_blocking=True)
            st.events[0].record()
        for t in range(1, N):
            with torch.inference_mode():
                if timing is not None:
                    ev[t][0].record()
                if st.decode_graph is not None:
                    st.decode_graph.replay()
                elif st.graphs or st.triton:
                    self._decode(st)
                else:
                    self._decode(st, host_pos=S + t - 1)
                if timing is not None:
                    ev[t][1].record()
                st.host[t].copy_(st.tok, non_blocking=True)
                st.events[t].record()
            st.events[t - 1].synchronize()
            yield st.host[t - 1].tolist()
        st.events[N - 1].synchronize()
        if timing is not None and CUDA:
            timing.extend(a.elapsed_time(b) for a, b in ev)
        yield st.host[N - 1].tolist()

    # ------------------------------------------------------------------ native reference (T0 + self-check)
    def _native(self, input_ids, N):
        current = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
        cache = None
        with torch.inference_mode():
            for _ in range(N):
                output = self.hf(input_ids=current, past_key_values=cache, use_cache=True,
                                 logits_to_keep=1, return_dict=True)
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist(), output.logits[:, -1, :]

    def _teacher_force(self, input_ids, toks):
        """Replay prompt + toks through native Qwen in one forward (row chunks).
        Returns (max margin, #non-argmax positions, first step with a non-argmax token or -1)."""
        K = len(toks[0])
        S = len(input_ids[0])
        chunk = max(1, 8192 // (S + K))
        worst, nonarg, first = 0.0, 0, -1
        with torch.inference_mode():
            for r0 in range(0, len(input_ids), chunk):
                seqs = [p + t[:-1] for p, t in zip(input_ids[r0:r0 + chunk], toks[r0:r0 + chunk])]
                ids = torch.tensor(seqs, dtype=torch.int64, device=DEVICE)
                logits = self.hf(input_ids=ids, logits_to_keep=K, use_cache=False).logits.float()
                tgt = torch.tensor(toks[r0:r0 + chunk], dtype=torch.int64, device=DEVICE)
                margin = logits.max(-1).values - logits.gather(-1, tgt[..., None])[..., 0]
                if not torch.isfinite(margin).all():
                    return float("inf"), -1, -1
                worst = max(worst, margin.max().item())
                bad = (margin > 0).any(0).nonzero()
                nonarg += int((margin > 0).sum().item())
                if len(bad):
                    step = int(bad[0].item())
                    first = step if first < 0 else min(first, step)
        return worst, nonarg, first

    def _select_tier(self, input_ids, N):
        """Warmup-only: validate tiers best-first on this prompt; keep the faster of the first two
        that pass (a new kernel tier can never make the engine wrong, nor slower than the next one)."""
        t_start = time.perf_counter()
        B, S = len(input_ids), len(input_ids[0])
        K = min(N, CHECK_STEPS)
        rows = input_ids[:CHECK_ROWS]
        native_toks, native_logits0 = [], None
        for step, logits in self._native(rows, K):
            native_toks.append(step)
            if native_logits0 is None:
                native_logits0 = logits.float()
        native_toks = [list(r) for r in zip(*native_toks)]
        if FORCE_TIER >= 0:
            order = [FORCE_TIER] if FORCE_TIER > 0 else []
        elif _KERNEL_ERR is None and CUDA:
            order = ([4] if _GEMV_ERR is None and _FUSED_ERR is None and B <= GEMV_MAX_B else [])                 + ([3] if _FUSED_ERR is None else []) + [2, 1]
        else:
            order = [1]
        passing = []                       # (wall ms for K tokens, tier, per-step GPU timings)
        for n, tier in enumerate(order):
            try:
                st = self._build(B, S, N, tier)
                steps = list(self._stream(st, input_ids, K))
                ours = [list(r) for r in zip(*steps)]
                pl = st.prefill_logits.float()
                finite = bool(torch.isfinite(pl).all().item())
                dlog = (pl[:len(rows)] - native_logits0).abs().max().item()
                margin, nonarg, first_nonarg = self._teacher_force(input_ids, ours)
                div = next((t for t in range(K) if any(ours[r][t] != native_toks[r][t] for r in range(len(rows)))), -1)
                log(f"check T{tier}: margin={margin:.3f} non_argmax={nonarg}/{B * K} first_non_argmax={first_nonarg} "
                    f"first_diverge_vs_native={div} prefill_logit_maxdiff={dlog:.3f} finite={finite} "
                    f"compile={st.compile_s:.1f}s capture={st.capture_s:.1f}s")
                if finite and margin <= MARGIN_LIMIT:
                    wall, timing = self._time_stream(st, input_ids, K)
                    passing.append((wall, tier, timing))
                    if len(passing) == 2 or K == 1 or tier == 1:
                        break
                    continue
                reason = "non-finite logits" if not finite else f"margin {margin:.3f} > {MARGIN_LIMIT}"
            except Exception as exc:
                reason = f"exception {repr(exc)[:200]}"
            nxt = order[n + 1] if n + 1 < len(order) else 0
            log(f"FALLBACK T{tier}->T{nxt}: {reason}")
            self.state = None
        check_s = time.perf_counter() - t_start

        if passing:
            wall, chosen, timing = min(passing)
            if len(passing) > 1:
                log("speed: " + " ".join(f"T{t}={w / K:.3f}ms/tok" for w, t, _ in passing) + f" -> T{chosen}")
            if self.state is None or self.state.tier != chosen:
                self._build(B, S, N, chosen)
            # free the native model's unpacked copies; our tiers never touch them again
            self.hf = None
            gc.collect()
            if CUDA:
                torch.cuda.empty_cache()
            if timing:
                dec = sorted(timing[1:]) or [0.0]
                host = (wall - sum(timing)) / max(1, K)
                log(f"timing T{chosen} {B}x{S}: prefill={timing[0]:.2f}ms step mean={sum(dec) / len(dec):.3f} "
                    f"p50={dec[len(dec) // 2]:.3f} max={dec[-1]:.3f}ms host_overhead/step={host:.3f}ms "
                    f"stream_wall={wall:.1f}ms for {K} tok")
        else:
            chosen = 0
        self.tier = chosen
        peak = torch.cuda.max_memory_reserved() / 1e9 if CUDA else 0.0
        log(f"tier=T{chosen} shape={B}x{S}x{N} check={check_s:.1f}s load={self.load_s:.1f}s peak={peak:.1f}GB")

    def _time_stream(self, st, input_ids, K):
        """Best-of-two wall time (ms) of a K-token stream, with per-step GPU timings."""
        best = None
        for _ in range(2):
            timing = []
            t0 = time.perf_counter()
            for _ in self._stream(st, input_ids, K, timing):
                pass
            wall = (time.perf_counter() - t0) * 1e3
            if best is None or wall < best[0]:
                best = (wall, timing)
        return best

    # ------------------------------------------------------------------ API
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        N = int(max_new_tokens)
        if N <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        gc_was = gc.isenabled()
        gc.disable()
        try:
            st = self.state
            if self.tier is None or (self.tier > 0 and (st is None or (st.B, st.S, st.N) != (B, S, N))):
                if self.hf is not None:
                    self._select_tier(input_ids, N)       # first call of a shape: the untimed warmup
                    st = self.state
                else:
                    st = self._build(B, S, N, self.tier)
            if self.tier == 0:
                for step, _ in self._native(input_ids, N):
                    yield step
                return
            yield from self._stream(st, input_ids, N)
        finally:
            if gc_was:
                gc.enable()
