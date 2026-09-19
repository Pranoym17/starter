"""Numerical tests of every engine kernel under the Triton CPU interpreter (TRITON_INTERPRET=1),
against torch references that follow transformers 4.51.3's arithmetic, plus an end-to-end tiny-model
run of the T4 decode path and the speculative verify path against native HF greedy.

  bash lab/run.sh python lab/interp_test.py [-k name]
"""
import argparse
import math
import os
import random
import sys
import tempfile
import time
import traceback

os.environ["TRITON_INTERPRET"] = "1"
os.environ["ENGINE_DEVICE"] = "cpu"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
from kernels import decode_attn, fused, gemv, rmsnorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interp_shims  # noqa: E402

from kernels import fast as _fastk  # noqa: E402
from kernels import mega as _megak  # noqa: E402
interp_shims.patch_libdevice(fused, gemv, _fastk, _megak)

BF16 = torch.bfloat16
EPS = 1e-6
D, NQ, NKV = 128, 32, 8
torch.manual_seed(0)


def ref_norm(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return w * xf.to(BF16)


def rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), -1)


def rope_tables(cap):
    inv = 1.0 / (5_000_000 ** (torch.arange(0, D, 2).float() / D))
    f = torch.arange(cap).float()[:, None] * inv[None]
    emb = torch.cat((f, f), -1)
    return emb.cos().to(BF16), emb.sin().to(BF16)


def close(a, b, what, atol=0.0, ulps=2):
    """BF16 results may differ by reduction order: allow a few BF16 ulps (relative 2^-7 each)."""
    a, b = a.float(), b.float()
    tol = atol + ulps * 2 ** -7 * b.abs()
    bad = (a - b).abs() > tol + 1e-30
    if bad.any():
        i = bad.nonzero()[0].tolist()
        raise AssertionError(f"{what}: {int(bad.sum())} mismatches, e.g. at {i}: {a[tuple(i)].item()} vs {b[tuple(i)].item()}")


# ------------------------------------------------------------------ kernel tests
def test_rmsnorm():
    x = (torch.randn(5, 6144) * 3).to(BF16)
    w = (torch.rand(256) + 0.5).to(BF16)
    close(rmsnorm.rms_norm_rows(x[:, :256].contiguous(), w, EPS).view(5, 256), ref_norm(x[:, :256], w), "hidden")
    wh = (torch.rand(D) + 0.5).to(BF16)
    got = rmsnorm.rms_norm_rows(x[:, 4096:], wh, EPS, NKV)
    close(got, ref_norm(x[:, 4096:5120].reshape(5, NKV, D), wh), "k heads")
    got = rmsnorm.rms_norm_rows(x, wh, EPS, NQ)
    close(got, ref_norm(x[:, :4096].reshape(5, NQ, D), wh), "q heads")


def _qk_ref(qkv, qn, kn, cos, sin, positions, B, T):
    q = ref_norm(qkv[:, :4096].reshape(B, T, NQ, D), qn)
    k = ref_norm(qkv[:, 4096:5120].reshape(B, T, NKV, D), kn)
    v = qkv[:, 5120:].reshape(B, T, NKV, D)
    c = cos[positions][:, :, None]          # [B, T, 1, D]
    s = sin[positions][:, :, None]
    return (q * c) + (rot(q) * s), (k * c) + (rot(k) * s), v


def test_qk_norm_rope():
    cap = 40
    cos, sin = rope_tables(cap)
    qn, kn = (torch.rand(D) + 0.5).to(BF16), (torch.rand(D) + 0.5).to(BF16)
    for mode, B, T in ((0, 2, 6), (1, 3, 1), (2, 3, 5)):
        qkv = (torch.randn(B * T, 6144) * 2).to(BF16)
        if mode == 0:
            pos = torch.zeros(1, dtype=torch.int64)
            positions = torch.arange(T)[None].expand(B, T)
        elif mode == 1:
            pos = torch.tensor([17])
            positions = (17 + torch.arange(T))[None].expand(B, T)
        else:
            pos = torch.tensor([3, 20, 31])
            positions = pos[:, None] + torch.arange(T)[None]
        kc = torch.full((B, NKV, cap, D), 7.0, dtype=BF16)
        vc = torch.full((B, NKV, cap, D), 7.0, dtype=BF16)
        if mode == 2:
            q_out = torch.empty(B, T, NQ, D, dtype=BF16)
            fused.qk_norm_rope_cache(qkv, qn, kn, cos, sin, pos, q_out, kc, vc, B, T, EPS, 2,
                                     q_strides=(T * 4096, D, 4096))
            q_got = q_out
        else:
            q_out = torch.empty(B, NQ, T, D, dtype=BF16)
            fused.qk_norm_rope_cache(qkv, qn, kn, cos, sin, pos, q_out, kc, vc, B, T, EPS, mode)
            q_got = q_out.transpose(1, 2)
        q_ref, k_ref, v_ref = _qk_ref(qkv, qn, kn, cos, sin, positions, B, T)
        close(q_got, q_ref, f"q mode={mode}")
        for b in range(B):
            for t in range(T):
                p = int(positions[b, t])
                close(kc[b, :, p], k_ref[b, t], f"k cache mode={mode}")
                close(vc[b, :, p], v_ref[b, t], f"v cache mode={mode}", ulps=0)
        written = torch.zeros(B, cap, dtype=torch.bool)
        for b in range(B):
            written[b, positions[b]] = True
        assert (kc.transpose(1, 2)[~written] == 7.0).all(), "qk kernel wrote outside its positions"


def test_silu_mul():
    gu = (torch.randn(3, 2 * 1100) * 4).to(BF16)
    ref = F.silu(gu[:, :1100]) * gu[:, 1100:]
    close(fused.silu_mul(gu, 1100), ref, "silu_mul", ulps=1)


def _attn_ref(q, kc, vc, limits):
    """q [R, 4, D] rows for one (b, kvh); keys 0..limit[r] inclusive, fp32 softmax."""
    out = []
    for r in range(q.shape[0]):
        k = kc[: limits[r] + 1].float()
        v = vc[: limits[r] + 1].float()
        s = (q[r].float() @ k.T) / math.sqrt(D)
        p = torch.softmax(s, -1)
        out.append(p @ v)
    return torch.stack(out)


def test_decode_attention():
    for B, cap, pos_v, sm in ((2, 50, 0, 132), (1, 300, 299, 132), (3, 200, 130, 132), (2, 257, 64, 4)):
        q = torch.randn(B, NQ, D).to(BF16)
        kc = torch.randn(B, NKV, cap, D).to(BF16)
        vc = torch.randn(B, NKV, cap, D).to(BF16)
        pos = torch.tensor([pos_v])
        out = torch.empty(B, NQ, D, dtype=BF16)
        att = decode_attn.DecodeAttention(B, NKV, cap, "cpu", sm)
        att(q, kc, vc, pos, out)
        for b in range(B):
            for g in range(NKV):
                ref = _attn_ref(q[b, g * 4:(g + 1) * 4][:, None].squeeze(1)[:, None].reshape(4, 1, D).squeeze(1)[:, None],
                                kc[b, g], vc[b, g], [pos_v] * 4) if False else None
                qs = q[b, g * 4:(g + 1) * 4]
                ref = torch.stack([_attn_ref(qs[h:h + 1], kc[b, g], vc[b, g], [pos_v])[0] for h in range(4)])
                close(out[b, g * 4:(g + 1) * 4], ref, f"decode attn B={B} cap={cap} pos={pos_v} splits={att.splits}",
                      atol=2e-2, ulps=4)


def test_verify_attention():
    for B, T, cap, poss, sm in ((1, 7, 120, [60], 132), (3, 5, 300, [0, 150, 290], 132), (2, 5, 257, [64, 200], 6)):
        q = torch.randn(B, T, NQ, D).to(BF16)
        kc = torch.randn(B, NKV, cap, D).to(BF16)
        vc = torch.randn(B, NKV, cap, D).to(BF16)
        pos = torch.tensor(poss)
        out = torch.empty(B, T, NQ, D, dtype=BF16)
        att = decode_attn.VerifyAttention(B, T, NKV, cap, "cpu", sm)
        att(q, kc, vc, pos, out)
        assert torch.isfinite(out.float()).all(), "verify attention produced non-finite values"
        for b in range(B):
            for t in range(T):
                for h in range(NQ):
                    g = h // 4
                    ref = _attn_ref(q[b, t, h][None], kc[b, g], vc[b, g], [poss[b] + t])[0]
                    close(out[b, t, h], ref, f"verify attn B={B} T={T} splits={att.splits} b={b} t={t}",
                          atol=2e-2, ulps=4)


def test_gemv():
    for M, N, K, epi, norm, cfg in ((1, 96, 256, gemv.EPI_NONE, True, (32, 128, 4, 4)),
                                    (3, 80, 512, gemv.EPI_RES, False, (16, 256, 4, 3)),
                                    (5, 64, 256, gemv.EPI_SILU, True, (32, 128, 4, 4)),
                                    (7, 48, 1024, gemv.EPI_RES, False, (16, 128, 4, 4)),
                                    (2, 40, 256, gemv.EPI_SILU, False, (64, 128, 4, 4))):
        for sm in (132, 4):             # sm=4 forces split-K on these small shapes
            x = torch.randn(M, K).to(BF16)
            wn = 2 * N if epi == gemv.EPI_SILU else N
            w = (torch.randn(wn, K) / math.sqrt(K)).to(BF16)
            nw = (torch.rand(K) + 0.5).to(BF16) if norm else None
            res = torch.randn(M, N).to(BF16) if epi == gemv.EPI_RES else None
            g = gemv.Gemv(M, N, K, epi, "cpu", sm, cfg=cfg, norm_eps=EPS if norm else None)
            out = torch.empty(M, N, dtype=BF16)
            g(x, w, out, res.clone() if res is not None else None, nw)
            xin = ref_norm(x, nw) if norm else x
            y = (xin.float() @ w.float().T)
            if epi == gemv.EPI_NONE:
                ref = y.to(BF16)
            elif epi == gemv.EPI_RES:
                ref = res + y.to(BF16)
            else:
                ref = F.silu(y[:, :N].to(BF16)) * y[:, N:].to(BF16)
            close(out, ref, f"gemv M={M} N={N} K={K} epi={epi} norm={norm} split={g.split}", atol=1e-2, ulps=2)
            # in-place residual (out aliases res), as the engine calls it
            if epi == gemv.EPI_RES:
                xr = res.clone()
                g(x, w, xr, xr, nw)
                close(xr, ref, f"gemv in-place residual split={g.split}", atol=1e-2, ulps=2)


def test_lm_head():
    for M, V, K, norm in ((1, 1000, 256, True), (5, 777, 256, False), (3, 2048, 512, True)):
        x = torch.randn(M, K).to(BF16)
        w = (torch.randn(V, K) / math.sqrt(K)).to(BF16)
        w[V // 2] = w[V // 3]                      # exact duplicate rows -> exact logit ties
        w[V - 1] = w[5]
        nw = (torch.rand(K) + 0.5).to(BF16) if norm else None
        for cfg in gemv.LmHeadArgmax.CANDIDATES:
            lm = gemv.LmHeadArgmax(M, V, K, "cpu", norm_eps=EPS if norm else None)
            lm.set_config(cfg)
            tok = torch.empty(M, dtype=torch.int64)
            lm(x, w, tok, nw)
            h = ref_norm(x, nw) if norm else x
            logits = (h.float() @ w.float().T).to(BF16)
            ref = torch.argmax(logits, -1)
            # reduction-order differences may flip genuine near-ties: accept any token whose bf16
            # logit equals the max (lowest index expected when exact ties exist)
            for m in range(M):
                lg = logits[m].float()
                assert lg[tok[m]] >= lg.max() - 2 ** -6 * lg.abs().max(), f"lm argmax row {m}: {tok[m]} vs {ref[m]}"


# ------------------------------------------------------------------ T5 kernels
def test_fast_embed_ss():
    from kernels import fast
    emb = torch.randn(50, 256).to(BF16)
    tok = torch.tensor([3, 49, 0])
    x = torch.empty(3, 256, dtype=BF16)
    ss = torch.full((5, 3), 9.0)
    fast.embed_ss(tok, emb, x, ss)
    assert torch.equal(x, emb[tok])
    close(ss[0], (emb[tok].float() ** 2).sum(-1), "ss0", ulps=0, atol=1e-3)
    assert (ss[1:] == 0).all()


def test_fast_gemm():
    from kernels import fast
    for M, N, K, epi, norm, ss_out in ((1, 96, 256, fast.EPI_NONE, True, False),
                                       (3, 80, 512, fast.EPI_RES, False, True),
                                       (5, 64, 256, fast.EPI_SILU, True, False),
                                       (7, 48, 1024, fast.EPI_RES, False, True)):
        for cfg in ((32, 128, 4, 4, 1), (16, 128, 4, 4, 2), (32, 256, 4, 3, 3), (64, 128, 4, 4, 4)):
            if K % cfg[1]:
                continue
            g = fast.FastGemm(M, N, K, epi, "cpu", norm=norm, ss_out=ss_out)
            g.set_config(cfg)
            x = torch.randn(M, K).to(BF16)
            wn = 2 * N if epi == fast.EPI_SILU else N
            w = (torch.randn(wn, K) / math.sqrt(K)).to(BF16)
            nw = (torch.rand(K) + 0.5).to(BF16) if norm else None
            ss_in = (x.float() ** 2).sum(-1) if norm else None
            res = torch.randn(M, N).to(BF16)
            ssb = torch.zeros(M) if ss_out else None
            out = res.clone() if epi == fast.EPI_RES else torch.empty(M, N, dtype=BF16)
            g(x, w, out, res=out if epi == fast.EPI_RES else None, norm_w=nw, ss_in=ss_in, ss_out=ssb)
            xin = ref_norm(x, nw) if norm else x
            y = xin.float() @ w.float().T
            if epi == fast.EPI_NONE:
                ref = y.to(BF16)
            elif epi == fast.EPI_RES:
                ref = res + y.to(BF16)
            else:
                ref = F.silu(y[:, :N].to(BF16)) * y[:, N:].to(BF16)
            what = f"fast gemm M={M} N={N} K={K} epi={epi} norm={norm} cfg={cfg} split={g.split}"
            close(out, ref, what, atol=1e-2, ulps=2)
            if ss_out:
                close(ssb, (out.float() ** 2).sum(-1), what + " ss_out", atol=1e-3, ulps=0)
            assert (g.ticket == 0).all(), what + ": ticket not reset"


def test_fast_attention():
    from kernels import fast
    cap = 200
    cos, sin = rope_tables(cap)
    qn, kn = (torch.rand(D) + 0.5).to(BF16), (torch.rand(D) + 0.5).to(BF16)
    for B, pos_v, sm in ((1, 1, 132), (2, 64, 132), (3, 130, 132), (2, 199, 132), (2, 150, 4)):
        qkv = (torch.randn(B, 6144) * 2).to(BF16)
        kc = torch.randn(B, NKV, cap, D).to(BF16)
        vc = torch.randn(B, NKV, cap, D).to(BF16)
        kc0, vc0 = kc.clone(), vc.clone()
        pos = torch.tensor([pos_v])
        out = torch.empty(B, NQ, D, dtype=BF16)
        att = fast.FastAttention(B, NKV, cap, "cpu", sm)
        att(qkv, qn, kn, cos, sin, pos, kc, vc, out, EPS)
        positions = torch.full((B, 1), pos_v)
        q_ref, k_ref, v_ref = _qk_ref(qkv, qn, kn, cos, sin, positions, B, 1)
        for b in range(B):
            close(kc[b, :, pos_v], k_ref[b, 0], "fast attn k write")
            close(vc[b, :, pos_v], v_ref[b, 0], "fast attn v write", ulps=0)
            other = torch.ones(cap, dtype=torch.bool)
            other[pos_v] = False
            assert torch.equal(kc[b][:, other], kc0[b][:, other]), "fast attn wrote outside pos"
            for g in range(NKV):
                for h in range(4):
                    ref = _attn_ref(q_ref[b, 0, g * 4 + h][None], kc[b, g], vc[b, g], [pos_v])[0]
                    close(out[b, g * 4 + h], ref, f"fast attn B={B} pos={pos_v} splits={att.splits}",
                          atol=2e-2, ulps=4)
        assert (att.ticket == 0).all(), "attention ticket not reset"


def test_fast_lm_head():
    from kernels import fast
    for M, V, K in ((1, 1000, 256), (4, 2053, 256)):
        x = torch.randn(M, K).to(BF16)
        w = (torch.randn(V, K) / math.sqrt(K)).to(BF16)
        w[V // 2] = w[V // 3]
        nw = (torch.rand(K) + 0.5).to(BF16)
        ss = (x.float() ** 2).sum(-1)
        for cfg in fast.FastLmHead.CANDIDATES:
            lm = fast.FastLmHead(M, V, K, "cpu")
            lm.set_config(cfg)
            tok = torch.empty(M, dtype=torch.int64)
            pos = torch.tensor([41])
            lm(x, w, tok, nw, ss, pos, True)
            logits = (ref_norm(x, nw).float() @ w.float().T).to(BF16).float()
            for m in range(M):
                assert logits[m, tok[m]] >= logits[m].max() - 2 ** -6 * logits[m].abs().max(), \
                    f"fast lm argmax row {m}"
                ties = (logits[m] == logits[m, tok[m]]).nonzero()
                assert tok[m] == ties.min(), "fast lm tie must pick the lowest index"
            assert int(pos) == 42 and int(lm.ticket) == 0


# ------------------------------------------------------------------ end to end (tiny model)
def _tiny_model(path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=5_000_000,
                      max_position_embeddings=4096, tie_word_embeddings=True, rms_norm_eps=1e-6)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(BF16)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(8.0)
    m.save_pretrained(path)
    return m.eval()


def _native(m, prompts, n):
    cur, cache, out = torch.tensor(prompts), None, []
    with torch.inference_mode():
        for _ in range(n):
            o = m(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            cur = o.logits[:, -1].argmax(-1, keepdim=True)
            cache = o.past_key_values
            out.append(cur[:, 0].tolist())
    return [list(r) for r in zip(*out)]


def test_end_to_end_t4_and_spec():
    from types import SimpleNamespace
    import importlib.util
    spec = importlib.util.spec_from_file_location("engine", os.path.join(ROOT, "engine", "engine.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as d:
        ref_model = _tiny_model(d)
        eng = mod.Engine(d)
        rng = random.Random(3)
        for B, S, N in ((1, 21, 9), (3, 16, 7)):
            prompts = [[rng.randrange(1000) for _ in range(S)] for _ in range(B)]
            want = _native(ref_model, prompts, N)
            # pre-seed tuned configs so no CUDA timing is needed
            for M in {B, B * 5, B * 7}:
                eng._tuned[M] = {"qkv": (32, 128, 4, 4), "o": (32, 128, 4, 4), "gu": (32, 128, 4, 4),
                                 "down": (16, 256, 4, 3), "lm": (64, 128, 4, 4)}
            for tier in (6, 5, 4, 3, 2):
                st = mod._State(eng, B, S, N, tier)
                st.prefill_graph = SimpleNamespace(replay=lambda st=st: eng._prefill(st))
                steps = []
                with torch.inference_mode():
                    st.ids.copy_(torch.tensor(prompts))
                    st.pos.fill_(S)
                    eng._prefill(st)
                    steps.append(st.tok.tolist())
                    for _ in range(N - 1):
                        eng._decode(st)
                        steps.append(st.tok.tolist())
                got = [list(r) for r in zip(*steps)]
                same = sum(a == b for x, y in zip(got, want) for a, b in zip(x, y))
                print(f"    T{tier} B={B} S={S} N={N}: {same}/{B * N} tokens equal native")
                assert same >= 0.9 * B * N, f"T{tier} diverges from native"
                if tier == 4 and st.spec_T:
                    st.verify_graph = SimpleNamespace(replay=lambda st=st: eng._verify(st))
                    with torch.inference_mode():
                        sp = list(mod.Engine._stream_spec(eng, st, prompts, N))
                    sgot = [list(r) for r in zip(*sp)]
                    same = sum(a == b for x, y in zip(sgot, got) for a, b in zip(x, y))
                    print(f"    T4+spec(k={st.spec_T - 1}) B={B}: {same}/{B * N} tokens equal plain T4")
                    assert same >= 0.9 * B * N, "spec diverges from plain T4"


TESTS = [test_rmsnorm, test_qk_norm_rope, test_silu_mul, test_decode_attention, test_verify_attention,
         test_gemv, test_lm_head, test_fast_embed_ss, test_fast_gemm, test_fast_attention, test_fast_lm_head,
         test_end_to_end_t4_and_spec]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="")
    args = ap.parse_args()
    fails = 0
    for t in TESTS:
        if args.k not in t.__name__:
            continue
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as exc:
            fails += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=8)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
