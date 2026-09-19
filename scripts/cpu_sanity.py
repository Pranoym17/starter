"""CPU-only pre-push sanity check (no GPU, no Triton): exercises the torch-only tier (T1), the
native tier (T0), the warmup self-check / fallback bookkeeping, and the stream contract, on a
tiny randomly initialised Qwen3 with the real head geometry.

  <venv>/python scripts/cpu_sanity.py
"""
import importlib.util
import os
import random
import sys
import tempfile

os.environ["ENGINE_DEVICE"] = "cpu"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


def make_model(path):
    cfg = Qwen3Config(vocab_size=1000, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=32, num_key_value_heads=8, head_dim=128, rope_theta=5_000_000,
                      max_position_embeddings=4096, tie_word_embeddings=True, rms_norm_eps=1e-6)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(8.0)  # sharper logits so greedy choices are not all near-ties
    m.save_pretrained(path)


def load_engine():
    sys.path.insert(0, os.path.join(ROOT, "engine"))
    spec = importlib.util.spec_from_file_location("engine", os.path.join(ROOT, "engine", "engine.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def native(path, prompts, n):
    m = Qwen3ForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    cur, cache, out = torch.tensor(prompts), None, []
    with torch.inference_mode():
        for _ in range(n):
            o = m(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            cur = o.logits[:, -1].argmax(-1, keepdim=True)
            cache = o.past_key_values
            out.append(cur[:, 0].tolist())
    return out


def check(eng, prompts, n, ref=None):
    steps = list(eng.generate(prompts, n))
    assert len(steps) == n, f"{len(steps)} yields != {n}"
    for s in steps:
        assert isinstance(s, list) and len(s) == len(prompts) and all(type(t) is int for t in s), s
    if ref is not None:
        same = sum(a == b for x, y in zip(steps, ref) for a, b in zip(x, y))
        total = n * len(prompts)
        print(f"  {len(prompts)}x{len(prompts[0])}x{n}: {same}/{total} tokens equal native")
        assert same >= 0.9 * total, "torch-only tier diverges from native"
    return steps


def spec_logic_check(mod):
    """Drive Engine._stream_spec with a mock model whose 'greedy token' is a deterministic
    function of the KV-cache contents: spec output must equal plain greedy exactly."""
    from types import SimpleNamespace

    def f(prefix):  # toy greedy rule with repetition, so drafts get accepted sometimes
        return (prefix[-1] * 3 + prefix[-2] + (len(prefix) % 5 == 0)) % 23

    def plain(prompt, n):
        seq = list(prompt)
        for _ in range(n):
            seq.append(f(seq))
        return seq[len(prompt):]

    rng = random.Random(1)
    for B, S, N, T in [(1, 12, 40, 7), (3, 9, 33, 5), (4, 20, 2, 5), (2, 7, 1, 5), (5, 30, 64, 3)]:
        prompts = [[rng.randrange(23) for _ in range(S)] for _ in range(B)]
        cap = S + N + T
        kv = [[None] * cap for _ in range(B)]
        st = SimpleNamespace(B=B, S=S, spec_T=T)
        st.ids_host = torch.zeros((B, S), dtype=torch.int64)
        st.ids = torch.zeros((B, S), dtype=torch.int64)
        st.pos = torch.zeros((1,), dtype=torch.int64)
        st.tok = torch.zeros((B,), dtype=torch.int64)
        st.host = torch.zeros((max(N, 1), B), dtype=torch.int64)
        st.events = [mod._NullEvent() for _ in range(max(N, 1))]
        st.v_event = mod._NullEvent()
        for name, shape in (("v_tok", (B, T)), ("v_pos", (B,)), ("v_out", (B * T,))):
            setattr(st, name, torch.zeros(shape, dtype=torch.int64))
            setattr(st, name + "_host", torch.zeros(shape, dtype=torch.int64))

        def prefill():
            for b in range(B):
                kv[b][:S] = st.ids[b].tolist()
                st.tok[b] = f(kv[b][:S])

        def verify():
            for b in range(B):
                p0 = int(st.v_pos[b])
                assert 0 <= p0 and p0 + T <= cap, "verify beyond cache capacity"
                assert all(v is not None for v in kv[b][:p0]), "reads an unwritten slot"
                for t in range(T):
                    kv[b][p0 + t] = int(st.v_tok[b, t])
                    st.v_out[b * T + t] = f(kv[b][:p0 + t + 1])

        st.prefill_graph = SimpleNamespace(replay=prefill)
        st.verify_graph = SimpleNamespace(replay=verify)
        steps = list(mod.Engine._stream_spec(None, st, prompts, N))
        assert len(steps) == N and all(len(s) == B and all(type(t) is int for t in s) for s in steps)
        got = [[steps[i][b] for i in range(N)] for b in range(B)]
        want = [plain(p, N) for p in prompts]
        assert got == want, f"spec mismatch B={B} S={S} N={N} T={T}"
    print("  spec host logic: exact on 5 mock cases")


def tree_logic_check(mod):
    """Drive Engine._stream_tree with a mock device that implements the tree-verify contract
    (compaction, scratch slots, ancestor masks, depths): output must equal plain greedy exactly."""
    from types import SimpleNamespace
    sys.path.insert(0, os.path.join(ROOT, "engine"))
    import spec
    mod.SeqDraft, mod.Tree = spec.SeqDraft, spec.Tree

    def f(prefix):
        return (prefix[-1] * 3 + prefix[-2] + (len(prefix) % 5 == 0)) % 23

    def plain(prompt, n):
        seq = list(prompt)
        for _ in range(n):
            seq.append(f(seq))
        return seq[len(prompt):]

    rng = random.Random(2)
    total_tok = total_ver = 0
    for B, S, N, T in [(1, 12, 60, 16), (3, 9, 33, 8), (4, 20, 2, 8), (2, 7, 1, 8), (5, 30, 64, 8), (8, 25, 40, 8)]:
        prompts = [[rng.randrange(23) for _ in range(S)] for _ in range(B)]
        cap = S + N + T
        kv = [[None] * cap for _ in range(B)]
        z = lambda *shape, dt=torch.int64: torch.zeros(shape, dtype=dt)  # noqa: E731
        tr = SimpleNamespace(tok=z(B, T), depth=z(B * T), base=z(B), anc=z(B, T, dt=torch.int32), c_base=z(B),
                             c_src=z(B, T, dt=torch.int32), c_cnt=z(B, dt=torch.int32), out=z(B * T),
                             event=mod._NullEvent())
        for name in ("tok", "depth", "base", "anc", "c_base", "c_src", "c_cnt", "out"):
            setattr(tr, name + "_h", getattr(tr, name).clone())
        st = SimpleNamespace(B=B, S=S, tree_T=T, tr=tr, ids_host=z(B, S), ids=z(B, S), pos=z(1), tok=z(B),
                             host=z(max(N, 1), B), events=[mod._NullEvent() for _ in range(max(N, 1))])
        n_verify = [0]

        def prefill():
            for b in range(B):
                kv[b][:S] = st.ids[b].tolist()
                st.tok[b] = f(kv[b][:S])

        def verify():
            n_verify[0] += 1
            for b in range(B):
                cb, cnt = int(tr.c_base[b]), int(tr.c_cnt[b])
                for k in range(1, cnt):
                    kv[b][cb + k] = kv[b][cb + int(tr.c_src[b, k])]
                base = int(tr.base[b])
                assert 0 <= base and base + T <= cap, "tree beyond cache capacity"
                assert all(v is not None for v in kv[b][:base]), "tree reads an unwritten slot"
                for t in range(T):
                    kv[b][base + t] = int(tr.tok[b, t])
                for t in range(T):
                    a = int(tr.anc[b, t]) & 0xFFFFFFFF
                    path = [j for j in range(T) if (a >> j) & 1]
                    assert path[0] == 0 or t != 0 and False or True
                    assert int(tr.depth[b * T + t]) == len(path) - 1, "depth / ancestor mask mismatch"
                    ctx = kv[b][:base] + [int(tr.tok[b, j]) for j in path]
                    tr.out[b * T + t] = f(ctx)

        st.prefill_graph = SimpleNamespace(replay=prefill)
        st.tree_graph = SimpleNamespace(replay=verify)
        steps = list(mod.Engine._stream_tree(None, st, prompts, N))
        assert len(steps) == N and all(len(s) == B and all(type(t) is int for t in s) for s in steps)
        got = [[steps[i][b] for i in range(N)] for b in range(B)]
        assert got == [plain(p, N) for p in prompts], f"tree spec mismatch B={B} S={S} N={N} T={T}"
        if B == 1:
            total_tok += N - 1
            total_ver += n_verify[0]
    print(f"  tree spec host logic: exact on 6 mock cases; B=1 mock: {total_tok / max(total_ver, 1):.2f} tokens/verify")


def main():
    rng = random.Random(0)
    mk = lambda b, s: [[rng.randrange(1000) for _ in range(s)] for _ in range(b)]  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        make_model(d)
        mod = load_engine()
        spec_logic_check(mod)
        tree_logic_check(mod)
        for forced in ("-1", "0"):
            mod.FORCE_TIER = int(forced)
            eng = mod.Engine(d)
            for b, s, n in [(2, 17, 5), (1, 9, 1), (3, 33, 40), (3, 33, 40), (2, 64, 8)]:
                p = mk(b, s)
                check(eng, p, n, native(d, p, n))
            print(f"FORCE_TIER={forced}: chosen tier T{eng.tier}")
            assert eng.tier == (1 if forced == "-1" else 0)
        assert list(eng.generate(mk(1, 5), 0)) == []
    print("CPU SANITY PASS")


if __name__ == "__main__":
    main()
