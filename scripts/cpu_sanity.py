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


def main():
    rng = random.Random(0)
    mk = lambda b, s: [[rng.randrange(1000) for _ in range(s)] for _ in range(b)]  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        make_model(d)
        mod = load_engine()
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
