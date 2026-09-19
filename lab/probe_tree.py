import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
which = sys.argv[1] if len(sys.argv) > 1 else ""
if which:
    import compile_check as cc
    import torch
    from kernels import tree
    t, D, NKV, NQ, cap = cc.t, 128, 8, 32, 551
    B, T = (1, 16) if which.endswith("b1") else (4, 8)
    if which.startswith("qk"):
        tree.tree_qk(t(B * T, 6144), t(D), t(D), t(cap, D), t(cap, D), t(B, dtype=torch.int64), t(B * T, dtype=torch.int64),
                     t(B, T, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D), B, T, 1e-6)
    elif which.startswith("attn"):
        tree.TreeAttention(B, T, NKV, cap, "cpu", 132)(t(B, T, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D),
                                                       t(B, dtype=torch.int64), t(B, T, dtype=torch.int32), t(B, T, NQ, D))
    else:
        tree.compact(t(2, B, NKV, cap, D), t(2, B, NKV, cap, D), t(B, dtype=torch.int64), t(B, T, dtype=torch.int32),
                     t(B, dtype=torch.int32), T)
    print("OK", which)
else:
    for w in ("qk_b1", "attn_b1", "compact_b1", "qk_b4", "attn_b4", "compact_b4"):
        r = subprocess.run([sys.executable, __file__, w], capture_output=True, text=True)
        print(w, "ok" if r.returncode == 0 else f"ABORT rc={r.returncode}: {(r.stderr.strip().splitlines() or [''])[0][:120]}")
