"""Time sm_90 compilation of the big kernels (compile happens on the CPU at warmup)."""
import tempfile as _tf, os as _os; _os.environ["TRITON_CACHE_DIR"] = _tf.mkdtemp()
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile_check as cc  # installs the fake sm_90 driver
import torch
from types import SimpleNamespace
from kernels import mega, fast
t, H, I, D = cc.t, 2560, 9728, 128
for B in (1, 16):
    cap = 551
    eng = SimpleNamespace(embed=t(1024, H), inter=I, lm_head=t(1024, H), w_qkv_all=t(1, 6144, H), w_o_all=t(1, H, 4096),
                          w_gu_all=t(1, 2 * I, H), w_d_all=t(1, H, I), ln1_all=t(1, H), ln2_all=t(1, H), qn_all=t(1, D),
                          kn_all=t(1, D), final_norm=t(H), layers=[None], eps=1e-6, sm_count=132)
    st = SimpleNamespace(B=B, capacity=cap, k_all=t(1, B, 8, cap, D), v_all=t(1, B, 8, cap, D), tok=t(B, dtype=torch.int64),
                         pos=t(1, dtype=torch.int64), x=t(B, H), ss=t(3, B, dtype=torch.float32), qkv_buf=t(B, 6144),
                         attn_out=t(B, 32, D), act_buf=t(B, I), cos=t(cap, D), sin=t(cap, D))
    t0 = time.time(); mega.MegaStep(eng, st)(); print(f"megakernel B={B}: {time.time() - t0:.1f}s")
    g = fast.FastGemm(B, 9728, H, fast.EPI_SILU, "cpu", norm=True)
    t0 = time.time(); g(t(B, H), t(2 * 9728, H), t(B, 9728), norm_w=t(H), ss_in=t(B, dtype=torch.float32)); print(f"fast gemm gu B={B}: {time.time() - t0:.1f}s")
