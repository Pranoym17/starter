"""Compile every engine kernel specialisation for sm_90 (H100) on a machine with no GPU.

A fake Triton driver reports an sm_90 target and turns launches into no-ops, so the engine's own
wrappers run with CPU tensors and trigger exactly the compilations the H100 container would run
(same argument specialisation). ptxas -v then reports registers / spills per kernel.

  bash lab/run.sh python lab/compile_check.py [--quick]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import traceback

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.runtime import driver as _driver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
PTXAS = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
H100_SMEM = 232448
COMPILED = []


class _Utils:
    @staticmethod
    def get_device_properties(device):
        return {"max_shared_mem": H100_SMEM, "multiprocessor_count": 132, "sm_clock_rate": 1980000,
                "mem_clock_rate": 2619000, "mem_bus_width": 5120}

    @staticmethod
    def load_binary(name, kernel, shared, device):
        return (None, None, 0, 0)


class FakeDriver:
    utils = _Utils()

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def set_current_device(self, device):
        pass

    def get_current_target(self):
        return GPUTarget("cuda", 90, 32)

    def get_device_interface(self):
        return torch.cuda

    def get_benchmarker(self):
        return None

    def launcher_cls(self, src, metadata):
        name = getattr(src, "name", None) or getattr(getattr(src, "fn", None), "__name__", "?")
        COMPILED.append((name, metadata))
        return lambda *a, **k: None

    @staticmethod
    def is_active():
        return True


_driver.set_active(FakeDriver())

from kernels import decode_attn, fast, fused, gemv, mega, rmsnorm  # noqa: E402

BF16 = torch.bfloat16


def t(*shape, dtype=BF16):
    return torch.zeros(shape, dtype=dtype)


def cases(quick):
    H, I, V, NQ, NKV, D = 2560, 9728, 151936, 32, 8, 128
    batches = [1, 4, 16] if quick else [1, 2, 3, 4, 8, 16, 32, 64]
    for B in batches:
        S, N = 512, 32
        cap = S + N + 7
        yield f"rmsnorm hidden B={B}", lambda B=B: rmsnorm.rms_norm_rows(t(B, H), t(H), 1e-6)
        yield f"rmsnorm heads B={B}", lambda B=B: rmsnorm.rms_norm_rows(t(B, 6144)[:, 4096:], t(D), 1e-6, NKV)
        yield f"decode_attn B={B}", lambda B=B, cap=cap: decode_attn.DecodeAttention(B, NKV, cap, "cpu", 132)(
            t(B, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D), t(1, dtype=torch.int64), t(B, NQ, D))
        for mode in (0, 1):
            T = S if mode == 0 else 1
            yield f"qk_norm_rope mode={mode} B={B}", lambda B=B, T=T, mode=mode, cap=cap: fused.qk_norm_rope_cache(
                t(B * T, 6144), t(D), t(D), t(cap, D), t(cap, D), t(1, dtype=torch.int64), t(B, NQ, T, D),
                t(B, NKV, cap, D), t(B, NKV, cap, D), B, T, 1e-6, mode)
        yield f"silu_mul B={B}", lambda B=B: fused.silu_mul(t(B * 4, 2 * I), I)
        for M in sorted({B, B * 5, B * 7} if B <= 12 else {B}):
            if M > 64:
                continue
            for cfg in gemv.Gemv.CANDIDATES:
                for name, n, k, epi, norm in (("qkv", 6144, H, gemv.EPI_NONE, True), ("o", H, 4096, gemv.EPI_RES, False),
                                               ("gu", I, H, gemv.EPI_SILU, True), ("down", H, I, gemv.EPI_RES, False),
                                               ("qkv-nonorm", 6144, H, gemv.EPI_NONE, False),
                                               ("gu-nonorm", I, H, gemv.EPI_SILU, False)):
                    if k % cfg[1]:
                        continue
                    wn = 2 * n if epi == gemv.EPI_SILU else n

                    def run(M=M, n=n, k=k, epi=epi, norm=norm, cfg=cfg, wn=wn):
                        g = gemv.Gemv(M, n, k, epi, "cpu", 132, cfg=cfg, norm_eps=1e-6 if norm else None)
                        g(t(M, k), t(wn, k), t(M, n), t(M, n) if epi == gemv.EPI_RES else None,
                          t(k) if norm else None)
                    yield f"gemv {name} M={M} cfg={cfg}", run
            for cfg in gemv.LmHeadArgmax.CANDIDATES:
                for norm in (True, False):
                    def run_lm(M=M, cfg=cfg, norm=norm):
                        lm = gemv.LmHeadArgmax(M, V, H, "cpu", norm_eps=1e-6 if norm else None)
                        lm.set_config(cfg)
                        lm(t(M, H), t(V, H), t(M, dtype=torch.int64), t(H) if norm else None)
                    yield f"lm_head M={M} cfg={cfg} norm={norm}", run_lm
        # T5
        yield f"fast embed B={B}", lambda B=B: fast.embed_ss(t(B, dtype=torch.int64), t(V, H), t(B, H),
                                                              t(73, B, dtype=torch.float32))
        for name, n, k, epi, norm, sso in (("qkv", 6144, H, fast.EPI_NONE, True, False),
                                            ("o", H, 4096, fast.EPI_RES, False, True),
                                            ("gu", I, H, fast.EPI_SILU, True, False),
                                            ("down", H, I, fast.EPI_RES, False, True)):
            for base in fast.FastGemm.BASE:
                if k % base[1]:
                    continue
                for split in (1, 3):
                    def run_fast(B=B, n=n, k=k, epi=epi, norm=norm, sso=sso, cfg=base + (split,)):
                        g = fast.FastGemm(B, n, k, epi, "cpu", norm=norm, ss_out=sso)
                        g.set_config(cfg)
                        wn = 2 * n if epi == fast.EPI_SILU else n
                        g(t(B, k), t(wn, k), t(B, n), res=t(B, n) if epi == fast.EPI_RES else None,
                          norm_w=t(k) if norm else None, ss_in=t(B, dtype=torch.float32) if norm else None,
                          ss_out=t(B, dtype=torch.float32) if sso else None)
                    yield f"fast gemm {name} B={B} cfg={base + (split,)}", run_fast
        yield f"fast attn B={B}", lambda B=B, cap=cap: fast.FastAttention(B, NKV, cap, "cpu", 132)(
            t(B, 6144), t(D), t(D), t(cap, D), t(cap, D), t(1, dtype=torch.int64), t(B, NKV, cap, D),
            t(B, NKV, cap, D), t(B, NQ, D), 1e-6)
        for cfg in fast.FastLmHead.CANDIDATES:
            def run_flm(B=B, cfg=cfg):
                lm = fast.FastLmHead(B, V, H, "cpu")
                lm.set_config(cfg)
                lm(t(B, H), t(V, H), t(B, dtype=torch.int64), t(H), t(B, dtype=torch.float32),
                   t(1, dtype=torch.int64), True)
            yield f"fast lm B={B} cfg={cfg}", run_flm
        # T6 persistent megakernel (only constexprs matter for compilation: small NL / V are fine)
        def run_mega(B=B, cap=cap):
            from types import SimpleNamespace
            from kernels import mega
            NL, Vs = 1, 1024
            eng = SimpleNamespace(embed=t(Vs, H), inter=I, lm_head=t(Vs, H), w_qkv_all=t(NL, 6144, H),
                                  w_o_all=t(NL, H, 4096), w_gu_all=t(NL, 2 * I, H), w_d_all=t(NL, H, I),
                                  ln1_all=t(NL, H), ln2_all=t(NL, H), qn_all=t(NL, D), kn_all=t(NL, D),
                                  final_norm=t(H), layers=[None] * NL, eps=1e-6, sm_count=132)
            st = SimpleNamespace(B=B, capacity=cap, k_all=t(NL, B, NKV, cap, D), v_all=t(NL, B, NKV, cap, D),
                                 tok=t(B, dtype=torch.int64), pos=t(1, dtype=torch.int64), x=t(B, H),
                                 ss=t(2 * NL + 1, B, dtype=torch.float32), qkv_buf=t(B, 6144),
                                 attn_out=t(B, NQ, D), act_buf=t(B, I), cos=t(cap, D), sin=t(cap, D))
            m = mega.MegaStep(eng, st)
            m()
            print(f"    mega B={B}: {m.describe()}")
        yield f"mega B={B}", run_mega
        # tree speculation kernels (T = 16 for B = 1, 8 for B <= 8)
        Tt = 16 if B == 1 else 8
        if B * Tt <= 64:
            from kernels import tree

            def run_tree(B=B, T=Tt, cap=cap):
                tree.tree_qk(t(B * T, 6144), t(D), t(D), t(cap, D), t(cap, D), t(B, dtype=torch.int64),
                             t(B * T, dtype=torch.int64), t(B, T, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D),
                             B, T, 1e-6)
                tree.TreeAttention(B, T, NKV, cap, "cpu", 132)(t(B, T, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D),
                                                               t(B, dtype=torch.int64),
                                                               t(B, T, dtype=torch.int32), t(B, T, NQ, D))
                tree.compact(t(2, B, NKV, cap, D), t(2, B, NKV, cap, D), t(B, dtype=torch.int64),
                             t(B, T, dtype=torch.int32), t(B, dtype=torch.int32), T)
            yield f"tree B={B} T={Tt}", run_tree
        for T in (5, 7):
            if B * T > 64:
                continue
            yield f"verify_attn B={B} T={T}", lambda B=B, T=T, cap=cap: decode_attn.VerifyAttention(
                B, T, NKV, cap, "cpu", 132)(t(B, T, NQ, D), t(B, NKV, cap, D), t(B, NKV, cap, D),
                                           t(B, dtype=torch.int64), t(B, T, NQ, D))
            yield f"qk_norm_rope mode=2 B={B} T={T}", lambda B=B, T=T, cap=cap: fused.qk_norm_rope_cache(
                t(B * T, 6144), t(D), t(D), t(cap, D), t(cap, D), t(B, dtype=torch.int64), t(B, T, NQ, D),
                t(B, NKV, cap, D), t(B, NKV, cap, D), B, T, 1e-6, 2, q_strides=(T * 4096, D, 4096))


def ptxas_report(kernel_cache):
    """registers / spills per compiled kernel via ptxas -v on its PTX."""
    rows = []
    for fn in kernel_cache:
        for dev_cache in fn.cache.values():
            for key, k in dev_cache.items():
                ptx = k.asm.get("ptx")
                if not ptx:
                    continue
                with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
                    f.write(ptx)
                out = subprocess.run([PTXAS, "-arch=sm_90a", "-v", f.name, "-o", os.devnull],
                                     capture_output=True, text=True).stderr
                os.unlink(f.name)
                regs = re.search(r"Used (\d+) registers", out)
                spill = re.search(r"(\d+) bytes spill stores", out)
                rows.append((k.name, int(regs.group(1)) if regs else -1, int(spill.group(1)) if spill else -1,
                             k.metadata.shared, k.metadata.num_warps))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--regs", action="store_true", help="run ptxas -v on every compiled kernel")
    args = ap.parse_args()
    fails = 0
    n = 0
    for name, fn in cases(args.quick):
        n += 1
        try:
            fn()
        except Exception as exc:
            fails += 1
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}")
            if fails <= 3:
                traceback.print_exc(limit=6)
    print(f"{n} launch sites, {len(COMPILED)} kernels compiled for sm_90, {fails} failures")
    if args.regs:
        jits = [decode_attn._attn_partial_kernel, decode_attn._attn_merge_kernel,
                decode_attn._attn_verify_partial_kernel, fused._qk_norm_rope_kernel, fused._silu_mul_kernel,
                gemv._gemv_kernel, gemv._splitk_reduce_kernel, gemv._lm_argmax_partial_kernel,
                gemv._lm_argmax_reduce_kernel, rmsnorm._rms_norm_rows_kernel, fast._gemm_kernel,
                fast._attn_fused_kernel, fast._lm_kernel, fast._embed_ss_kernel, mega._mega_kernel]
        rows = ptxas_report(jits)
        worst = sorted(rows, key=lambda r: (-r[2], -r[1]))[:25]
        print(f"{'kernel':<34}{'regs':>6}{'spill B':>9}{'smem':>8}{'warps':>6}")
        for r in worst:
            print(f"{r[0]:<34}{r[1]:>6}{r[2]:>9}{r[3]:>8}{r[4]:>6}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
