"""Lab-only fixes for gaps in Triton's CPU interpreter (never imported by the engine).

The interpreter stores BF16 as raw uint16 bits and (a) converts fp32<->bf16 with a buggy/rounding-
mode-less routine (truncation, or "RTNE" without exponent carry), (b) runs tl.dot and elementwise ops
directly on the uint16 bit patterns. The GPU computes BF16 products/dots exactly in fp32 and rounds to
nearest-even when storing BF16. These shims make the interpreter do the same, plus:
  - `range(a, b, c)` with runtime scalars (tensor.__index__ on size-1 arrays),
  - libdevice.exp -> tl.exp (no interpreter implementation).
"""
import types

import numpy as np
import triton.language as tl
from triton.runtime import interpreter as _it


def bf16_to_f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    f = np.asarray(f32, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r = np.where(np.isnan(f), np.uint16(0x7FC0), r)
    return r.reshape(f.shape)


_orig_convert = _it._convert_float


def _convert_float(input, input_dtype, output_dtype, rounding_mode):
    if input_dtype == tl.float32 and output_dtype == tl.bfloat16:
        return f32_to_bf16(np.asarray(input).view(np.float32))
    if input_dtype == tl.bfloat16 and output_dtype == tl.float32:
        return bf16_to_f32(np.asarray(input).view(np.uint16)).view(np.uint32)
    return _orig_convert(input, input_dtype, output_dtype, rounding_mode)


_it._convert_float = _convert_float

_Builder = next(v for v in vars(_it).values() if isinstance(v, type) and hasattr(v, "create_dot"))
_orig_binary = _Builder.binary_op
_orig_dot = _Builder.create_dot


def _is_bf16(h):
    return getattr(h, "dtype", None) == tl.bfloat16


def _binary_op(self, lhs, rhs, op):
    if _is_bf16(lhs) or _is_bf16(rhs):
        a = bf16_to_f32(lhs.data) if _is_bf16(lhs) else lhs.data.astype(np.float32)
        b = bf16_to_f32(rhs.data) if _is_bf16(rhs) else rhs.data.astype(np.float32)
        return _it.TensorHandle(f32_to_bf16(op(a, b).astype(np.float32)), tl.bfloat16)
    return _orig_binary(self, lhs, rhs, op)


def _create_dot(self, a, b, d, *args, **kw):
    if _is_bf16(a) or _is_bf16(b):
        a = _it.TensorHandle(bf16_to_f32(a.data), tl.float32) if _is_bf16(a) else a
        b = _it.TensorHandle(bf16_to_f32(b.data), tl.float32) if _is_bf16(b) else b
    return _orig_dot(self, a, b, d, *args, **kw)


_Builder.binary_op = _binary_op
_Builder.create_dot = _create_dot
for _name in ("create_fadd", "create_fmul", "create_fdiv", "create_fsub", "create_precise_divf"):
    _op = {"create_fadd": np.add, "create_fmul": np.multiply, "create_fdiv": np.divide,
           "create_fsub": np.subtract, "create_precise_divf": np.divide}[_name]
    setattr(_Builder, _name, (lambda op: lambda self, lhs, rhs: _binary_op(self, lhs, rhs, op))(_op))

_orig_patch_tensor = _it._patch_lang_tensor


def _patch_lang_tensor(tensor):
    _orig_patch_tensor(tensor)
    tensor.__index__ = lambda self: int(np.asarray(self.handle.data).reshape(-1)[0])


_it._patch_lang_tensor = _patch_lang_tensor


def patch_libdevice(*modules):
    for m in modules:
        if hasattr(m, "libdevice"):
            m.libdevice = types.SimpleNamespace(exp=lambda x: tl.exp(x))
