"""CUDA fused prefill indexer logits kernel (sm_120) for the DSV4 harness.

Exposes ``fp8_mqa_logits(q, k, k_scale, weights, ks, ke) -> logits`` and registers
it with the harness as ``cuda``.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # so `import harness` works

_CUDA_HOME = os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.2")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")

_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load

        _MOD = load(
            name="dsv4_prefill_logits_sm120",
            sources=[os.path.join(_HERE, "fused_logits.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_120a",
                "--use_fast_math",
                "-lineinfo",
                "-Xptxas=-v",
                "--expt-relaxed-constexpr",
            ],
            extra_cflags=["-O3"],
            verbose=bool(int(os.environ.get("DSV4_VERBOSE", "0"))),
            build_directory=_build_dir(),
        )
    return _MOD


def _build_dir():
    d = os.path.join(_HERE, "build")
    os.makedirs(d, exist_ok=True)
    return d


# Config id -> (BM, BN, WM, WN, STAGES, NMAJOR); see fused_logits.cu switch.
CFGS = {
    0: "BM64 BN128 W32x32 s3 nmajor",
    1: "BM64 BN128 W32x64 s3 nmajor",
    2: "BM64 BN256 W32x64 s3 nmajor",
    3: "BM128 BN128 W32x64 s3 nmajor",
    4: "BM64 BN256 W32x64 s4 nmajor",
    5: "BM128 BN256 W32x64 s3 nmajor",
    6: "BM64 BN128 W32x32 s3 mmajor",
    7: "BM64 BN256 W32x64 s3 mmajor",
    8: "BM64 BN512 W32x64 s3 nmajor",
    9: "BM32 BN256 W32x64 s3 nmajor",
    10: "BM64 BN256 W32x32 s3 nmajor (512t)",
    11: "BM128 BN256 W64x32 s3 nmajor (512t)",
    12: "BM64 BN256 W64x64 s3 nmajor (128t)",
    13: "BM64 BN256 W32x64 s5 nmajor",
    14: "BM128 BN256 W32x32 s3 nmajor (1024t)",
    15: "BM64 BN128 W64x64 s3 nmajor (64t)",
    16: "BM32 BN256 W32x32 s3 nmajor (256t)",
    17: "BM64 BN256 W32x32 s4 nmajor (512t)",
    18: "BM128 BN128 W32x32 s3 nmajor (512t)",
    19: "BM128 BN256 W32x64 s2 nmajor (512t)",
    20: "BM64 BN512 W32x64 s2 nmajor (512t)",
    21: "BM128 BN128 W32x64 s4 nmajor (256t)",
    22: "BM64 BN512 W32x128 s3 nmajor (256t)",
    23: "BM128 BN256 W32x128 s3 nmajor (256t)",
    24: "BHOIST BM64 BN256 W32x32 s3 (512t)",
    25: "BHOIST BM64 BN256 W32x64 s3 (256t)",
    26: "BHOIST BM64 BN128 W32x32 s3 (256t)",
    27: "BHOIST BM128 BN256 W32x64 s3 (512t)",
    28: "BHOIST BM64 BN512 W32x64 s3 (512t)",
    29: "BHOIST BM128 BN128 W32x32 s3 (512t)",
    30: "BHOIST BM64 BN256 W32x128 s3 (128t)",
    31: "BHOIST BM64 BN128 W32x64 s3 (128t)",
    32: "BHOIST BM32 BN256 W32x64 s3 (128t)",
    33: "BHOIST BM32 BN512 W32x64 s3 (256t)",
    34: "BHOIST BM64 BN256 W32x64 s4 (256t)",
    35: "BHOIST BM64 BN256 W32x64 s3 mmajor (256t)",
    36: "BHOIST BM32 BN256 W32x64 s4 (128t)",
}

DEFAULT_CFG = int(os.environ.get("DSV4_CFG", "25"))


def make(cfg: int):
    def fn(q, k, k_scale, weights, ks, ke):
        if isinstance(k, (tuple, list)):
            k, k_scale = k
        return _mod().fp8_mqa_logits_cuda(q, k, k_scale, weights, ks, ke, cfg)

    fn.__name__ = f"cuda_cfg{cfg}"
    return fn


def fp8_mqa_logits(q, k, k_scale, weights, ks, ke):
    """Fused fp8 MQA indexer logits. Returns [M, N] float32."""
    if isinstance(k, (tuple, list)):
        k, k_scale = k
    return _mod().fp8_mqa_logits_cuda(q, k, k_scale, weights, ks, ke, DEFAULT_CFG)


try:
    from harness import register

    register("cuda", fp8_mqa_logits)
    for _c in CFGS:
        register(f"cuda{_c}", make(_c))
except ImportError:  # harness not importable (standalone use)
    pass
