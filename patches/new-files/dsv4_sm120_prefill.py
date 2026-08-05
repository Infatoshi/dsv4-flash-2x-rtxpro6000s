"""sm_120 prefill Lightning-Indexer logits kernel (fused triton).

Replaces the chunked fp8_mqa_logits_torch fallback in
sparse_attn_indexer.py on sm_120: ~75x at 500k context (930ms -> 12.3ms
per 2048-query chunk), fp32 accumulation, rel err ~5e-7 vs fp32 golden.
Source of truth: opt/dsv4_sm120_prefill.py (copy in venv site-packages).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

INT32_MAX = tl.constexpr(2147483647)


@triton.jit
def _prefill_logits_kernel(
    q_ptr,        # fp8e4m3 [M, H, D] contiguous
    k_ptr,        # fp8e4m3 [N, D]    contiguous
    kscale_ptr,   # fp32    [N]
    w_ptr,        # fp32    [M, H], or [H, M] when W_T
    ks_ptr,       # int32   [M]
    ke_ptr,       # int32   [M]
    out_ptr,      # fp32    [M, N]
    M,
    N,
    num_m,
    num_n,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M_FAST: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    W_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if M_FAST:
        pid_m = pid % num_m
        pid_n = pid // num_m
    else:
        pid_n = pid % num_n
        pid_m = pid // num_n

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    rm = tl.arange(0, BLOCK_M)
    offs_m = m0 + rm
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Ragged edges are handled by CLAMPING every input index into range rather
    # than predicating the loads: a tail lane then reads a valid-but-irrelevant
    # element, which the store predicate throws away. This keeps every input
    # load a full unpredicated vector access on all tiles, instead of making all
    # 1953 tiles of an N=500000 row pay for the one ragged tile at the end.
    if EVEN_M:
        offs_m_c = offs_m
        m_inb = tl.full([BLOCK_M], 1, tl.int1)
    else:
        offs_m_c = tl.minimum(offs_m, M - 1)
        m_inb = offs_m < M
    if EVEN_N:
        offs_n_c = offs_n
        n_inb = tl.full([BLOCK_N], 1, tl.int1)
    else:
        offs_n_c = tl.minimum(offs_n, N - 1)
        n_inb = offs_n < N

    ks_v = tl.load(ks_ptr + offs_m_c)
    ke_v = tl.load(ke_ptr + offs_m_c)

    # [M, N] can exceed int32: 64-bit scalar row base + 32-bit in-tile offsets.
    out_ptrs = (out_ptr + m0.to(tl.int64) * N) + (rm[:, None] * N + offs_n[None, :])
    store_mask = m_inb[:, None] & n_inb[None, :]

    # Tile lies entirely outside every row's window: emit -inf, skip the GEMM.
    # (Clamped ks/ke on pad rows only ever make this test more conservative.)
    if (n0 >= tl.max(ke_v)) | (n0 + BLOCK_N <= tl.min(ks_v)):
        tl.store(out_ptrs, tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32),
                 mask=store_mask)
        return

    k_tile = tl.load(k_ptr + offs_n_c[:, None] * D + offs_d[None, :])   # [BLOCK_N, D]
    kt = tl.trans(k_tile)                 # [D, BLOCK_N]: k-major, the mma B layout
    scale = tl.load(kscale_ptr + offs_n_c)

    q_base = q_ptr + offs_m_c[:, None] * (H * D) + offs_d[None, :]
    if W_T:
        w_base = w_ptr + offs_m_c
        w_step = M
    else:
        w_base = w_ptr + offs_m_c * H
        w_step = 1

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    if tl.min(scale) >= 0.0:
        # Fast path: relu(s*c)*w == c*relu(s)*w for c >= 0, so hoist the scale.
        for h in range(0, H):
            qh = tl.load(q_base + h * D)
            s = tl.dot(qh, kt, out_dtype=tl.float32)
            w = tl.load(w_base + h * w_step)
            acc += tl.maximum(s, 0.0) * w[:, None]
        acc = acc * scale[None, :]
    else:
        for h in range(0, H):
            qh = tl.load(q_base + h * D)
            s = tl.dot(qh, kt, out_dtype=tl.float32)
            w = tl.load(w_base + h * w_step)
            acc += tl.maximum(s * scale[None, :], 0.0) * w[:, None]

    valid = (offs_n[None, :] >= ks_v[:, None]) & (offs_n[None, :] < ke_v[:, None])
    tl.store(out_ptrs, tl.where(valid, acc, float("-inf")), mask=store_mask)


# --------------------------------------------------------------------------- #
# Config selection
# --------------------------------------------------------------------------- #
# (BLOCK_M, BLOCK_N, num_warps, num_stages, M_FAST), chosen by measured sweep.
# BLOCK_M=32 / BLOCK_N=256 wins nearly everywhere: the head loop needs two live
# [BLOCK_M, BLOCK_N] fp32 tiles (per-head scores + running output), so a short M
# keeps register pressure low, while a long N amortises the q tile that each
# program re-reads for all H heads.
#   * tiny problems cannot fill 188 SMs with BLOCK_N=256, so they use 128.
#   * M_FAST (m-index varies fastest) only pays off on the largest cells, where
#     it keeps one K tile resident across all M tiles; below that, n-fast keeps
#     the q tile hot instead, which is worth 1-5%.
_CFG_TINY = (32, 128, 4, 2, 0)
_CFG_MID = (32, 256, 4, 3, 0)
_CFG_HUGE = (32, 256, 4, 3, 1)

_TINY_MAX_ELEMS = 1 << 23
_HUGE_MIN_ELEMS = 1 << 29

# Below this many output elements the [H, M] weight transpose costs more launch
# overhead than the ~4% it buys in the head loop.
_WT_MIN_ELEMS = 1 << 25


def _env_cfg():
    s = os.environ.get("PREFILL_TRITON_CFG")
    if not s:
        return None
    parts = [int(x) for x in s.replace(" ", "").split(",")]
    assert len(parts) == 5, "PREFILL_TRITON_CFG=BM,BN,warps,stages,mfast"
    return tuple(parts)


def _pick_config(M: int, N: int):
    cfg = _env_cfg()
    if cfg is not None:
        return cfg
    elems = M * N
    if elems < _TINY_MAX_ELEMS:
        return _CFG_TINY
    if elems < _HUGE_MIN_ELEMS:
        return _CFG_MID
    return _CFG_HUGE


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def fp8_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """logits [M, N] fp32; -inf outside [ks[m], ke[m])."""
    assert q.is_cuda and q.dim() == 3
    # slices from the serving path are contiguous; anything else must fail
    # loudly, not read garbage (see the decode-kernel page-stride bug).
    assert q.is_contiguous() and k.is_contiguous() and weights.is_contiguous()
    assert ks.is_contiguous() and ke.is_contiguous()
    M, H, D = q.shape
    N = k.shape[0]
    assert k.shape[1] == D

    k_scale = k_scale.reshape(-1)
    out = torch.empty((M, N), dtype=torch.float32, device=q.device)

    BLOCK_M, BLOCK_N, num_warps, num_stages, m_fast = _pick_config(M, N)

    use_wt = (M * N) >= _WT_MIN_ELEMS and os.environ.get("PREFILL_TRITON_WT") != "0"
    w_arg = weights.t().contiguous() if use_wt else weights

    num_m = triton.cdiv(M, BLOCK_M)
    num_n = triton.cdiv(N, BLOCK_N)

    _prefill_logits_kernel[(num_m * num_n,)](
        q, k, k_scale, w_arg, ks, ke, out,
        M, N, num_m, num_n,
        H=H, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        M_FAST=m_fast,
        EVEN_M=(M % BLOCK_M == 0),
        EVEN_N=(N % BLOCK_N == 0),
        W_T=use_wt,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
